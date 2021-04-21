# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, PolygonMasks
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from models.backbone import Joiner
#from models.detr import DETR, SetCriterion
from models.deformable_detr import DeformableDETR, SetCriterion
from models.matcher import HungarianMatcher
from models.position_encoding import PositionEmbeddingSine
#from models.transformer import Transformer
from models.deformable_transformer import DeformableTransformer

from models.segmentation import DETRsegm, PostProcessPanoptic, PostProcessSegm
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor
from datasets.coco import convert_coco_poly_to_mask

__all__ = ["DeformableDetr"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        #self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels
        #self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels
        #backbone_shape[list(backbone_shape.keys())[-1]].channels
        self.num_channels = [512, 1024, 2048]

#        self.strides[-1] = self.feature_strides[-1] // 2
#        print('stride: ', self.feature_strides[-1] // 2)
        self.feature_strides[-1] = self.feature_strides[-1] // 2
#         self.feature_strides[-1] // 2
        self.strides = self.feature_strides


#        print('feature_strides: ', self.feature_strides)
#        print('num_channels: ', self.num_channels)
#       return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}}
#        feature_strides:  [4, 8, 16, 32]
#        num_channels:  2048

#        feature_strides:  [4, 8, 16]
#        num_channels:  1024]

#feature_strides:  [8, 16, 32]
#num_channels:  2048]

    # ImageList
    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class DeformableDetr(nn.Module):
    """
    Implement Deformable-Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DeformableDETR.NUM_CLASSES
        self.mask_on = cfg.MODEL.MASK_ON
        hidden_dim = cfg.MODEL.DeformableDETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DeformableDETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DeformableDETR.NHEADS
        dropout = cfg.MODEL.DeformableDETR.DROPOUT
        dim_feedforward = cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DeformableDETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DeformableDETR.DEC_LAYERS
#        pre_norm = cfg.MODEL.DeformableDETR.PRE_NORM
        activation = cfg.MODEL.DeformableDETR.ACTIVATION
        return_intermediate_dec = cfg.MODEL.DeformableDETR.RETURN_INTERMEDIATE_DEC
        num_feature_levels = cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS
        dec_n_points = cfg.MODEL.DeformableDETR.DEC_N_POINTS
        enc_n_points = cfg.MODEL.DeformableDETR.ENC_N_POINTS
        two_stage = cfg.MODEL.DeformableDETR.TWO_STAGE
        two_stage_num_proposals = cfg.MODEL.DeformableDETR.TWO_STAGE_NUM_PPOPOSALS

        # Loss parameters:
        giou_weight = cfg.MODEL.DeformableDETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DeformableDETR.L1_WEIGHT
        deep_supervision = cfg.MODEL.DeformableDETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DeformableDETR.NO_OBJECT_WEIGHT

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        with_box_refine = cfg.MODEL.DeformableDETR.WITH_BOX_REFINE

#        transformer = Transformer(
#            d_model=hidden_dim,
#            dropout=dropout,
#            nhead=nheads,
#            dim_feedforward=dim_feedforward,
#            num_encoder_layers=enc_layers,
#            num_decoder_layers=dec_layers,
#            normalize_before=pre_norm,
#            return_intermediate_dec=deep_supervision,
#        )

        transformer = DeformableTransformer(
            nhead=nheads,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            d_model=hidden_dim,
            activation=activation,
            return_intermediate_dec=return_intermediate_dec,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            two_stage=two_stage,
            two_stage_num_proposals=num_queries,
        )

#        self.detr = DeformableDETR(
#            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision
#        )
        self.ddetr = DeformableDETR(
            backbone,
            transformer,
            num_classes=self.num_classes,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            aux_loss=deep_supervision,
            with_box_refine=with_box_refine,
            two_stage=two_stage,
        )

        if self.mask_on:
            frozen_weights = cfg.MODEL.DeformableDETR.FROZEN_WEIGHTS
            if frozen_weights != '':
                print("LOAD pre-trained weights")
                weight = torch.load(frozen_weights, map_location=lambda storage, loc: storage)['model']
                new_weight = {}
                for k, v in weight.items():
                    if 'ddetr.' in k:
                        new_weight[k.replace('ddetr.', '')] = v
                    else:
                        print(f"Skipping loading weight {k} from frozen model")
                del weight
                self.ddetr.load_state_dict(new_weight)
                del new_weight
            self.ddetr = DETRsegm(self.ddetr, freeze_detr=(frozen_weights != ''))
            self.seg_postprocess = PostProcessSegm

        self.ddetr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        if self.mask_on:
            losses += ["masks"]
        self.criterion = SetCriterion(
            #self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses,
            self.num_classes, matcher=matcher, weight_dict=weight_dict,
            losses=losses,
            #eos_coef=no_object_weight,
            focal_alpha=no_object_weight,
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)
        output = self.ddetr(images)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            targets = self.prepare_targets(gt_instances)
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            mask_pred = output["pred_masks"] if self.mask_on else None
            results = self.inference(box_cls, box_pred, mask_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
            if self.mask_on and hasattr(targets_per_image, 'gt_masks'):
                gt_masks = targets_per_image.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                new_targets[-1].update({'masks': gt_masks})
        return new_targets

    def inference(self, box_cls, box_pred, mask_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
            scores, labels, box_pred, image_sizes
        )):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            if self.mask_on:
                mask = F.interpolate(mask_pred[i].unsqueeze(0), size=image_size, mode='bilinear', align_corners=False)
                mask = mask[0].sigmoid() > 0.5
                B, N, H, W = mask_pred.shape
                mask = BitMasks(mask.cpu()).crop_and_resize(result.pred_boxes.tensor.cpu(), 32)
                result.pred_masks = mask.unsqueeze(1).to(mask_pred[0].device)

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images
