# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_ddetr_config(cfg):
    """
    Add config for DETR.
    """
    cfg.MODEL.DeformableDETR = CN()
    cfg.MODEL.DeformableDETR.NUM_CLASSES = 80

    # For Segmentation
    cfg.MODEL.DeformableDETR.FROZEN_WEIGHTS = ''

    # LOSS
    cfg.MODEL.DeformableDETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DeformableDETR.L1_WEIGHT = 5.0
    cfg.MODEL.DeformableDETR.DEEP_SUPERVISION = True
    #cfg.MODEL.DeformableDETR.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DeformableDETR.NO_OBJECT_WEIGHT = 0.25

    # TRANSFORMER
    cfg.MODEL.DeformableDETR.NHEADS = 8
    cfg.MODEL.DeformableDETR.DROPOUT = 0.1
    cfg.MODEL.DeformableDETR.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DeformableDETR.ENC_LAYERS = 6
    cfg.MODEL.DeformableDETR.DEC_LAYERS = 6
    cfg.MODEL.DeformableDETR.ACTIVATION = 'relu'
    cfg.MODEL.DeformableDETR.HIDDEN_DIM = 256

    cfg.MODEL.DeformableDETR.RETURN_INTERMEDIATE_DEC = True
    cfg.MODEL.DeformableDETR.NUM_FEATURE_LEVELS = 4
    cfg.MODEL.DeformableDETR.DEC_N_POINTS = 4
    cfg.MODEL.DeformableDETR.ENC_N_POINTS = 4
    cfg.MODEL.DeformableDETR.TWO_STAGE = False
    cfg.MODEL.DeformableDETR.TWO_STAGE_NUM_PPOPOSALS = 100

#    cfg.MODEL.DeformableDETR.PRE_NORM = False

    cfg.MODEL.DeformableDETR.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DeformableDETR.WITH_BOX_REFINE = False
    cfg.MODEL.DeformableDETR.DILATION = True
#    return DeformableTransformer(
#            nhead=args.nheads,
#            dropout=args.dropout,
#            dim_feedforward=args.dim_feedforward,
#            num_encoder_layers=args.enc_layers,
#            num_decoder_layers=args.dec_layers,
#            d_model=args.hidden_dim,
#            activation="relu",
#            return_intermediate_dec=True,
#            num_feature_levels=args.num_feature_levels,
#            dec_n_points=args.dec_n_points,
#            enc_n_points=args.enc_n_points,

#            two_stage=args.two_stage,
#            two_stage_num_proposals=args.num_queries))

    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
