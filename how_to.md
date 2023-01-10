## 1. Dataset Format
Data should be in COCO format. You might need to slightly restructure your 
dataset in format below : 

Keep Training images at : 
`DATA_DIR/train/images` <br>
Keep Training images annotations/labels at : `DATA_DIR/train/images/train.json` <br>
Keep Validation images at : `DATA_DIR/valid/images` <br>
Keep Validation images annotations/labels at : `DATA_DIR/train/images/valid.json`

( To setup paths differently, just do changes on `datasets/coco.py`. Inside the 
`def build(image_set, args):`).

## 2. Training Notes:
Assuming DATA_DIR as `custom_files`.

Then we can finetune over a trained model as : 

`python -u main.py --output_dir exps/iter_refine/ --with_box_refine --two_stage --resume  ./saved_models/r50_deformable_detr_plus_iterative_bbox_refinement_plus_plus_two_stage-checkpoint.pth --coco_path ./custom_files --num_classes=3`

**Important Model Flags** : 
- `coco_path` : this will be our `DATA_DIR`
- `output_dir` : this will be where model will be saved. 

- `resume`  : this flag will continue finetuning from the supplied model. Checkout available models in the Original Deformable-DETR repo. Or given enough dataset, we could even train our own model from scratch. 

- `num_classes` : 
  Deformable DETR is originally trained on 91 classes. Suppose, to finetune with  2 classes say, yes-checkbox and no-checkbox. 

  **Set the `num_classes` to 3 (Total Labels + 1). Plus 1 is done to account for no-object class.**  

   This way, Last linear layer will output 3 vectors instead of original 91 vectors. And during model-loading, 
   weights of last linear layer will be discarded.


## 3. Inference Notes
To infer using the trained model and visualize, Check the notebook : `inference.ipynb`.

## 4. For Gcloud users [Extra]
In gcloud, using `jupyter-notebook` or `jupyter-lab` would be beneficial.
To setup jupyter-lab, these are the steps :
```bash
## make changes to instance-name, region , project name e.t.c as necessary 
gcloud beta compute ssh --zone "region_name" "instance_name" --project "project_name" -- -L 8888:localhost:8888

# inside the remote server
conda activate your_detr_environment
conda install notebook
conda install jupyterlab
conda install ipykernel
python -m ipykernel install --user --name=name_of_kernel
# finally open the jupyter lab server
jupyter lab --no-browser --port=8888 --allow-root
# Now, click on link provided in the standard output below this line :
# To access the server, open this file in a browser: ...
```
