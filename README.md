# Deformable DETR Working Repo
This repo makes slight changes to the original Deformable-DETR repo for easy training/finetuning purposes. And also addresses some errors. 

## Usage
1. First Go through [original-repo readme](https://github.com/fundamentalvision/Deformable-DETR) first for setup. 

NOTE : 
Do this Before the `Compiling CUDA operators` in above README.
```bash
# First check nvidia-driver exists
nvidia-smi
# Incase you are using gcloud(debian machine) and `nvidia-smi` command is not working, run `install-driver.sh` to fresh install nvidia-driver. Not sure if it works for other linux distros.
./install-driver.sh
```

2. After you have setup the environment, checkout [how_to.md](https://github.dev/robinnarsinghranabhat/Deformable-DETR/how_to.md)