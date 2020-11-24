# Video Classification using PyTorch
## Installation

### Dependencies
torch
torchvision
torchsummary
tensorboard
opencv-python
numpy
matplotlib
moviepy
einops


### Clone the repository
```
git@github.com:hoel-bagard/PyTorch-Video-Classification.git
```

### Get some data and format it:

You need to split the data between two folders: "Train" and "Validation" (the names are hard coded). You then need to create a classes.names next to the Train and Validation folder, with the names of the classes (one per line). More details can be found [here](https://github.com/hoel-bagard/PyTorch-Video-Classification/wiki/Dataset)


## Config files
In the config folder of this repo you will find two config template files. You need to copy them and remove the "_template" part.

### DataConfig
Contains most of the parameters regarding the data. Most of the values in the template can be kept as they are. The 3 paths usually need to be modified for each training (`DATA_PATH`, `CHECKPOINT_DIR` & `TB_DIR`). 

### ModelConfig
Contains the parameters that influence training. The default values should work fine, but you can try to tweak them to get better results. For the `MAX_EPOCHS` value, usually around 400 or 600 epochs is enough, you will need to train at least once to get an idea for your particular dataset.

## Train
Once you have the environment all set up and your two config files ready, training an AI is straight forward. Just connect to the server of your choice and run the following command: 
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

Notes:
The whole code folder will be copied in the checkpoint directory, in order to always have code that goes with the checkpoints. This means that you should not put your virtualenv or checkpoint directory in the code folder.
`CUDA_VISIBLE_DEVICES` is used to select with GPU to use. Check that the one you plan to use is free before you use it by running the `nvidia-smi` command.
