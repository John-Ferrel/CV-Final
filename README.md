# CV-Final



The final Project of CV, including 3 tasks.

Author: Hao Xiao, Bin Li

## Best Model
Download URL: https://drive.google.com/drive/folders/1IxhMYnqWGLe52mJV2nLEbODqnqY7qDK7?usp=sharing

- Task1: 权重文件-CV-PJ-task3-1.zip
- Task2: 权重文件-CV-PJ-3-2.zip
- Task3: 权重文件-CV-PJ-3-3.zip


## How to Train/Test
 ## Task1
 Run `SimCLR.py` ->  `ImageNet_ResNet18.py` ->  `CIFAR100_ResNet18.py`  for trainning&testing three sub-tasks.

 ## Task2
 Set some hyperparameters in `setting.py`.
 Then train/test in `train.py`.

 ## Task3

Firstly run `extractFrames.py` for extracting pictures from any videos.

`sparse/0` contains 3D reconstruction result of these pictures.

`LLFF-master/imgs2poses.py` helps to generate LLFF dataset.

Finally `nerf-pytorch-master/run_nerf.py` train  the NeRF.