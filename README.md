# Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition
This is the code repository for ["Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition"](https://openaccess.thecvf.com/content/CVPR2022/html/Huang_Task-Adaptive_Negative_Envision_for_Few-Shot_Open-Set_Recognition_CVPR_2022_paper.html) (accepted by CVPR 2022).
 

## Installation
This repo is tested with Python 3.6, Pytorch 1.8, CUDA 10.1. More recent versions of Python and Pytorch with compatible CUDA versions should also support the code. 


## Data Preparation
MiniImageNet image data are provided by [RFS](https://github.com/WangYueFt/rfs), available at [DropBox](https://www.dropbox.com/sh/6yd1ygtyc3yd981/AABVeEqzC08YQv4UZk7lNHvya?dl=0). We also provide the word embeddings for the class names [here](https://drive.google.com/file/d/1CpF3M_qySCBhIWOSURIT_LpA1B61tsFb/view?usp=sharing). For TieredImageNet, we use the image data and word embeddings provided by [AW3](https://github.com/ServiceNow/am3), available at [GoogleDrive](https://drive.google.com/file/d/1Letu5U_kAjQfqJjNPWS_rdjJ7Fd46LbX/view). Download and put them under your <*data_dir*>.


## Pre-trained models
We provide the pre-trained models for TieredImageNet and MiniImageNet, which can be downloaded [here](https://drive.google.com/drive/folders/1mj8j5ZChRFLcYMBWEsBBhst8uQTOz_WJ?usp=sharing). Save the pre-trained model to <*pretrained_model_path*>.

## Training 
An example of training command for 5-way 1-shot FSOR:
```
python train.py --dataset <dataset> --logroot <log_root>  --data_root <data_dir> \ 
                --n_ways 5  --n_shots 1 \
                --pretrained_model_path <pretrained_model_path> \
                --featype OpenMeta \
                --learning_rate 0.03 \
                --tunefeat 0.0001 \
                --tune_part 4 \
                --cosine \
                --base_seman_calib 1 \
                --train_weight_base 1 \
                --neg_gen_type semang                 
```

## Testing
An example of testing command for 5-way 1-shot FSOR:
```
python test.py --dataset <dataset>  --data_root <data_dir> \
               --n_ways 5  --n_shots 1 \
               --pretrained_model_path <pretrained_model_path> \
               --featype OpenMeta \
               --test_model_path <test_model_path> \
               --n_test_runs 1000 \
               --seed <seed> 
```

## Pre-training
We also provide the code for the pre-training stage under `pretrain` folder. An example of running command for pre-training on miniImageNet:
```
python batch_process.py --featype EntropyRot --learning_rate 0.05
```

## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@InProceedings{Huang_2022_CVPR,
    author    = {Huang, Shiyuan and Ma, Jiawei and Han, Guangxing and Chang, Shih-Fu},
    title     = {Task-Adaptive Negative Envision for Few-Shot Open-Set Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {7171-7180}
}
```


## Acknowledgement
Our code and data are based upon [RFS](https://github.com/WangYueFt/rfs) and [AW3](https://github.com/ServiceNow/am3). 