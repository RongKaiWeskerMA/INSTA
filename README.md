# INSTA: Learning Instance and Task-Aware Dynamic Kernels for Few Shot Learning

<p align="center">
  <img src="https://github.com/RongKaiWeskerMA/INSTA/blob/master/visual/concept.png" width="700" height="300">
</p>
  
This repository provides the implementation and demo of **Learning Instance and Task-Aware Dynamic Kernels for Few Shot Learning** on [Prototypical Network](https://arxiv.org/pdf/1703.05175.pdf). The dynamic enviroment of few-shot learning (FSL) requires a model capable of rapidly adapting to the novel tasks. Moreover, given the low-data regime of FSL, it requires the model to encode rich information for per-data sample. To tackle this problem, we propose to learn a dynamic kernel that is both **ins**tance and **t**ask-**a**ware: **INSTA** for each channel and spatial location of a feature map, given the task (episode) at hands. Beyond that, we further incorporate the inforamtion from the fequency domain to generate our dynamic kernel. 
<p align="center">
  <img src="https://github.com/RongKaiWeskerMA/INSTA/blob/master/visual/heatmap.png">
</p>

## Prerequisites
We use anaconda to manage the virtual environment. Please install the following packages to run this repository. If there is a "No module" error, please install the suggested packages according to the error message.
* python 3.8 
* [pytorch 1.7.0](https://pytorch.org/get-started/previous-versions/)
* torchvision 0.8.0
* torchaudio 0.7.0
* tqdm
* tensorboardX

## Dataset

### Tiered-ImageNet

Tiered-ImageNet is also a subset of the ImageNet. This dataset consists of 608 classes from 34 categories and is split into 351 classes from 20 categories for training, 97 classes from 6 categories for validation, and 160 classes from 8 categories for testing. You can download the processed dataset in this [repository](https://github.com/icoz69/DeepEMD). Once the dataset is downloaded, please move it to /data direcotry. Note that the images have been resized into 84x84.

### Mini-ImageNet
```Shell
├── data
    ├── Mini-ImageNet
        ├── split
            ├── train
            ├── validation
            ├── test
        ├── images 
            ├── im_0.jpg
            ├── im_1.jpg
            .
            .
            .
            ├── im_n.jpg
 
```

Mini-ImageNet is sampled from ImageNet. This dataset has 100 classes, with each having 600 samples. We follow the standard protocol to split the dataset into 64 training, 16 validation, and 20 testing classes. For downloading the corresponding split and data files, please refer to [this repository](https://github.com/Sha-Lab/FEAT).

### CUB

The CUB is a fine-grained dataset, which consists of 11,788 images from 200 different breeds of birds. We follow the standard settings, in which the dataset is split into 100/50/50 breeds for training, validation, and testing, respectively. For ResNet-12 backbone, please refer to [this repository](https://github.com/icoz69/DeepEMD) to split the datasset and for ResNet-18 backbone, please refer to [this repository ](https://github.com/imtiazziko/LaplacianShot).

### FC100

FC100 dataset is a variant of the standard CIFAR100 dataset, which contains images from 100 classes, with each class containing 600 samples. We follow the standard setting, where the dataset is split into 60/20/20 classes for training, validation and testing, respectively. For downloading and split the data, please refer to [eepEMD repository](https://github.com/icoz69/DeepEMD).

## Training

We provide the example command line for Tiered-ImageNet below:
```shell
$ python train_fsl.py --max_epoch 200 --model_class INSTA_ProtoNet --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance_1 1 --balance_2 0 --temperature 32 --temperature2 64 --lr 0.0002 --lr_mul 100 --lr_scheduler cosine --step_size 40 --gamma 0.5 --gpu 1 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1  --use_euclidean
```
```shell
$ python train_fsl.py --max_epoch 200 --model_class DDFNet --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance_1 1 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 30 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 0 --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean
 ```
 ## Acknowledgements
 We acknowledge the following repositories to provide valuable insight of our code construciton:

* [FEAT](https://github.com/Sha-Lab/FEAT)
* [DeepEMD](https://github.com/icoz69/DeepEMD)
* [Chen *etal*](https://github.com/wyharveychen/CloserLookFewShot)
* [FCANet](https://github.com/cfzd/FcaNet)
* [Fan *etal*](https://github.com/fanq15/FSOD-code)
