# ShoeRinsics: Shoeprint Prediction for Forensics with Intrinsic Decomposition

<!-- This repo contains the official Pytorch implementation of: -->

This is the official repo for our paper:

[ShoeRinsics: Shoeprint Prediction for Forensics with Intrinsic Decomposition](https://arxiv.org/abs/2205.02361)

[Samia Shafique](https://sites.google.com/site/samiashafique067/), [Bailey Kong](https://baileykong.com/), [Shu Kong](http://www.cs.cmu.edu/~shuk/), and [Charless Fowlkes](https://www.ics.uci.edu/~fowlkes/)

<!-- CVPR 2021 (oral)

For more details, please check our [project website](https://www.ics.uci.edu/~yunhaz5/cvpr2021/cpp.html) -->

### Abstract
Shoe tread impressions are one of the most common types of evidence left at crime scenes. However, the utility of such evidence is limited by the lack of databases of footwear impression patterns that cover the huge and growing number of distinct shoe models. We propose to address this gap by leveraging shoe tread photographs collected by online retailers. The core challenge is to predict the impression pattern from the shoe photograph since ground-truth impressions or 3D shapes of tread patterns are not available. We develop a model that performs intrinsic image decomposition (predicting depth, normal, albedo and lighting) from a single tread photo. Our approach, which we term ShoeRinsics, combines domain adaptation and re-rendering losses in order to leverage a mix of fully supervised synthetic data and unsupervised retail image data. To validate model performance, we also collected a set of paired shoe-sole images and corresponding prints, and define a benchmarking protocol to quantify accuracy of predicted impressions. On this benchmark, ShoeRinsics outperforms existing methods for depth prediction and synthetic-to-real domain adaptation.

**Keywords**: Shoeprints, Forensic Evidence, Depth Prediction, Intrinsic Decomposition, and Domain Adaptation 

### Overview 

<p align="center">
    <img src='git/figures/architecture.png' width='500'/>
</p>

The flowchart of our method ShoeRinsics in training. The training data consists of synthetic images labeled with intrinsic components and unlabeled real images. Conceptually, ShoeRinsics incorporates intrinsic decomposition (right part) and domain adaptation (left part) to learn a depth predictor for real shoe-sole images. We use a renderer pre-trained on synthetic data to regularize intrinsic decomposition from which we obtain depth predictions. We find this works better than learning to predict depth only, presumably because intrinsic decomposition leverages extra supervision from synthetic data that helps depth prediction learning.


<!-- ### Reference
If you find our work useful in your research please consider citing our paper:
```
@inproceedings{zhao2021camera,
  title={Camera Pose Matters: Improving Depth Prediction by Mitigating Pose Distribution Bias},
  author={Zhao, Yunhan and Kong, Shu and Fowlkes, Charless},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15759--15768},
  year={2021}
}
```

### Contents
- [Requirments](#requirements)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained Models](#pretrained-models)


### Requirements
1. Python 3.6 with Ubuntu 16.04
2. Pytorch 1.1.0
3. Apex 0.1 (optional)

You also need other third-party libraries, such as numpy, pillow, torchvision, and tensorboardX (optional) to run the code. We use apex when training all models but it is not strictly required to run the code. 

### Dataset
We use InteriorNet and ScanNet in this project. The detailed data file lists are located in `dataset` folder where each file correspinds to one training/testing distribution (natural, uniform or restricted). Please download and extract the appropriate files before training.
####  Dataset Structure (e.g. interiorNet_training_natural_10800)
```
interiorNet_training_natural_10800
    | rgb
        | rgb0.png
        | ...
    | depth
        | depth0.png
        | ...
    cam_parameter.txt
```
`cam_parameter.txt` contains the intrinsics and camera pose for each sample in the subset. Feel free to sample your own distribution and train with your own data. 

### Training
All training steps use one common `train.py` file so please make sure to comment/uncomment for training with CPP, PDA, or CPP + PDA.
```bash
CUDA_VISIBLE_DEVICES=<GPU IDs> python train.py \
  --data_root=<your absolute path to InteriorNet or ScanNet> \
  --training_set_name=interiorNet_training_natural_10800 \
  --testing_set_name=interiorNet_testing_natural_1080 \
  --batch_size=12 --total_epoch_num=200 --is_train --eval_batch_size=10
```
`batch_size` and `eval_batch_size` are flexible to change given your working environment. Feel free to swap `interiorNet_training_natural_10800` and `interiorNet_testing_natural_1080` to train and test on different distributions.

### Evaluations
Evaluate the final results
```bash
CUDA_VISIBLE_DEVICES=<GPU IDs> python train.py \
  --data_root=<your absolute path to InteriorNet or ScanNet> \
  --training_set_name=interiorNet_training_natural_10800 \
  --testing_set_name=interiorNet_testing_natural_1080 \
  --eval_batch_size=10
``` 
If you want to evaluate with your own data, please create your own testing set with the dataset structure described above.

### Pretrained Models
Pretrained models will be uploaded soon. -->

### Questions
Please feel free to email me at (sshafiqu [at] ics [dot] uci [dot] edu) if you have any questions.
