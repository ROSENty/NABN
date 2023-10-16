# NABN

This repo holds the pytorch implementation of NABN:<br />

**Regularizing Deep Neural Networks for Medical Image Analysis with Augmented Batch Normalization.**

## Requirements
Python 3.8.8<br />
PyTorch==2.0.1<br />
MONAI version: 1.1.0<br />

## Usage
### 0. Installation
* Clone this repo
```
git clone git@github.com:ROSENty/NABN.git
cd NABN
```
### 1. Data Preparation
Before starting, datasets should be prepared.

Dataset name | Data source
--- | :---:
OCT dataset | [data](https://data.mendeley.com/datasets/rscbjbr9sj/3)
chest X-ray dataset | [data](https://data.mendeley.com/datasets/rscbjbr9sj/3)
MSD Liver | [data](http://medicaldecathlon.com/)

* Download and put these datasets in `data/`.
* For image classification on OCT and chest X-ray datasets, replace the data path in text file `pneumonia_data/dataList` and `zk_OCT_data/dataList`.
* For image segmentation on MSD Liver dataset, replace the `data_root_dir` in `UNet_MSD_liver.py`.

The folder structure of OCT dataset should be like

    data/zk_OCT/
    └── train
    |    └── CNV
    |        ├── CNV-5477211-11.jpeg
    |        ├── ...
    |    └── DME
    |        ├── DME-4643364-16.jpeg
    |        ├── ...
    |    └── DRUSEN
    |        ├── DRUSEN-3424668-31.jpeg
    |        ├── ...
    |    └── NORMAL
    |        ├── NORMAL-508852-11.jpeg
    |        ├── ...
    └── test
    |    └── CNV
    |        ├── ...
    |    └── DME
    |        ├── ...
    |    └── DRUSEN
    |        ├── ...
    |    └── NORMAL
    |        ├── ...


The folder structure of chest X-ray dataset should be like

    data/zk_pneumonia/
    └── train
    |    └── PNEUMONIA
    |        ├── BACTERIA-198200-0002.jpeg
    |        ├── ...
    |    └── NORMAL
    |        ├── NORMAL-7103127-0001.jpeg
    |        ├── ...
    └── test
    |    └── PNEUMONIA
    |        ├── ...
    |    └── NORMAL
    |        ├── ...

The folder structure of MSD Liver dataset should be like

    data/MSD/Task03_Liver/
    └── imagesTr
    |    ├── liver_0.nii.gz
    |    ├── ...
    └── labelsTr
    |    ├── liver_0.nii.gz
    |    ├── ...

### 2. Training and Evaluation
* For image classification on CIFAR-10, modify `net_name` in `ResNet_CIFAR.py`, and run
```
python ResNet_CIFAR.py
```

* For image classification on OCT and chest X-ray datasets, modify `net_name` and `dataset` in `ResNet_OCT_pneumonia.py`, and run
```
python ResNet_OCT_pneumonia.py
```

* For image segmentation on MSD Liver dataset, modify `net_name` in `UNet_MSD_liver.py`, and run
```
python UNet_MSD_liver.py
```











