#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from pprint import pprint, pformat
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os, datetime, pdb, shutil, random
import argparse
import utils

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    RandAffined,
    SpatialPadd,
    ToTensord,
    MapLabelValued
)
from monai.inferers import SliceInferer
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import numpy as np
import monai

from networks.UNet_BN import UNet_MSD_liver_BN
from networks.UNet_NABN import UNet_MSD_liver_NABN

class Config(object):
    def __init__(self):
        
        self.net_name = "UNet_MSD_liver_BN" # Unet_MSD_liver_BN | Unet_MSD_liver_NABN
        
        self.ID = 'UNet_MSD_liver_BN'
        self.log_base_dir = './'
        self.summaryWrite_base_dir = './'
        self.checkpoint_dir = './'
        self.dataset = "Task03_Liver"
        
        self.manual_seed = 0
        self.data_root_dir = '/root/workspace/project/tutorials/monai/data/MSD'
        self.HU_min_max = [-310, 390]
        self.spacing=[1, 1, 3]
        self.gpus = "0"
        self.channel = 1
        self.num_classes = 2
        self.optimizer = "Adam" # optimizer is 'Adam', or 'SGD'
        self.lr = 1e-4
        self.train_batch_size = 4
        self.test_batch_size = 1
        self.epochs = 200
        self.num_workers = 2

# ------------ logger ------------

def get_logger(config):
    log_path = os.path.join(config.log_base_dir, 'logs', config.dataset, config.net_name, '{}.log'.format(config.ID))
    if os.path.exists(log_path):
        delete_log = input("The log file %s exist, delete it or not (y/n) \n"%(log_path))
        if delete_log in ['y', 'Y']:
            os.remove(log_path)
        else:
            log_path = os.path.join(config.log_base_dir, 'logs', config.dataset, config.net_name, '{}_{}.log'.format(config.ID, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
    print(f'log_path is {log_path}')
    logger = utils.get_logger(log_path)
    return logger


# In[ ]:


def get_SummaryWriter(config):
    writer_path = os.path.join(config.summaryWrite_base_dir, 'runs/scalar_example', config.dataset, config.net_name, config.ID)
    if os.path.exists(writer_path):
        delete_summary = input("The summaries folder %s exist, delete it or not (y/n) \n"%(writer_path))
        if delete_summary in ['y', 'Y']:
            shutil.rmtree(writer_path)
        else:
            writer_path = os.path.join(config.summaryWrite_base_dir, 'runs/scalar_example', config.dataset, config.net_name, config.ID+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f'writer_path is {writer_path}')
    writer = SummaryWriter(writer_path)
    return writer


# In[ ]:


def set_seed(config):
    if config.manual_seed is None:
        config.manual_seed = random.randint(1, 10000)
    np.random.seed(config.manual_seed)
    random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)
    torch.cuda.manual_seed(config.manual_seed)
    torch.cuda.manual_seed_all(config.manual_seed)


# In[ ]:


def train(model, data_loader, device, criterion, optimizer, epoch, writer):
    model.train()
    epoch_loss = []
    step = 0
    time_start = datetime.datetime.now()
    with tqdm(len(data_loader)) as pbar:
        for batch_data in data_loader:
            step += 1
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device),)
            inputs = torch.squeeze(inputs, dim=4)
            labels = torch.squeeze(labels, dim=4)
            optimizer.zero_grad()
            outputs = model(inputs)
#             loss = loss_function(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        
            pbar.update(1)
            pbar.set_description("Epoch: %d, Batch %d/%d, Train batch loss: %.4f"%
                                 (epoch, step, len(data_loader), np.mean(loss.item())))
    time_end = datetime.datetime.now()
    avg_loss = np.mean(epoch_loss)
    
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    time_spend = (time_end-time_start).seconds
    return avg_loss, time_spend


# In[ ]:


def test(model, data_loader, device, criterion, epoch, writer, mode = "validation"):
    model.eval()
    epoch_loss = []
    step = 0
    time_start = datetime.datetime.now()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        for val_data in data_loader:
            step += 1
            val_inputs, val_labels = (val_data["image"].to(device),val_data["label"].to(device),)
            val_inputs = val_inputs.permute(0, 1, 4, 2, 3)
            axial_inferer = SliceInferer(roi_size=(256, 256), sw_batch_size=16, cval=-1, progress=True)
            val_outputs = axial_inferer(val_inputs, model) # torch.Size([1, 2, 25, 512, 402])
            val_outputs = val_outputs.permute(0, 1, 3, 4, 2) # torch.Size([1, 2, 512, 402, 25])
            
            val_outputs_4loss = torch.squeeze(val_outputs.permute(4, 1, 2, 3, 0), dim=4)
            val_labels_4loss = torch.squeeze(val_labels.permute(4, 1, 2, 3, 0), dim=4)
            loss = criterion(val_outputs_4loss, val_labels_4loss)
            epoch_loss.append(loss.item())
            
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
        
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        
    time_end = datetime.datetime.now()
    avg_loss = np.mean(epoch_loss)
    
    writer.add_scalar('{}/epoch_dice'.format(mode), metric, epoch)
    writer.add_scalar('{}/epoch_loss'.format(mode), avg_loss, epoch)
    time_spend = (time_end-time_start).seconds
    return metric, avg_loss, time_spend


# In[ ]:


def log_best_acc(val_metric_list, test_metric_list, cur_epoch_idx, logger, state, save_path, save_model=True):
    # metric is dice —— 越大越好
    if len(val_metric_list) == 0:
        return
    else:
        best_val_idx = np.argmax(val_metric_list)
        best_val_metric = val_metric_list[best_val_idx]
        test_metric = test_metric_list[best_val_idx]
        if best_val_idx == cur_epoch_idx:
            logger.info("Epoch: %d, Validation mean dice increased to %.6f. Current Test mean dice is: %.6f"%(cur_epoch_idx, best_val_metric, test_metric))
            if save_model:
                dir_path = os.path.dirname(save_path)  # get parent path
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(state, save_path)
                logger.info("Model saved in file: %s"%(save_path))
        else:
            logger.info("Epoch: %d, Validation mean dice didn't increase. Best Validation mean dice is %.6f in epoch %d. Corresponding Test mean dice is %.6f."%(cur_epoch_idx, best_val_metric, best_val_idx, test_metric))

def get_transform(config):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config.HU_min_max[0], a_max=config.HU_min_max[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=config.spacing, mode=("bilinear", "nearest")),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(256, 256, 1),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(256, 256, 1),
                rotate_range=(0, 0, np.pi/15),
                scale_range=(0.1, 0.1, 0.1)),
            MapLabelValued(keys=["label"], orig_labels=[0,1,2], target_labels=[0,1,1]),
            ToTensord(keys=["image", "label"])
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"], a_min=config.HU_min_max[0], a_max=config.HU_min_max[1],
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=config.spacing, mode=("bilinear", "nearest")),
            MapLabelValued(keys=["label"], orig_labels=[0,1,2], target_labels=[0,1,1]),
            ToTensord(keys=["image", "label"]),
            
#             CropForegroundd(select_fn=select_fg, keys=["image", "label"], source_key="label"), # 仅使用前景图像预测
#             SpatialPadd(keys=["image", "label"], spatial_size=val_size), # 至少将抠出来的前景样本补零至val_size
        ]
    )
    return train_transforms, val_transforms


# In[ ]:


def get_dataloader(config, train_transforms, val_transforms):
    data_dir = os.path.join(config.data_root_dir, config.dataset)
    print(f'data_dir is {data_dir}')
    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files, test_files = data_dicts[:-60], data_dicts[-60:-30], data_dicts[-30:]
    
#     train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=8)
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.train_batch_size, shuffle=True, num_workers=8)
    
#     val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=8)
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=config.test_batch_size, num_workers=8)
    
#     test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=8)
    test_ds = monai.data.Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=config.test_batch_size, num_workers=8)
    
    return train_loader, val_loader, test_loader


# In[ ]:


def calculate_params(net, logger):
    num_params = sum([param.nelement() for param in net.parameters()])
    logger.info("Total number of parameters: {}".format(num_params))
    logger.info('Model architectures:\n{}'.format(net))
    logger.info('Parameters and size:')
    for name, param in net.named_parameters():
        logger.info('{}: {}'.format(name, list(param.size())))

def main():
    config = Config()
    logger = get_logger(config)
    logger.info(pformat(config.__dict__))
    writer = get_SummaryWriter(config)
    set_seed(config)
    set_determinism(seed=config.manual_seed)
    train_transforms, val_transforms = get_transform(config)
    train_loader, val_loader, test_loader = get_dataloader(config, train_transforms, val_transforms)
    gpus = range(len(config.gpus.split(",")))
    device = torch.device("cuda:{}".format(gpus[0]))
    net = globals()[config.net_name](config.channel, config.num_classes)
    criterion = DiceLoss(to_onehot_y=True, softmax=True)

    torch.backends.cudnn.deterministic=True # this item has impact on the deterministic results when the optimizers are adaptive ones
    torch.cuda.set_device(gpus[0]) # set the default device in order to allocate bernoulli variable on GPU
    if len(gpus) > 1:
        net = nn.DataParallel(net, gpus)
    net.to(device)
    if config.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        use_scheduler = False
    elif config.optimizer == "Adadelta":
        optimizer = optim.Adadelta(net.parameters(), lr = 1.0)
        use_scheduler = False
    elif config.optimizer == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        use_scheduler = True
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_scheduler, gamma=config.lr_gamma)
    else:
        raise("Unknown optimizer %"%(config.optimizer))
    calculate_params(net, logger)
    
    val_dice_list, test_dice_list = [], []
    for epoch in range(config.epochs):
        if use_scheduler:
            lr_scheduler.step()
            logger.info("Epoch: %d, Learning rate: %.5f"%(epoch, lr_scheduler.get_lr()[0]))
        train_loss, train_time = train(net, train_loader, device, criterion, optimizer, epoch, writer)
        logger.info("Epoch: %d, Train Time: %s(s), Train Loss: %.4f"%(epoch, train_time, train_loss))

        test_dice, test_loss, test_time = test(net, test_loader, device, criterion, epoch, writer, mode = "test")

        if val_loader is not None:
            val_dice, val_loss, val_time = test(net, val_loader, device, criterion, epoch, writer, mode = "validation")
            logger.info("Epoch: %d, Validation time: %s(s), Validation Loss: %.4f, Validation Dice: %.4f"%(epoch, val_time, val_loss, val_dice))
            val_dice_list.append(val_dice)
        else:
            val_dice_list.append(test_dice)

        logger.info("Epoch: %d, Test time: %s(s), Test Loss: %.4f, Test Dice: %.4f"%(epoch, test_time, test_loss, test_dice))
        test_dice_list.append(test_dice)

        model_out_path = os.path.join(config.checkpoint_dir, "checkpoint", config.dataset, config.net_name, '{}_epoch{}.pth'.format(config.ID, epoch))
        log_best_acc(val_dice_list, test_dice_list,  epoch, logger, 
                     net,
                      model_out_path, save_model=True)

main()

