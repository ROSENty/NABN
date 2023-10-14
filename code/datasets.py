import torch
import torchvision, os, pdb
import torchvision.transforms as transforms
import numpy as np
import random
# from dataloaders.single_imageset import Image_dataset
# import dataloaders.image_transforms as tr
from torch.utils.data import Dataset
import cv2
from PIL import Image


def _split_val(trainset, testset, valid_size, batch_size, seed, num_workers = 4):
    # previsous num_workers is 4, reduced to 2 to avoid memory problem.
    
    def _init__fn(worker_id):
        np.random.seed(seed)
        random.seed(seed)
        
    if valid_size:
        indices = torch.randperm(len(trainset))
        train_indices = indices[:len(indices) - valid_size]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_indices = indices[len(indices) - valid_size:]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
    
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                           pin_memory=False, 
                                           num_workers=num_workers, worker_init_fn=_init__fn)
        
        validloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                                           pin_memory=False, 
                                           num_workers=num_workers, worker_init_fn=_init__fn)
    
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           pin_memory=False,
                                           num_workers=num_workers, worker_init_fn=_init__fn, shuffle=True)
        validloader = None
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                      pin_memory=False,
                                      num_workers=num_workers, worker_init_fn=_init__fn)
    
    return trainloader, validloader, testloader

class Cifar10_aug(object):
#     def __init__(self, batch_size = 64, valid_size = 5000, seed=0, num_workers = 4, data_root = "/root/workspace/public_data/cifar10"):
    def __init__(self, batch_size = 64, valid_size = 5000, seed=0, num_workers = 4, data_root = "/root/workspace/project/BatchN_GroupN/data"):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
         ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
         ])
        
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform_test)
        
        self.trainloader, self.validloader, self.testloader = _split_val(trainset, testset, valid_size, batch_size, seed, num_workers)

class OCTDataset(Dataset):
    
    def __init__(self, txt_dir, transform=None):
        self.x_list, self.y_list = self.concate(txt_dir)
        self.transform = transform
        
    def __getitem__(self, index):
        if not os.path.exists(self.x_list[index]):
            print('------------------ self.x_list[index] is ', self.x_list[index])
            raise NameError
        img = cv2.imread(self.x_list[index])
        if type(img) is not np.ndarray:
            print('------------------ self.x_list[index] is ', self.x_list[index])
        img = Image.fromarray(img, mode='RGB')
        label = self.y_list[index]
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    
    def __len__(self):
        return len(self.y_list)
    
    def concate(self, txt_dir):
        x_list = []
        y_list = []
        label_dict = {'CNV': 0, 'DME': 1, 'DRUSEN': 2, 'NORMAL': 3}
        txt_fileNameList = os.listdir(txt_dir)
        for txt_fileName in txt_fileNameList:
            if txt_fileName.endswith('.txt'):
                print('txt_fileName is ', txt_fileName)
                for key in label_dict.keys():
                    if key in txt_fileName:
                        label = label_dict[key]
                print('label is ', label)
                print('label type is ', type(label))
                with open(os.path.join(txt_dir, txt_fileName), 'r') as f:
                    for x in f.readlines():
                        x = x.strip('\n')
                        x_list.append(x)
                        y_list.append(label)
        return x_list, y_list
    
class pneumoniaDataset(Dataset):
    
    def __init__(self, txt_dir, transform=None):
        self.x_list, self.y_list = self.concate(txt_dir)
        self.transform = transform
        
    def __getitem__(self, index):
        if not os.path.exists(self.x_list[index]):
            print('------------------ self.x_list[index] is ', self.x_list[index])
            raise NameError
        img = cv2.imread(self.x_list[index])
        if type(img) is not np.ndarray:
            print('------------------ self.x_list[index] is ', self.x_list[index])
        img = Image.fromarray(img, mode='RGB')
        label = self.y_list[index]
        if self.transform is not None:
            img = self.transform(img) 
        return img, label
    
    def __len__(self):
        return len(self.y_list)
    
    def concate(self, txt_dir):
        x_list = []
        y_list = []
        label_dict = {'NORMAL': 0, 'PNEUMONIA': 1}
        txt_fileNameList = os.listdir(txt_dir)
        for txt_fileName in txt_fileNameList:
            if txt_fileName.endswith('.txt'):
                print('txt_fileName is ', txt_fileName)
                for key in label_dict.keys():
                    if key in txt_fileName:
                        label = label_dict[key]
                print('label is ', label)
                print('label type is ', type(label))
                with open(os.path.join(txt_dir, txt_fileName), 'r') as f:
                    for x in f.readlines():
                        x = x.strip('\n')
                        x_list.append(x)
                        y_list.append(label)
        return x_list, y_list