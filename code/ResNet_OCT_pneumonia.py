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

from networks.ResNet50_BN import ResNet_50_OCT_pneumonia_BN
from networks.ResNet50_NABN import ResNet_50_OCT_pneumonia_NABN
from networks.metrics import Confusion

from datasets import Cifar10_aug, OCTDataset, pneumoniaDataset

class Config(object):
    def __init__(self):
        
        self.net_name = "ResNet_50_OCT_pneumonia_BN"
        
        self.summary_name = 'ResNet_OCT_pneumonia' + '_' + self.net_name
        self.suffix = self.summary_name
        self.note = self.summary_name
        
        self.gpus = "0" # multiple gpu is "0, 1"
        self.dataset = "OCTDataset" # OCTDataset | pneumoniaDataset
        self.channel = 3
        self.width = 224
        self.height = 224
        self.num_classes = 4
        self.manual_seed = 6083
        
        self.optimizer = "Adam" # optimizer is 'Adam', or 'SGD'
        self.lr = 0.001 # SGD default lr is 0.1, Adam default 0.001
        
        self.lr_gamma = 0.1 # used in lr schedule
        self.lr_scheduler = [60, 100, 140]
        self.weight_decay = 0.0001 # 0.0001 | 2e-05
        self.momentum = 0.9
        
        self.batch_size = 16
        self.epochs = 200
        self.num_workers = 4

config = Config()


def log_best_acc(val_errors, test_errors, cur_epoch_idx, logger, state, save_path, save_model=True):
    if len(val_errors) == 0:
        return
    else:
        best_val_idx = np.argmin(val_errors)
        best_val_error = val_errors[best_val_idx]
        test_error = test_errors[best_val_idx]
        if best_val_idx == cur_epoch_idx:
            logger.info("Epoch: %d, Validation error decreased to %.6f. Current Test error is: %.6f"%(cur_epoch_idx, best_val_error, test_error))
            if save_model:
                dir_path = os.path.dirname(save_path)  # get parent path
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                torch.save(state, save_path)
                logger.info("Model saved in file: %s"%(save_path))
        else:
            logger.info("Epoch: %d, Validation Error didn't decrease. Best Validation Error is %.6f in epoch %d. Corresponding Test Error is %.6f."%(cur_epoch_idx, best_val_error, best_val_idx, test_error))


# In[ ]:


def train(model, data_loader, device, criterion, optimizer, epoch, writer):
    model.train()
    correct, total = 0, 0
    epoch_loss = []
    time_start = datetime.datetime.now()
    with tqdm(len(data_loader)) as pbar:
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs  = inputs.to(device)
            # inputs ---- torch.Size([16, 3, 224, 224])
            targets = targets.to(device)
            
            outputs = model(inputs)
                        
            loss = criterion(outputs, targets)
            loss.backward()
            epoch_loss.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()

            _, predicted = torch.max(outputs.detach(), 1)

            total += targets.size(0)

            correct += torch.sum(predicted.detach() == targets.detach())

            pbar.update(1)
            pbar.set_description("Epoch: %d, Batch %d/%d, Train loss: %.4f, Train error: %.4f"%(epoch, 
                                                                               batch_idx+1, len(data_loader), 
                                                                               np.mean(epoch_loss), float(total-correct)/total))
    time_end = datetime.datetime.now()
    error = float(total - correct) / total
    avg_loss = np.mean(epoch_loss)
    
    writer.add_scalar('train/epoch_error', error, epoch)
    writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    time_spend = (time_end-time_start).seconds
    return error, avg_loss, time_spend

def test(model, data_loader, device, num_classes, criterion, epoch, writer, mode = "validation"):
    model.eval()
    correct, total = 0, 0
    epoch_loss, epoch_logits, y_logits, outputs_softmax = [], [], [], []
    time_start = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # outputs --- torch.Size([16, 4])
            loss = criterion(outputs, targets)

            epoch_loss.append(loss.item())
            _, predicted = torch.max(outputs.detach(), 1)

            total += targets.size(0)
            correct += torch.sum(predicted.detach() == targets.detach())
            
            epoch_logits = epoch_logits + [idx for idx in predicted.detach().cpu().numpy()]
            y_logits = y_logits + [idx for idx in targets.detach().cpu().numpy()]

    confusion_mat, recall, precision, f1 = statistic(epoch_logits, y_logits, num_classes)
    if mode == "test":
        logger.info("%s Confusion matrix: \n %s \n Recall: %s \n Precision: %s \n F1: %s"%(mode, str(confusion_mat), str(recall), str(precision), str(f1)))
            
    time_end = datetime.datetime.now()
    error = float(total - correct) / total
    avg_loss = np.mean(epoch_loss)
    
    writer.add_scalar('{}/epoch_error'.format(mode), error, epoch)
    writer.add_scalar('{}/epoch_loss'.format(mode), avg_loss, epoch)
    time_spend = (time_end-time_start).seconds
    return error, avg_loss, time_spend

def statistic(pred_logits, y_logits, label_size=2):
    '''get recall, precision and f1
    '''
    confusion_mat = np.zeros((label_size, label_size))
    for i, sample in enumerate(pred_logits):
        confusion_mat[y_logits[i], pred_logits[i]] += 1
    c = Confusion(confusion_mat)
    acc = c.accuracy()
    recall_list = c.recall()
    precision_list = c.precision()
    f1_list = c.f1()      
    return confusion_mat, recall_list, precision_list, f1_list


# In[ ]:




# In[ ]:


log_path = os.path.join('logs', config.dataset, config.net_name, '{}.log'.format(config.suffix))
if os.path.exists(log_path):
    delete_log = input("The log file %s exist, delete it or not (y/n) \n"%(log_path))
    if delete_log in ['y', 'Y']:
        os.remove(log_path)
    else:
        log_path = os.path.join('logs', config.dataset, config.net_name, '{}_{}.log'.format(config.suffix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))

base_summary_path = './runs/scalar_example'
summary_dir = 'ResNet_OCT_pneumonia'
summary_ffp = os.path.join(base_summary_path, summary_dir, config.summary_name)
writer_path = summary_ffp
    
# writer_path = os.path.join("summaries", config.dataset, config.net_name, config.suffix)
if os.path.exists(writer_path):
    delete_summary = input("The summaries folder %s exist, delete it or not (y/n) \n"%(writer_path))
    if delete_summary in ['y', 'Y']:
        shutil.rmtree(writer_path)
    else:
        writer_path = os.path.join("summaries", config.dataset, config.net_name, config.suffix+"_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

if config.manual_seed is None:
    config.manual_seed = random.randint(1, 10000)
    
np.random.seed(config.manual_seed)
random.seed(config.manual_seed)
torch.manual_seed(config.manual_seed)
torch.cuda.manual_seed(config.manual_seed)
torch.cuda.manual_seed_all(config.manual_seed)

# os.environ['PYTHONHASHSEED'] = str(config.manual_seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


logger = utils.get_logger(log_path)
writer = SummaryWriter(writer_path)
logger.info(pformat(config.__dict__))


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
gpus = range(len(config.gpus.split(",")))
device = torch.device("cuda:{}".format(gpus[0]))

net = globals()[config.net_name](config.width, config.height, config.channel, config.num_classes)
    
# dataset = globals()[config.dataset](batch_size = config.batch_size, seed = config.manual_seed, num_workers = config.num_workers)

transform_train = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
         ])

transform_test = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
         ])

base_dir = './data/zk_OCT_data/dataList'
train_txt_dir = os.path.join(base_dir, 'train')
train_set = OCTDataset(train_txt_dir, transform_train)

val_txt_dir = os.path.join(base_dir, 'validation')
val_set = OCTDataset(val_txt_dir, transform_test)

test_txt_dir = os.path.join(base_dir, 'test')
test_set = OCTDataset(test_txt_dir, transform_test)

trainloader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
validloader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
testloader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        

num_params = sum([param.nelement() for param in net.parameters()])

logger.info('Model architectures:\n{}'.format(net))
logger.info('Parameters and size:')
for name, param in net.named_parameters():
    logger.info('{}: {}'.format(name, list(param.size())))
logger.info("Total number of parameters: {}".format(num_params))

torch.backends.cudnn.deterministic=True # this item has impact on the deterministic results when the optimizers are adaptive ones
torch.cuda.set_device(gpus[0]) # set the default device in order to allocate bernoulli variable on GPU
if len(gpus) > 1:
    net = nn.DataParallel(net, gpus)
net.to(device)

criterion = nn.CrossEntropyLoss()
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


# In[ ]:

val_errors, test_errors = [], []
for epoch in range(config.epochs):
    if use_scheduler:
        lr_scheduler.step()
        logger.info("Epoch: %d, Learning rate: %.5f"%(epoch, lr_scheduler.get_lr()[0]))
    train_error, train_loss, train_time = train(net, trainloader, device, criterion, optimizer, epoch, writer)
    logger.info("Epoch: %d, Train Time: %s(s), Train Loss: %.4f, Train Error: %.4f"%(epoch, train_time, train_loss, train_error))
    
    test_error, test_loss, test_time = test(net, testloader, device, config.num_classes, criterion, epoch, writer, mode = "test")
    test_errors.append(test_error)
    
    val_error, val_loss, val_time = test(net, validloader, device, config.num_classes, criterion, epoch, writer, mode = "validation")
    val_errors.append(val_error)
    
    logger.info("Epoch: %d, Validation time: %s(s), Validation Loss: %.4f, Validation Error: %.4f"%(epoch, val_time, val_loss, val_error))
    logger.info("Epoch: %d, Test time: %s(s), Test Loss: %.4f, Test Error: %.4f"%(epoch, test_time, test_loss, test_error))
    
    model_out_path = os.path.join("checkpoint", config.dataset, config.net_name, '{}_epoch{}.pth'.format(config.suffix, epoch))
    log_best_acc(val_errors, test_errors,  epoch, logger, net, model_out_path, save_model=True)