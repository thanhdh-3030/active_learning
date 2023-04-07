
import os
import torch
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
# import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from pprint import pprint
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from shutil import copyfile
from copy import deepcopy
# import torch_optimizer as optim
from sklearn.model_selection import train_test_split

# Apply WanDB
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning import  seed_everything

# Commented out IPython magic to ensure Python compatibility.
# os.environ["WANDB_CACHE_DIR"] = "/mnt/sda/shadow_user/tmp/"

# os.environ["WANDB_DIR"] = "/mnt/sda/shadow_user/tmp/"

# os.environ["WANDB_SILENT"] = "True"
# %env JOBLIB_TEMP_FOLDER=/tmp

use_wandb = True
wandb_key = "416ff8e8f97b3ca056e121705709bec3d83e929b"
wandb_project = "Polyp Active Learning1"
# wandb_entity = "ssl-online"
# wandb_name = "IC (1)"
# wandb_group = "Down 400 dist"
# wandb_dir = "./wandb"

# wandb.login(key=wandb_key)

"""## Setup some constance"""
n_drops = 5
trainsize = 352
max_epochs = 100
total_budget_size = 80
device = 'cuda:0'
device_ln = [int(device.split(':')[-1])]
seed = 2022

# Setup random seed

def set_seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    
set_seed_everything(seed)

class ActiveDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_paths=[], gt_paths=[], trainsize=352, transform=None):
        self.trainsize = trainsize
        self.images = image_paths
        self.masks = gt_paths
        self.size = len(self.images)

        self.transform = transform
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        mask = self.binary_loader(self.masks[index])
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask / 255
            
        sample = dict(image=image, mask=mask.unsqueeze(0), image_path=self.images[index], mask_path=self.masks[index], index=index)
        
        return sample
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.Resampling.BILINEAR)
            return np.array(img.convert('RGB'))

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.Resampling.NEAREST)
            img = np.array(img.convert('L'))
            return img

    def __len__(self):
        return self.size
class ActivePolybDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_paths=[],trainsize=352, transform=None):
        self.trainsize = trainsize
        self.images = image_paths
        self.masks = [image_path.replace('image','mask') for image_path in image_paths]
        self.size = len(self.images)

        self.transform = transform
        
    def __getitem__(self, index):
        image_path=self.images[index]
        mask_path=self.masks[index]
        image = self.rgb_loader(image_path)
        mask = self.binary_loader(mask_path)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            mask = mask / 255
            
        sample = dict(image=image, mask=mask.unsqueeze(0), image_path=self.images[index], mask_path=self.masks[index], 
                      index=index
                      )
        
        return sample
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.Resampling.BILINEAR)
            return np.array(img.convert('RGB'))

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f).resize((trainsize, trainsize), Image.Resampling.NEAREST)
            img = np.array(img.convert('L'))
            return img

    def __len__(self):
        return self.size

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

"""## Define some augmentation """

# Training labeled with weak augmentation 
train_transform = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

# Semi supervised transform for unlabled data with strong augmentation 
semi_transform = A.Compose(
    [
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


val_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ]
)

import ttach as tta
transforms = tta.Compose(
    [   
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ]
)
def tta_predict(model, image):
    '''
    image : B, C, H, W
    '''
    masks = []
    for transformer in transforms: 
        
        # augment image(B,C,H,W)
        augmented_image = transformer.augment_image(image)
        # pass to model
        res = model(augmented_image)
        res = res.sigmoid()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        # reverse augmentation for mask and label
        deaug_mask = transformer.deaugment_mask(res)
        
        # save results
        masks.append(deaug_mask)
        
    # reduce results as you want, e.g mean/max/min
    # mask = sum(masks)/len(masks)
    masks = torch.concat(masks, dim=0)
    return masks
