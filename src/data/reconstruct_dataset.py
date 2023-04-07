from torch.utils.data import Dataset
from .pertubation import *
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
from .datasets import SUPPORTED_DATASETS, get_dataset
import copy
from PIL import Image, ImageFile
import PIL
import torch
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

class SemanticGenesis_Dataset(Dataset):
    def __init__(self, dataset, input_size=(256, 256, 3), transform=False, simclr_transform=None):
        self.dataset = get_dataset(dataset, download=True, unlabeled=True, train=True)
        self.size = len(self.dataset)
        size = input_size[0]
        self._augmentations = A.Compose([
            A.Resize(size, size, always_apply=True),
            ToTensorV2()
        ])
        self.transform = transform
        self.simclr_transform = simclr_transform
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        data, path = self.dataset[index]
        # img_aug1, img_aug2 = self.simclr_transform(data)
        # img_aug1 = img_aug1 / 255.
        # img_aug2 = img_aug2 / 255.
        
        h,w = data.size[0], data.size[1]
        y_img = np.array(data).reshape((1, h, w))
        x_ori = copy.deepcopy(y_img)
        y_trans = ''


        r = random.random()
        if r <= 0.25:
            x = local_pixel_shuffling(y_img)
            y_trans = 'local-shuffle'
        elif 0.25 < r <= 0.5:
            x = nonlinear_transformation(y_img)
            y_trans = 'non-linear'
        elif 0.5 < r <= 0.75:
            x = image_in_painting(y_img)
            y_trans = 'in-paint'
        else:
            x = y_img
            y_trans = 'out-paint'
            
        x = np.resize(x, (1, 256, 256))
        x_ori = np.resize(x_ori, (1, 256, 256))
        
        x = torch.from_numpy(x) / 255.
        x_ori = torch.from_numpy(x_ori) / 255.
            

        # if self.transform:
        #     x = self._augmentations(image=x)['image'] / 255.
        #     x_ori = self._augmentations(image=x_ori)['image'] / 255.

        return x_ori, x_ori, x, x_ori, y_trans, path
        

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = np.array(img.convert('L'))
            return img

    def __len__(self):
        return self.size