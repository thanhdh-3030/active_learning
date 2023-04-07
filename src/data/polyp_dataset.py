from torch.utils.data import Dataset
from .pertubation import *
import numpy as np
import torchvision.transforms as transforms
import copy
from PIL import Image, ImageFile
import PIL
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ActiveDataset(Dataset):
    """
    dataloader for polyp segmentation tasks
    """

    def __init__(self, image_paths=[], gt_paths=[], trainsize=352, transform=None):
        self.trainsize = trainsize
        assert len(image_paths) > 0, "Can't find any images in dataset"
        assert len(gt_paths) > 0, "Can't find any mask in dataset"
        self.images = image_paths
        self.masks = gt_paths
        self.size = len(self.images)
        self.filter_files()
        self.transform = transform

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])


        if self.transform is not None:
            image = self.transform(image)

        return image, self.images[index]

    def filter_files(self):
        assert len(self.images) == len(self.masks)
        images = []
        masks = []
        for img_path, mask_path in zip(self.images, self.masks):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if img.size == mask.size:
                images.append(img_path)
                masks.append(mask_path)
        self.images = images
        self.masks = masks

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')
            return img

    def __len__(self):
        return self.size




class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
