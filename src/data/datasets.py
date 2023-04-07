"""This file contains dataset stats information"""

from torchvision import datasets

from torch.utils.data import Dataset
from .cifar_20 import CIFAR20
from .polyp_dataset import ActiveDataset
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_images = glob.glob('newdataset/*/image/*')
train_masks = glob.glob('newdataset/*/mask/*')

test_folder = "TestDataset"
test_images = glob.glob(f'{test_folder}/*/images/*')
test_masks = glob.glob(f'{test_folder}/*/masks/*')



SUPPORTED_DATASETS = ['stl10', 'cifar10', 'cifar20', 'polyp']

IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CIFAR_STATS = {
    'mean': [0.491, 0.482, 0.447],
    'std': [0.247, 0.243, 0.261]
}

DATASET_STATS = {
    'cifar10': CIFAR_STATS,
    'cifar20': CIFAR_STATS,
    'stl10': IMAGENET_STATS,
    'polyp': IMAGENET_STATS
}

NUM_CLASSES = {
    'cifar10': 10,
    'cifar20': 20,
    'stl10': 10,
}


def get_dataset(dataset: str, train: bool,
                transform=None,
                download: bool = False,
                unlabeled: bool = False) -> Dataset:
    """Returns dataset

    Args:
        dataset: dataset name

        train: if True, then train split will be returned

        transform: transform to apply to images

        download: if True, then dataset will be downloaded, if not downloaded

        unlabeled: if True unlabeled split will be returned. Only for STL10

    Returns:
        Dataset: dataset
    """

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Unsupported dataset')

    if dataset == 'stl10':

        if train and unlabeled:
            split = 'train+unlabeled'
        elif train:
            split = 'train'
        elif unlabeled:
            split = 'unlabeled'
        else:
            split = 'test'

        return datasets.STL10('./data', split=split, download=download, transform=transform)
    elif dataset == 'cifar10':
        return datasets.CIFAR10('./data', train=train, download=download, transform=transform)
    elif dataset == 'cifar20':
        return CIFAR20('./data/cifar-20', train=train, download=download, transform=transform)
    elif dataset == 'polyp':
        return ActiveDataset(train_images+test_images, train_masks+test_masks, trainsize=352, transform=None)
