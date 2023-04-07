'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from pathlib import Path

import os
import argparse
import random
import numpy as np

from src.models.unet import UNet
# from loader import Loader, RotationLoader
from src.utils.data_loading import BasicDataset, CarvanaDataset
from tqdm import tqdm
# from utils.utils import progress_bar
dir_img = Path('./TrainDataset/image')
dir_mask = Path('./TrainDataset/image')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_scale=1
testset = CarvanaDataset(dir_img, dir_mask, img_scale)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

net = UNet(n_channels=3, n_classes=3)
# net.linear = nn.Linear(512, 4)
net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

checkpoint = torch.load('./checkpoints/pretexts/checkpoint_epoch30.pth')
net.load_state_dict(checkpoint)

criterion = nn.MSELoss()

def test(epoch):
    global best_acc
    net.eval()

    with torch.no_grad():
        for batch_idx, results in enumerate(tqdm(testloader)):
            inputs,targets=results['image'],results['mask']
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss = loss.item()
            s = str(float(loss)) + '_' + str(results['path'][0]) + "\n"

            with open('./polyp_reconstruction_loss.txt', 'a') as f:
                f.write(s)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    # test(1)
    CYCLE=6
    samples_per_batch=round(1450/CYCLE)
    with open('./losses/polyp_reconstruction_loss.txt', 'r') as f:
        losses = f.readlines()

    loss_1 = []
    name_2 = []

    for j in losses:
        loss_1.append(float(j[:-1].split('_')[0]))
        name_2.append(j[:-1].split('_')[1])

    s = np.array(loss_1)
    sort_index = np.argsort(s)
    x = sort_index.tolist()
    x.reverse()
    sort_index = np.array(x) # convert to high loss first

    # if not os.path.isdir('loss'):
    #     os.mkdir('loss')
    # for i in range(CYCLE):
    #     # sample minibatch from unlabeled pool 
    #     sample5000 = sort_index[i*samples_per_batch:(i+1)*samples_per_batch]
    #     # sample1000 = sample5000[[j*5 for j in range(1000)]]
    #     # b = np.zeros(10)
    #     # for jj in sample5000:
    #     #     b[int(name_2[jj].split('/')[-2])] +=1
    #     # print(f'{i} Class Distribution: {b}')
    #     s = './losses/sixbatches_factorize/batch_' + str(i) + '.txt'
    #     for k in sample5000:
    #         with open(s, 'a') as f:
    #             f.write(name_2[k]+'\n')
    for i in range (CYCLE):
        batch_samples=sort_index[i::CYCLE]
        s = './losses/sixbatches_factorize/batch_' + str(i) + '.txt'
        for k in batch_samples:
            with open(s, 'a') as f:
                f.write(name_2[k]+'\n')