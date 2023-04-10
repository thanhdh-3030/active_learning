import os
import torch
import glob
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, ConcatDataset
import glob
# Apply WanDB
import wandb
import pytorch_lightning as pl
from src.models.active_model import *
from src.trainer.config import *
from strategies import *
from torch.utils.data.sampler import SubsetRandomSampler
from src.evaluation.metric import full_val
import imageio
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def eval_prioritization_strategy(prioritizer, experiment_name="Least confident", num_cluster=3, CYCLES=10, budget_size=80, device='cuda',use_wandb=False,num_epochs=100,batch_size=32):
    if use_wandb:
        wandb.init(project="Polyp Active Learning",
                group=experiment_name,
                name='(1)',
                entity='ssl-online')
    model =ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=1,
                                        checkpoint_path=f'checkpoints/active/fpn_densenet169.pth')
    # X_train_subset = []
    torch.save(model,'./checkpoints/active/initial_ckpt.pth')
    pool_samples=glob.glob('TrainDataset/image/*png')
    # labeled_samples=[]
    train_dataset = ActivePolybDataset(pool_samples, transform=semi_transform)
    labeled_flags=np.zeros(len(pool_samples), dtype=bool)
    labeled_idxs=[]
    for cycle in range(1, CYCLES):
        print('*'*50, cycle, '*'*50)
        if cycle==0:
            # random select k indexs
            tmp_idxs=np.arange(len(pool_samples))
            np.random.shuffle(tmp_idxs)
            queried_idxs=tmp_idxs[:budget_size]
            labeled_flags[queried_idxs] = True
        else:
            queried_idxs,labeled_flags=prioritizer(model,labeled_flags,train_dataset,budget_size)
        
        model=torch.load('./checkpoints/active/initial_ckpt.pth')

        # update labeled samples
        labeled_idxs.extend(queried_idxs)
        # pool_samples=pool_samples-queried_samples                
        # train_dataset = ActivePolybDataset(labeled_samples, transform=semi_transform)
        sampler= SubsetRandomSampler(labeled_idxs)
        train_dataloader = DataLoader(train_dataset, sampler=sampler,batch_size=16, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)
        trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=num_epochs)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)
        full_val(model.model, sum(labeled_flags), device=device, use_wandb=use_wandb)
        print('='*20, sum(labeled_flags), '='*20)
        previous_cycle_cpt_path='./checkpoints/active/fpn_densenet169.pth'
        os.rename(previous_cycle_cpt_path,f'./checkpoints/active/main_{cycle}.pth')
        torch.cuda.empty_cache()

    wandb.finish(quiet=False) 

    return None
eval_prioritization_strategy(bvsb_selection, "abc",CYCLES=6, budget_size=100,use_wandb=False,device='cuda',num_epochs=2,batch_size=16)
