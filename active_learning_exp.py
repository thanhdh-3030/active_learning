import os
import torch
import glob
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob
# Apply WanDB
import wandb
import pytorch_lightning as pl
from src.models.active_model import *
from src.trainer.config import *
from strategies import *
from torch.utils.data.sampler import SubsetRandomSampler
from src.evaluation.metric import full_val
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import argparse

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
def eval_prioritization_strategy(prioritizer, experiment_name="Least confident", CYCLES=10, budget_size=80, device='cuda',use_wandb=False,num_epochs=100,batch_size=32):
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
    for cycle in range(0, CYCLES):
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
        train_dataloader = DataLoader(train_dataset, sampler=sampler,batch_size=16, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=8)
        trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=num_epochs)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)
        full_val(model.model, sum(labeled_flags), device=device, use_wandb=use_wandb)
        print('='*20, sum(labeled_flags), '='*20)
        previous_cycle_cpt_path='./checkpoints/active/fpn_densenet169.pth'
        os.rename(previous_cycle_cpt_path,f'./checkpoints/active/main_{cycle}.pth')
        torch.cuda.empty_cache()

    wandb.finish(quiet=False) 

    return None
# eval_prioritization_strategy(strategies[opt.strategy], opt.exp_name,CYCLES=6, budget_size=100,use_wandb=True,device='cuda',num_epochs=opt.n_epochs,batch_size=opt.batch_size)
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--strategy",type=str,default="coreset")
    parser.add_argument("--exp_name",type=str,default="coreset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--use_wandb", type=bool, default=False)

    opt = parser.parse_args()
    print(opt)
    strategies={
        'coreset':core_set_selection,
        'coreset_en':core_set_selection_en,
        'coreset_pca':core_set_selection_pca,
        'bvsb':bvsb_selection,
        'lc':least_confidence_selection
    }
    eval_prioritization_strategy(strategies[opt.strategy], opt.exp_name,CYCLES=6, budget_size=100,use_wandb=opt.use_wandb,device='cuda',num_epochs=opt.n_epochs,batch_size=opt.batch_size)
