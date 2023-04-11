from src.trainer.config import *
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

@torch.no_grad()
def random_selection(model,labeled_flags,train_dataset,budget,device='cuda',n_drop=10):
    unlabeled_idxs=np.arange(len(labeled_flags))[~labeled_flags]
    np.random.shuffle(unlabeled_idxs)
    query_idxs=unlabeled_idxs[:budget]
    labeled_flags[query_idxs]=True
    return query_idxs,labeled_flags