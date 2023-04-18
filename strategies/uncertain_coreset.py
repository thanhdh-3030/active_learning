import torch
from src.trainer.config import *
from .mc_dropout import mc_dropout_selection
from .kcenter_greedy import core_set_selection_en

@torch.no_grad()
def uncertain_core_set_selection(model,labeled_flags,train_dataset,budget,device='cuda'):
    ratio=0.5
    num_uncertain_samples=int(budget*ratio)
    num_diverse_samples=budget-num_uncertain_samples
    uncertain_query_idxs,labeled_flags=mc_dropout_selection(model,labeled_flags,train_dataset,num_uncertain_samples)
    diverse_query_idxs,labeled_flags=core_set_selection_en(model,labeled_flags,train_dataset,num_diverse_samples)
    query_idxs=np.union1d(uncertain_query_idxs,diverse_query_idxs)
    return query_idxs,labeled_flags
