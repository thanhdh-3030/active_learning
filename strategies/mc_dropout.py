from src.trainer.config import *
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

@torch.no_grad()
def mc_dropout_selection(model,labeled_flags,train_dataset,budget,device='cuda',n_drop=10):
    unlabeled_idxs=np.where(labeled_flags==False)
    sampler=SubsetRandomSampler(unlabeled_idxs[0])    
    unlabeled_loader=torch.utils.data.DataLoader(train_dataset,sampler=sampler,batch_size=1, shuffle=False, num_workers=2)
    model.train()
    model.to(device)
    entropy_scores=[]
    for index, batch in enumerate(tqdm(unlabeled_loader)):
        images = batch['image'].to(device)
        probs=model.forward_dropout_split(images,n_drop=n_drop) # shape [n_drop,B,C,W,H]
        pb = probs.mean(0) # shape [B,C,W,H]
        entropy1 = (-pb*torch.log(pb)).sum(1) # shape [B,W,H]
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0) # shape [B,W,H]
        uncertainty_score=entropy1-entropy2
        avg_uncertainty=torch.mean(uncertainty_score)
        entropy_scores.append(avg_uncertainty.cpu().numpy().item())
        # if index ==50:
        #     break
    idxs=np.argsort(entropy_scores)
    query_idxs=unlabeled_idxs[0][idxs[-budget:]] # query samples have highest entropy score
    labeled_flags[query_idxs]=True
    return query_idxs,labeled_flags