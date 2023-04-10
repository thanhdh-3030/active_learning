from src.trainer.config import *
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
@torch.no_grad()
def bvsb_selection(model,labeled_flags,train_dataset,budget):
    unlabeled_idxs=np.where(labeled_flags==False)
    sampler=SubsetRandomSampler(unlabeled_idxs[0])
    unlabeled_loader=torch.utils.data.DataLoader(train_dataset,sampler=sampler,batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    model.to(device)
    confidence_scores=[]
    for index, batch in enumerate(tqdm(unlabeled_loader)):
        images = batch['image'].to(device)
        logits=model(images)
        probs=torch.sigmoid(logits)
        bvsb = torch.abs(2*probs - 1)
        uncertainty = 1 - bvsb
        avg_uncertainty = torch.mean(uncertainty)
        confidence_scores.append(avg_uncertainty.cpu().numpy().item())
    idxs=np.argsort(confidence_scores)
    query_idxs=unlabeled_idxs[0][idxs[:budget]] # query smallest margin samples
    labeled_flags[query_idxs]=True
    return query_idxs,labeled_flags