from src.trainer.config import *
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

@torch.no_grad()
def least_confidence_selection(model,labeled_flags,train_dataset,budget,device='cuda'):
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
        class_probs=torch.max(probs,1-probs)
        mean_probs=torch.sum(class_probs,dim=[-1,-2])/(logits.shape[-1]*logits.shape[-2])
        confidence_scores.append(mean_probs.cpu().numpy().item())
    idxs=np.argsort(confidence_scores)
    query_idxs=unlabeled_idxs[0][idxs[:budget]] # query smallest confidence score samples
    labeled_flags[query_idxs]=True
    return query_idxs,labeled_flags