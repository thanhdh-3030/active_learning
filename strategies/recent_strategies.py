from src.trainer.config import *
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
# def get_predictions(model, unlabeled_loader, device='cuda:0', n_drop=5):
#     enable_dropout(model)
#     model.to(device)

#     probs = []
#     pseudo_list = []
#     with torch.no_grad():
#         for index, batch in enumerate(tqdm(unlabeled_loader), start=1):
#             image = batch['image'].to(device)
#             image_path = batch['image_path'][0]
    
            
#             logits = model.forward_dropout(image, n_drop=n_drop)

#             mask = logits.detach().cpu().numpy().squeeze()
#             mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
#             mask = mask.round().astype(np.uint8)*255
#             imageio.imsave(image_path.replace('image', 'pseudo_mask'), mask)
                
#             pseudo_list.append(image_path.replace('image', 'pseudo_mask'))
#             probs.append(logits)
            
#         return probs, pseudo_list


"""## Define prioritizer"""

# def highest_uncertainty_prediction_selection(train_indexes, predictions, i):
#     # Calculate number of pixels in range 0.35 and 0.65 as uncertainty score
#     measure = lambda x: x[np.where(np.logical_and(x >= 0.25 - 0.01*i, x <= 0.75 + 0.01*i))].shape[0]
#     scores = [measure(pred.cpu().numpy()) for pred in predictions]
    
#     max_uncertain = list(zip(train_indexes, scores))
#     max_uncertain.sort(key=lambda x: x[1])
    
#     return list(zip(*max_uncertain))[0]

# def highest_entropy_selection(train_indexes, predictions, i):
#     scores = [torch.mean(pred * torch.log2(pred)) for pred in predictions]
#     p = list(zip(train_indexes, scores))
#     p.sort(reverse=True, key=lambda x : x[1]) # sort in descending order
    
#     return list(zip(*p))[0]

# def least_confidence_selection(train_indexes, predictions, i):
#     scores = [torch.mean(pred) for pred in predictions]
#     max_logit = list(zip(train_indexes, scores))
#     max_logit.sort(key=lambda x: x[1]) # sort in ascending order
    
    # return list(zip(*max_logit))[0]

# def mc_dropout_selection(train_indexes, predictions, i):
#     scores = []
    
#     low_threshold = 0.25
#     high_threshold = 0.75

#     for index in range(len(predictions)):
#         pred_arr = predictions[index].cpu().numpy()
#         score = len(pred_arr[np.where(np.logical_and(pred_arr > low_threshold, pred_arr < high_threshold))])
#         scores.append(score)
    
#     max_uncertain = list(zip(train_indexes, scores))
#     max_uncertain.sort(key=lambda x: x[1], reverse=True)
    
#     return list(zip(*max_uncertain))[0]

@torch.no_grad()
def highest_uncertainty_selection(model,unlabeled_samples,budget):
    unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
    unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=2)
    model.eval()
    model.to(device)
    uncertainty_scores=[]
    for index, batch in enumerate(tqdm(unlabeled_loader)):
        images = batch['image'].to(device)
        # image_path = batch['image_path'][0]
        # logits = model.forward_dropout(image, n_drop=n_drop)
        logits=model(images)
        probs=torch.sigmoid(logits)
        measure = lambda probs: probs[np.where(np.logical_and(probs >= 0.25, probs <= 0.75))].shape[0]/(probs.shape[-1]*probs.shape[-2])
        uncertainty_scores.append(measure(probs.cpu().numpy()))
    idxs=np.argsort(uncertainty_scores)
    unlabeled_samples=np.array(unlabeled_samples)
    queried_samples=unlabeled_samples[idxs[-budget:]]
    return queried_samples


# @torch.no_grad()
# def mc_dropout_selection(model,unlabeled_samples,budget,device='cuda:0',n_drop=5):
#     enable_dropout(model)
#     model.to(device)
#     unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
#     unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=2)
#     confidence_scores=[]
#     for index, batch in enumerate(tqdm(unlabeled_loader), start=1):
#         images = batch['image'].to(device)
#         probs = model.forward_dropout(images, n_drop=n_drop)
#         class_probs=torch.max(probs,1-probs)
#         mean_probs=torch.sum(class_probs,dim=[-1,-2])/(probs.shape[-1]*probs.shape[-2])
#         confidence_scores.append(mean_probs.cpu().numpy().item())
#     idxs=np.argsort(confidence_scores)
#     unlabeled_samples=np.array(unlabeled_samples)
#     queried_samples=unlabeled_samples[idxs[:budget]]
#     return queried_samples
# Loss function and optimizer
# def entropy_2d(p, dim=0, keepdim=False):
#     """ 
#     We compute the entropy along the first dimension, for each value of the tensor
#     :param p: tensor of probabilities
#     :param dim: dimension along which we want to compute entropy (the sum across this dimension must be equal to 1)
#     """
#     entrop = - torch.sum(p * torch.log(p + 1e-18), dim=dim, keepdim=keepdim)
#     return entrop


# def compute_aver_entropy_2d(prob_input, entropy_dim=1, aver_entropy_dim=[1, 2]):
#     tot_entropy = entropy_2d(prob_input, dim=entropy_dim)
#     aver_entropy = torch.mean(tot_entropy, dim=aver_entropy_dim)
#     return aver_entropy


# def compute_entropy_aver_2d(prob_input, p_ave_dim=[2, 3], entropy_aver_dim=1):
#     p_ave = torch.mean(prob_input, dim=p_ave_dim)
#     entropy_aver = entropy_2d(p_ave, dim=entropy_aver_dim)
#     return entropy_aver
# def JSD(prob_dists, alpha=0.5, p_ave_dim=0, entropy_aver_dim=0, entropy_dim=1, aver_entropy_dim=0):
#     """
#     JS divergence JSD(p1, .., pn) = H(sum_i_to_n [w_i * p_i]) - sum_i_to_n [w_i * H(p_i)], where w_i is the weight given to each probability
#                                   = Entropy of average prob. - Average of entropy

#     :param prob_dists: probability tensors (shape: B, C, H, W)
#     :param alpha: weight on terms of the JSD
#     """
#     entropy_mean = compute_entropy_aver_2d(prob_dists, p_ave_dim=p_ave_dim, entropy_aver_dim=entropy_aver_dim)
#     mean_entropy = compute_aver_entropy_2d(prob_dists, entropy_dim=entropy_dim, aver_entropy_dim=aver_entropy_dim)
#     jsd = alpha * entropy_mean - (1 - alpha) * mean_entropy
    
#     return jsd
"""## Main active learning strategy"""
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
    query_idxs=unlabeled_idxs[idxs[:budget]] # query smallest margin samples
    labeled_flags[query_idxs]=True
    return query_idxs,labeled_flags
@torch.no_grad()
def least_confidence_selection(model,unlabeled_samples,budget):
    unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
    unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=2)
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
    unlabeled_samples=np.array(unlabeled_samples)
    queried_samples=unlabeled_samples[idxs[:budget]]
    return queried_samples
@torch.no_grad()
def mc_dropout_selection(model,unlabeled_samples,budget,device='cuda:0',n_drop=10):
    model.train()
    model.to(device)
    unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
    unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=1)
    uncertainty_scores=[]
    for index, batch in enumerate(tqdm(unlabeled_loader)):
        images = batch['image'].to(device)
        probs=model.forward_dropout_split(images,n_drop=n_drop) # shape [n_drop,B,C,W,H]
        pb = probs.mean(0) # shape [B,C,W,H]
        entropy1 = (-pb*torch.log(pb)).sum(1) # shape [B,W,H]
        entropy2 = (-probs*torch.log(probs)).sum(2).mean(0) # shape [B,W,H]
        uncertainty_score=entropy2-entropy1
        avg_uncertainty=torch.mean(uncertainty_score)
        uncertainty_scores.append(avg_uncertainty.cpu().numpy().item())
    idxs=np.argsort(uncertainty_scores)
    unlabeled_samples=np.array(unlabeled_samples)
    queried_samples=unlabeled_samples[idxs[:budget]]
    return queried_samples
