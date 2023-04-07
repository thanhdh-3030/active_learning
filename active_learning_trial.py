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
from src.evaluation.metric import full_val
import imageio
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
def mc_dropout_selection(model,unlabeled_samples,budget,device='cuda:0',n_drop=5):
    enable_dropout(model)
    model.to(device)
    unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
    unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=2)
    confidence_scores=[]
    for index, batch in enumerate(tqdm(unlabeled_loader), start=1):
        images = batch['image'].to(device)
        probs = model.forward_dropout(images, n_drop=n_drop)
        class_probs=torch.max(probs,1-probs)
        mean_probs=torch.sum(class_probs,dim=[-1,-2])/(probs.shape[-1]*probs.shape[-2])
        confidence_scores.append(mean_probs.cpu().numpy().item())
    idxs=np.argsort(confidence_scores)
    unlabeled_samples=np.array(unlabeled_samples)
    queried_samples=unlabeled_samples[idxs[:budget]]
    return queried_samples
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
def bvsb_selection(model,unlabeled_samples,budget):
    unlabeled_set=ActivePolybDataset(unlabeled_samples,transform=semi_transform)
    unlabeled_loader=torch.utils.data.DataLoader(unlabeled_set, batch_size=1, shuffle=False, num_workers=2)
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
        # mean_probs=torch.sum(class_probs,dim=[-1,-2])/(logits.shape[-1]*logits.shape[-2])
        confidence_scores.append(avg_uncertainty.cpu().numpy().item())
    idxs=np.argsort(confidence_scores)
    unlabeled_samples=np.array(unlabeled_samples)
    queried_samples=unlabeled_samples[idxs[:budget]]
    return queried_samples
@torch.no_grad()
def mc_dropout_selection_true(model,unlabeled_samples,budget,device='cuda:0',n_drop=10):
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
# @torch.no_grad
def eval_prioritization_random(prioritizer, experiment_name="Random"):
    if use_wandb:
        wandb.init(project="Polyp Active Learning", 
                group=experiment_name,
                name='(1)', 
                dir='./wandb',
                entity='ssl-online')
    X_train = glob.glob('newdataset/*/image/*')
    y_train = [i.replace('image', 'mask') for i in X_train]

    train_indices = list(range(len(X_train)))

    test_accuracies = []
    X_train_subset, y_train_subset = np.array([]), np.array([])
    budget_size = 80
    CYCLES = 10
    for i in range(CYCLES):
        print('*'*50, i, '*'*50)
        selected_indices = train_indices[:budget_size]
        train_indices = train_indices[budget_size:]
        # Define dataset for training
        X_train_subset = np.concatenate((X_train_subset, np.array(X_train)[selected_indices,...]))
        y_train_subset = np.concatenate((y_train_subset, np.array(y_train)[selected_indices,...]))
        
        train_dataset = ActiveDataset(X_train_subset, y_train_subset, transform=train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)
        
        # Define unlabeled dataset for training
        X_unlabeled_subset = np.array(X_train)[train_indices,...]
        y_unlabeled_subset = np.array(y_train)[train_indices,...]
        
        unlabeled_dataset = ActiveDataset(X_unlabeled_subset, y_unlabeled_subset, transform=val_transform)
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False, num_workers=8)
        
        # Init model from scratch 
        model = ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=1, labeled_dataloader=train_dataloader,
                                        checkpoint_path=f'runs/checkpoints/fpn_densenet169_cycle_{i}.pth')

        # Training model from scratch
        trainer = pl.Trainer(accelerator="gpu", devices=[1], max_epochs=100)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)
        full_val(model.model.to(device), (i+1)*budget_size, device)
        
        predictions = get_predictions(model, unlabeled_dataloader)
        # Select next train index
        train_indices = prioritizer(train_indices, predictions, (i+1))
        # Clear cached cuda 
        torch.cuda.empty_cache()
        # Delete variables
        del model, train_dataset, train_dataloader, unlabeled_dataset, unlabeled_dataloader, predictions

    wandb.finish(quiet=False)

def eval_prioritization_strategy(prioritizer, experiment_name="Least confident", num_cluster=3, CYCLES=10, budget_size=80, device='cuda:0',use_wandb=False):
    if use_wandb:
        wandb.init(project="Polyp Active Learning",
                group=experiment_name,
                name='(1)',
                entity='ssl-online')   
    # train_indices = {}
    # data_image_dict = {}
    # data_mask_dict = {}
    # list_cluster = [i for i in range(num_cluster)]
    
    # for c in range(num_cluster):
    #     train_indices[c] = []
    #     data_image_dict[c] = []
    #     data_mask_dict[c] = []
        
    #     with open(f'runs/rotation_txt_{num_cluster}/sorted_dist_{c}.txt', 'r') as f:
    #         list_ = f.readlines()
    #         data_image_dict[c] = [i[:-1] for i in list_]
    #         data_mask_dict[c] = [i.replace('image', 'mask') for i in data_image_dict[c]]
    #         train_indices[c] = list(range(len(data_image_dict[c])))
    #         print(len(data_image_dict[c]))
    
    # test_accuracies = []
    model =ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=1,
                                        checkpoint_path=f'checkpoints/active/fpn_densenet169.pth')
    # model =ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=2,
    #                                     checkpoint_path=f'checkpoints/active/fpn_densenet169.pth')
    X_train_subset = []
    # save initial model
    torch.save(model,'./checkpoints/active/initial_ckpt.pth')
    for cycle in range(0, CYCLES):
        print('*'*50, cycle, '*'*50)
        # Define dataset for training
        # for c in list_cluster:
        #     selected_indices = np.array(train_indices[c][:budget_size]).ravel()
        #     train_indices[c] = train_indices[c][budget_size:]
        #     X_train_subset = X_train_subset + np.array(data_image_dict[c])[selected_indices].tolist()
        #     y_train_subset = [i.replace('image', 'mask') for i in X_train_subset]
            
        #     if selected_indices.shape[0] < budget_size:
        #         budget_size = (budget_size * num_cluster) // (num_cluster - 1)
        #         list_cluster.remove(c)
        #         num_cluster = num_cluster - 1
        with open(f'./losses/sixbatches_factorize/batch_{cycle}.txt', 'r') as f:
            samples = f.readlines()
        samples=[sample[:-1] for sample in samples] # remove newline character
        # randomly sampling
        if cycle==0:
            samples = np.array(samples)
            queried_samples = samples[[int(j*len(samples)/budget_size) for j in range(budget_size)]]
        # use best previous model to query
        # if cycle==(CYCLES-1):
        #     queried_samples=highest_uncertainty_selection1(mode,sample,budget=budget_size)
        else:
            # print('>> Getting previous checkpoint')
            # checkpoint = torch.load(f'./checkpoints/active/main_{cycle-1}.pth')
            # model.load_state_dict(checkpoint)
            # checkpoint_path=f'./checkpoints/active/main_{cycle-1}.pth'
            # model.load_from_checkpoint(checkpoint_path=checkpoint_path)
            queried_samples=prioritizer(model,samples,budget_size)

        model=torch.load('./checkpoints/active/initial_ckpt.pth')
        X_train_subset.extend(queried_samples)                
        train_dataset = ActivePolybDataset(X_train_subset, transform=semi_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)


        # trainer = pl.Trainer(accelerator="gpu", devices=[1], max_epochs=100,auto_lr_find=True)
        trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=100)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        # trainer.tune(model,train_dataloaders=train_dataloader, val_dataloaders=None)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)

        full_val(model.model, len(X_train_subset), device=device, use_wandb=use_wandb)
        print('='*20, len(X_train_subset), '='*20)
        previous_cycle_cpt_path='./checkpoints/active/fpn_densenet169.pth'
        os.rename(previous_cycle_cpt_path,f'./checkpoints/active/main_{cycle}.pth')
        # pseudo_list = []
        # update train indice from unlabeled pools
        # for c in list_cluster:
        #     X_unlabeled_subset = np.array(data_image_dict[c])[train_indices[c],...].tolist()
        #     y_unlabeled_subset = [i.replace('image', 'mask') for i in data_image_dict[c]]         
        #     unlabeled_dataset = ActiveDataset(X_unlabeled_subset, y_unlabeled_subset, transform=val_transform)
        #     unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=8)
        #     predictions, pseudo_subset = get_predictions(model, unlabeled_dataloader, device=device, n_drop=5)
        #     # pseudo_list.extend(pseudo_subset)
        #     # Select next train index
        #     train_indices[c] = prioritizer(train_indices[c], predictions, cycle)
            
        # Prepare data to train semi-supervised
        # nb_img = pseudo_list #[:len(X_train_subset)]
        # semi_img = X_train_subset + [i.replace('pseudo_mask', 'image') for i in nb_img]
        # semi_mask = y_train_subset + nb_img
            
        # semi_dataset = ActiveDataset(semi_img, semi_mask, transform=semi_transform)
        # semi_dataloader = DataLoader(semi_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=8)

        # semi_model = ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=1, labeled_dataloader=semi_dataloader,
        #                                 checkpoint_path=f'runs/checkpoints/fpn_densenet169_student.pth')

        # semi_trainer = pl.Trainer(accelerator="gpu", devices=[1], max_epochs=100)
        # semi_trainer.fit(semi_model, train_dataloaders=semi_dataloader, val_dataloaders=None)
        # full_val(semi_model.model, len(semi_img), device=device, use_wandb=use_wandb)
        # print('='*20, len(semi_img), '='*20)

        # Clear cached cuda 
        torch.cuda.empty_cache()
        # Delete variables
        # del model, train_dataset, train_dataloader, unlabeled_dataset, unlabeled_dataloader, predictions, semi_model, semi_dataset, semi_dataloader

    wandb.finish(quiet=False) 

    return None

# eval_prioritization_random(mc_dropout_selection, 'Random', )

eval_prioritization_strategy(mc_dropout_selection_true, 'pretext task with mc dropout aa',CYCLES=6, budget_size=100,use_wandb=True,device='cuda')

# wandb.init(project="Polyp Active Learning", 
#         group=f'Supervised',
#         name='(1)', 
#         entity='ssl-online')   
# X_train_subset = glob.glob('TrainDataset/image/*')
# y_train_subset = [i.replace('image', 'mask') for i in X_train_subset]         
# train_dataset = ActiveDataset(X_train_subset, y_train_subset, transform=semi_transform)
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)


# model = ActiveSegmentationModel("FPN", "densenet169", in_channels=3, out_classes=1, labeled_dataloader=train_dataloader,
#                                 checkpoint_path=f'runs/checkpoints/fpn_densenet169.pth')

# trainer = pl.Trainer(accelerator="gpu", devices=[1], max_epochs=100)  
# trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=None)
# full_val(model.model, len(X_train_subset), device='cuda:1', use_wandb=False)
# print('='*20, len(X_train_subset), '='*20)