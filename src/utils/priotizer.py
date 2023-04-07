
import numpy as np
import torch

"""## Define prioritizer"""

def highest_uncertainty_prediction_selection(train_indexes, predictions, i):
    # Calculate number of pixels in range 0.35 and 0.65 as uncertainty score
    measure = lambda x: x[np.where(np.logical_and(x >= 0.25 - 0.01*i, x <= 0.75 + 0.01*i))].shape[0]
    scores = [measure(pred.cpu().numpy()) for pred in predictions]
    
    max_uncertain = list(zip(train_indexes, scores))
    max_uncertain.sort(key=lambda x: x[1])
    
    return list(zip(*max_uncertain))[0]

def highest_entropy_selection(train_indexes, predictions, i):
    scores = [torch.mean(pred * torch.log2(pred)) for pred in predictions]
    p = list(zip(train_indexes, scores))
    p.sort(reverse=True, key=lambda x : x[1]) # sort in descending order
    return list(zip(*p))[0]

def least_confidence_selection(train_indexes, predictions, i):
    scores = [torch.mean(pred) for pred in predictions]
    max_logit = list(zip(train_indexes, scores))
    max_logit.sort(key=lambda x: x[1]) # sort in ascending order
    
    return list(zip(*max_logit))[0]

def mc_dropout_selection(train_indexes, predictions, pseudo_list, i):
    scores = []
    
    low_threshold = 0.25
    high_threshold = 0.75

    for index in range(len(predictions)):
        print(predictions[index].shape)
        pred_arr = predictions[index].cpu().numpy()
        score = len(pred_arr[np.where(np.logical_and(pred_arr > low_threshold, pred_arr < high_threshold))])
        scores.append(score)
    
    max_uncertain = list(zip(train_indexes, scores))
    max_uncertain.sort(key=lambda x: x[1], reverse=True)

    descent_list = list(zip(pseudo_list, scores))
    descent_list.sort(key=lambda x: x[1], reverse=True)
    
    return list(zip(*max_uncertain))[0], list(zip(*descent_list))[0]


def JSD_selection(train_indexes, predictions, pseudo_list):
    scores = [torch.mean(pred) for pred in predictions]
    max_uncertain = list(zip(train_indexes, scores))
    max_uncertain.sort(key=lambda x: x[1], reverse=True) # sort in descending order
    
    descent_list = list(zip(pseudo_list, scores))
    descent_list.sort(key=lambda x: x[1], reverse=True)
    
    return list(zip(*max_uncertain))[0], list(zip(*descent_list))[0]