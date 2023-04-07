from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from copy import deepcopy
from .constants import *
import torch

def range_transform(sample):
    """
    Range normalization for 255 range of values
    :param sample: numpy array for normalize
    :return: normalize numpy array
    """
    if (np.round(np.max(sample)) <= 1.):
        sample = sample * 255

    m = 255 / (np.max(sample) - np.min(sample))
    n = 255 - m * np.max(sample)
    return (m * sample + n) / 255


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def compute_uncertain(sample, prediction, model):
    """
    Computes uncertainty map for a given sample and its prediction for a given model, based on the
    number of step predictions defined in constants file.
    :param sample: input sample.
    :param prediction: input sample prediction.
    :param model: unet model with Dropout layers.
    :return: uncertainty map.
    """
    prediction = cv2.threshold(prediction, 0.5, 1, cv2.THRESH_BINARY)[1].astype('uint8')
    X = np.zeros([1, img_rows, img_cols])

    for t in range(nb_step_predictions):
        pred = model(sample)
        pred = pred.sigmoid().detach().cpu().numpy().reshape([1, img_rows, img_cols])
        X = np.concatenate((X, pred))

    X = np.delete(X, [0], 0)

    if (apply_edt):
        # apply distance transform normalization.
        var = np.var(X, axis=0)
        transform = range_transform(edt(prediction))
        return np.sum(var * transform)

    else:
        return np.sum(np.var(X, axis=0))


def get_prediction_uncertain(model, unlabeled_loader, device='cuda:1'):
    mc_model = enable_dropout(model)
    mc_model.to(device)
    model.to(device)
    
    # get model prediction 
    predictions = []
    uncertains = []
    
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            image = batch['image'].to(device)

            logits = model(image)
            logits = logits.sigmoid().detach().cpu().numpy()
            
            for batch_idx in range(logits.shape[0]):
                predictions.append(logits[batch_idx])
                
                uncertain = compute_uncertain(image, logits[batch_idx], mc_model)
                uncertains.append(uncertain)
    
        predictions = np.array(predictions)
        uncertains = np.array(uncertains)
    
    return predictions, uncertains



def interval(data, start, end):
    """
    Returns the index of data within range values from start to end.
    :param data: numpy array of data.
    :param start: starting value.
    :param end: ending value.
    :return: numpy array of data index.
    """
    p = np.where(data >= start)[0]
    return p[np.where(data[p] < end)[0]]


def get_pseudo_index(uncertain, nb_pseudo):
    """
    Gives the index of the most certain data, to make the pseudo annotations.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_pseudo: Total of pseudo samples.
    :return: Numpy array of index.
    """
    h = np.histogram(uncertain, 80)

    pseudo = interval(uncertain, h[1][np.argmax(h[0])], h[1][np.argmax(h[0]) + 1])
    np.random.shuffle(pseudo)
    return pseudo[0:nb_pseudo]


def random_index(uncertain, nb_random):
    """
    Gives the index of the random selection to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_random: Total of random samples.
    :return: Numpy array of index.
    """
    histo = np.histogram(uncertain, 80)
    # TODO: automatic selection of random range
    index = interval(uncertain, histo[1][np.argmax(histo[0]) + 6], histo[1][len(histo[0]) - 33])
    np.random.shuffle(index)
    return index[0:nb_random]


def no_detections_index(uncertain, nb_no_detections):
    """
    Gives the index of the no detected samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :return: Numpy array of index.
    """
    return np.argsort(uncertain)[0:nb_no_detections]


def most_uncertain_index(uncertain, nb_most_uncertain, rate):
    """
     Gives the index of the most uncertain samples to be manually annotated.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    TODO: automatic selection of rate.
    :return: Numpy array of index.
    """
    data = np.array([]).astype('int')

    histo = np.histogram(uncertain, 80)

    p = np.arange(len(histo[0]) - rate, len(histo[0]))  # index of last bins above the rate
    pr = np.argsort(histo[0][p])  # p index accendent sorted
    cnt = 0
    pos = 0
    index = np.array([]).astype('int')

    while (cnt < nb_most_uncertain and pos < len(pr)):
        sbin = histo[0][p[pr[pos]]]

        index = np.append(index, p[pr[pos]])
        cnt = cnt + sbin
        pos = pos + 1

    for i in range(0, pos):
        data = np.concatenate((data, interval(uncertain, histo[1][index[i]], histo[1][index[i] + 1])))

    np.random.shuffle(data)
    return data[0:nb_most_uncertain]


def get_oracle_index(uncertain, nb_no_detections, nb_random, nb_most_uncertain, rate):
    """
    Gives the index of the unlabeled data to annotated at specific CEAL iteration, based on their uncertainty.
    :param uncertain: Numpy array with the overall uncertainty values of the unlabeled data.
    :param nb_no_detections: Total of no detected samples.
    :param nb_random: Total of random samples.
    :param nb_most_uncertain: Total of most uncertain samples.
    :param rate: Hash threshold to define the most uncertain area. Bin of uncertainty histogram.
    :return: Numpy array of index.
    """
    return np.concatenate((no_detections_index(uncertain, nb_no_detections),
                           most_uncertain_index(uncertain, nb_most_uncertain, rate)))


def compute_dice_coef(y_true, y_pred):
    """
    Computes the Dice-Coefficient of a prediction given its ground truth.
    :param y_true: Ground truth.
    :param y_pred: Prediction.
    :return: Dice-Coefficient value.
    """
    smooth = 1.  # smoothing value to deal zero denominators.
    y_true_f = y_true.reshape([1, img_rows * img_cols])
    y_pred_f = y_pred.reshape([1, img_rows * img_cols])
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def compute_train_sets(X_train, y_train, labeled_index, unlabeled_loader, model, iteration):
    """
    Performs the Cost-Effective Active Learning labeling step, giving the available training data for each iteration.
    :param X_train: Overall training data.
    :param y_train: Overall training labels. Including the unlabeled samples to simulate the oracle annotations.
    :param labeled_index: Index of labeled samples.
    :param unlabeled_index: Index of unlabeled samples.
    :param weights: pre-trained unet weights.
    :param iteration: Currently CEAL iteration.
    :return: X_labeled_train: Update of labeled training data, adding the manual and pseudo annotations.
    :return: y_labeled_train: Update of labeled training labels, adding the manual and pseudo annotations.
    :return: labeled_index: Update of labeled index, adding the manual annotations.
    :return: unlabeled_index: Update of labeled index, removing the manual annotations.
    """
    print("\nActive iteration " + str(iteration))
    print("-" * 50 + "\n")


    predictions, uncertain = get_prediction_uncertain(model, unlabeled_loader, device=device)
    np.save(global_path + "logs/uncertain" + str(iteration), uncertain)

    oracle_index = get_oracle_index(uncertain, nb_no_detections, nb_random, nb_most_uncertain,
                                    most_uncertain_rate)

    oracle_rank = unlabeled_index[oracle_index]

    np.save(global_path + "ranks/oracle" + str(iteration), oracle_rank)
    np.save(global_path + "ranks/oraclelogs" + str(iteration), oracle_index)

    labeled_index = np.concatenate((labeled_index, oracle_rank))

    if (iteration >= pseudo_epoch):

        pseudo_index = get_pseudo_index(uncertain, nb_pseudo_initial + (pseudo_rate * (iteration - pseudo_epoch)))
        pseudo_rank = unlabeled_index[pseudo_index]

        np.save(global_path + "ranks/pseudo" + str(iteration), pseudo_rank)
        np.save(global_path + "ranks/pseudologs" + str(iteration), pseudo_index)

        X_labeled_train = np.concatenate((X_train[labeled_index], X_train[pseudo_index]))
        y_labeled_train = np.concatenate((y_train[labeled_index], predictions[pseudo_index]))

    else:
        X_labeled_train = np.concatenate((X_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])
        y_labeled_train = np.concatenate((y_train[labeled_index])).reshape([len(labeled_index), 1, img_rows, img_cols])

    unlabeled_index = np.delete(unlabeled_index, oracle_index, 0)

    return X_labeled_train, y_labeled_train, labeled_index, unlabeled_index