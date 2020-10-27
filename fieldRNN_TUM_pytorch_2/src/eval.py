#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:56:27 2019

@author: tmehmet
"""


import sys
sys.path.append("src")
sys.path.append("src/models")

import torch.optim
from tqdm import tqdm
import numpy as np

def test(model, dataloader):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    inputs_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):

        inputs, targets = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        z = model.forward(inputs)[0]
        z = z.cpu().detach().numpy()
        
        inputs_list.append(x)
        targets_list.append(y)
        logprobabilities.append(z)
        
    return np.vstack(logprobabilities), np.vstack(inputs_list), np.concatenate(targets_list) # np.vstack(targets_list)

def test2(model, dataloader):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    inputs_list = list()
    gt_instance_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        inputs, targets, gt_instance = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        y_i = gt_instance.cpu().detach().numpy()
        z = model.forward(inputs).cpu().detach().numpy()
        
        inputs_list.append(x)
        targets_list.append(y)
        logprobabilities.append(z)
        gt_instance_list.append(y_i)
        
    return np.vstack(logprobabilities), np.vstack(inputs_list), np.concatenate(targets_list), np.vstack(gt_instance_list)




def confusion_matrix_to_accuraccies(confusion_matrix):

    confusion_matrix = confusion_matrix.astype(float)
    # sum(0) <- predicted sum(1) ground truth

    total = np.sum(confusion_matrix)
    n_classes, _ = confusion_matrix.shape
    overall_accuracy = np.sum(np.diag(confusion_matrix)) / total

    # calculate Cohen Kappa (https://en.wikipedia.org/wiki/Cohen%27s_kappa)
    N = total
    p0 = np.sum(np.diag(confusion_matrix)) / N
    pc = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / N ** 2
    kappa = (p0 - pc) / (1 - pc)

    recall = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + 1e-12)
    precision = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=0) + 1e-12)
    f1 = (2 * precision * recall) / ((precision + recall) + 1e-12)

    # Per class accuracy
    cl_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-12)
    print(cl_acc)
    
    return overall_accuracy, kappa, precision, recall, f1, cl_acc

def build_confusion_matrix(targets, predictions):
    
    nclasses = len(np.unique(targets))
    cm, _, _ = np.histogram2d(targets, predictions, bins=nclasses)
    
    return cm

def print_report(overall_accuracy, kappa, precision, recall, f1, cl_acc):
    
    report="""
    overall accuracy: \t{:.3f}
    kappa \t\t{:.3f}
    precision \t\t{:.3f}
    recall \t\t{:.3f}
    f1 \t\t\t{:.3f}
    """.format(overall_accuracy, kappa, precision.mean(), recall.mean(), f1.mean())

    print(report)
    print('Per-class acc:', cl_acc)
    
def evaluate(model, dataset, batchsize=1, workers=0):
    label_list_local_1 = [0.,1,1,1,1,1,1,1,1,1,2,2,2,2,2] #unknown, fieldcrop, grassland 
    label_list_local_2 = [0.,1,2,1,2,3,3,3,3,4,5,5,6,6,6] #unknown, smallCreal, largeCreal, broadLeaf, veg, meadow, pastures
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, inputs, targets = test(model, dataloader)
    predictions = logprobabilites.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    
    #Ignore unknown class class_id=0
    valid_crop_samples = targets != 0
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    
    predictions_local_1_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    predictions_local_2_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    targets_local_1_wo_unknown = np.ones_like(targets_wo_unknown)*999
    targets_local_2_wo_unknown = np.ones_like(targets_wo_unknown)*999
    
    for i in range(1,len(label_list_local_1)):
        predictions_local_1_wo_unknown[predictions_wo_unknown==i] = label_list_local_1[i]
        targets_local_1_wo_unknown[targets_wo_unknown==i] = label_list_local_1[i]

        predictions_local_2_wo_unknown[predictions_wo_unknown==i] = label_list_local_2[i]
        targets_local_2_wo_unknown[targets_wo_unknown==i] = label_list_local_2[i]
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    pix_acc_local_1 = np.sum( predictions_local_1_wo_unknown==targets_local_1_wo_unknown ) / predictions_local_1_wo_unknown.shape[0]
    pix_acc_local_2 = np.sum( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) / predictions_local_2_wo_unknown.shape[0]

    print('Pix acc = %.4f'%pix_acc)
    print('Pix acc - local 1 = %.4f'%pix_acc_local_1)
    print('Pix acc - local 2 = %.4f'%pix_acc_local_2)

    
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    
    return confusion_matrix


def evaluate_fieldwise(model, dataset, batchsize=1, workers=0):
    label_list_local_1 = [0.,1,1,1,1,1,1,1,1,1,2,2,2,2,2] #unknown, fieldcrop, grassland 
    label_list_local_2 = [0.,1,2,1,2,3,3,3,3,4,5,5,6,6,6] #unknown, smallCreal, largeCreal, broadLeaf, veg, meadow, pastures
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, inputs, targets, gt_instance = test2(model, dataloader)
    predictions = logprobabilites.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    
    #Ignore unknown class class_id=0
    valid_crop_samples = targets != 0
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]


    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
    #target_wo_fieldwise = np.ones_like(predictions_wo_unknown)*9999

    
    #confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    #print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    

    predictions_local_1_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    predictions_local_2_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    targets_local_1_wo_unknown = np.ones_like(targets_wo_unknown)*999
    targets_local_2_wo_unknown = np.ones_like(targets_wo_unknown)*999
    
    for i in range(1,len(label_list_local_1)):
        predictions_local_1_wo_unknown[predictions_wo_unknown==i] = label_list_local_1[i]
        targets_local_1_wo_unknown[targets_wo_unknown==i] = label_list_local_1[i]

        predictions_local_2_wo_unknown[predictions_wo_unknown==i] = label_list_local_2[i]
        targets_local_2_wo_unknown[targets_wo_unknown==i] = label_list_local_2[i]
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    pix_acc_local_1 = np.sum( predictions_local_1_wo_unknown==targets_local_1_wo_unknown ) / predictions_local_1_wo_unknown.shape[0]
    pix_acc_local_2 = np.sum( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) / predictions_local_2_wo_unknown.shape[0]

    print('Pix acc = %.4f'%pix_acc)
    print('Pix acc - local 1 = %.4f'%pix_acc_local_1)
    print('Pix acc - local 2 = %.4f'%pix_acc_local_2)


    
    num_field = np.unique(gt_instance_wo_unknown).shape[0]
    target_field = np.ones(num_field)*8888
    prediction_field = np.ones(num_field)*9999
    
    count=0
    for i in np.unique(gt_instance_wo_unknown).tolist():
        field_indexes =  gt_instance_wo_unknown==i 
        
        pred = predictions_wo_unknown[field_indexes]
        pred = np.bincount(pred)
        pred =  np.argmax(pred)
        prediction_wo_fieldwise[field_indexes] = pred
        prediction_field[count] = pred 
    
        target = targets_wo_unknown[field_indexes]
        target = np.bincount(target)
        target =  np.argmax(target)
        #target_wo_fieldwise[field_indexes] = target    
        target_field[count] = target
        count+=1
    
    fieldwise_pix_accuracy = np.sum( prediction_wo_fieldwise==targets_wo_unknown ) / prediction_wo_fieldwise.shape[0]
    fieldwise_accuracy = np.sum( prediction_field==target_field ) / prediction_field.shape[0]
    print('Number of fields: ', count+1)
    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy)
    print('Fieldwise acc = %.4f'%fieldwise_accuracy)
 
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
    #np.save('./cm.npy', confusion_matrix)
    
    return pix_acc




