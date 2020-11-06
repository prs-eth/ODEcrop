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
    logprobabilities_local_1 = list()
    logprobabilities_local_2 = list()
    targets_list = list()
    targets_list_local_1 = list()
    targets_list_local_2 = list()
    inputs_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):

        inputs, targets, targets_local_1, targets_local_2 = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        y_local_1 = targets_local_1.cpu().detach().numpy()
        y_local_2 = targets_local_2.cpu().detach().numpy()
        z, z_local_1, z_local_2 = model.forward(inputs)

        z = z.cpu().detach().numpy()
        z_local_1 = z_local_1.cpu().detach().numpy()
        z_local_2 = z_local_2.cpu().detach().numpy()
        
        inputs_list.append(x)
        targets_list.append(y)
        targets_list_local_1.append(y_local_1)
        targets_list_local_2.append(y_local_2)

        logprobabilities.append(z)
        logprobabilities_local_1.append(z_local_1)
        logprobabilities_local_2.append(z_local_2)
        
    return np.vstack(logprobabilities), np.vstack(logprobabilities_local_1), np.vstack(logprobabilities_local_2), np.vstack(inputs_list), np.concatenate(targets_list), np.concatenate(targets_list_local_1), np.concatenate(targets_list_local_2)

def test2(model, dataloader):
    model.eval()

    logprobabilities = list()
    logprobabilities_local_1 = list()
    logprobabilities_local_2 = list()
    targets_list = list()
    targets_list_local_1 = list()
    targets_list_local_2 = list()
    inputs_list = list()
    gt_instance_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        inputs, targets, targets_local_1, targets_local_2, gt_instance = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        y_local_1 = targets_local_1.cpu().detach().numpy()
        y_local_2 = targets_local_2.cpu().detach().numpy()
        y_i = gt_instance.cpu().detach().numpy()
        z, z_local_1, z_local_2 = model.forward(inputs)
        
        z = z.cpu().detach().numpy()
        z_local_1 = z_local_1.cpu().detach().numpy()
        z_local_2 = z_local_2.cpu().detach().numpy()
        
        inputs_list.append(x)
        targets_list.append(y)
        targets_list_local_1.append(y_local_1)
        targets_list_local_2.append(y_local_2)
        logprobabilities.append(z)
        logprobabilities_local_1.append(z_local_1)
        logprobabilities_local_2.append(z_local_2)
        gt_instance_list.append(y_i)
        
    #return np.vstack(logprobabilities), np.vstack(logprobabilities_local_1), np.vstack(logprobabilities_local_2), np.vstack(inputs_list), np.concatenate(targets_list), np.concatenate(targets_list_local_1), np.concatenate(targets_list_local_2), np.vstack(gt_instance_list)
    return np.vstack(logprobabilities), np.vstack(logprobabilities_local_1), np.vstack(logprobabilities_local_2), np.concatenate(targets_list), np.concatenate(targets_list_local_1), np.concatenate(targets_list_local_2), np.vstack(gt_instance_list)




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
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites,logprobabilites_local_1 ,logprobabilites_local_2, inputs, targets, targets_local_1, targets_local_2 = test(model, dataloader)
    predictions = logprobabilites.argmax(1)
    predictions_local_1 = logprobabilites_local_1.argmax(1)
    predictions_local_2 = logprobabilites_local_2.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()    
    predictions_local_1 = predictions_local_1.flatten()
    predictions_local_2 = predictions_local_2.flatten()
    targets_local_1 = targets_local_1.flatten()
    targets_local_2 = targets_local_2.flatten()
    
    
    #Ignore unknown class class_id=0
    valid_crop_samples = targets != 0
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    targets_local_1_wo_unknown = targets_local_1[valid_crop_samples]
    predictions_local_1_wo_unknown = predictions_local_1[valid_crop_samples]
    targets_local_2_wo_unknown = targets_local_2[valid_crop_samples]
    predictions_local_2_wo_unknown = predictions_local_2[valid_crop_samples]
    
    #confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    #print_report(*confusion_matrix_to_accuraccies(confusion_matrix))

    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    pix_acc_local_1 = np.sum( predictions_local_1_wo_unknown==targets_local_1_wo_unknown ) / predictions_local_1_wo_unknown.shape[0]
    pix_acc_local_2 = np.sum( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) / predictions_local_2_wo_unknown.shape[0]

    print('Pix acc = %.4f'%pix_acc)
    print('Pix acc - local 1 = %.4f'%pix_acc_local_1)
    print('Pix acc - local 2 = %.4f'%pix_acc_local_2)
   
    return pix_acc


def evaluate_fieldwise(model, dataset, batchsize=1, workers=0):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, logprobabilites_local_1, logprobabilites_local_2, targets, targets_local_1, targets_local_2, gt_instance = test2(model, dataloader)
    predictions = logprobabilites.argmax(1)
    predictions_local_1 = logprobabilites_local_1.argmax(1)
    predictions_local_2 = logprobabilites_local_2.argmax(1)
    
    #Save the results
    #np.savez('./viz/test_res.npz', targets, predictions, dataset.label_list_glob, dataset.label_list_glob_name)   
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    predictions_local_1 = predictions_local_1.flatten()
    predictions_local_2 = predictions_local_2.flatten()
    targets_local_1 = targets_local_1.flatten()
    targets_local_2 = targets_local_2.flatten()
    
    valid_crop_samples = targets != 0

    #Stage of classification
    log3 = logprobabilites.max(1)
    log2 = logprobabilites_local_1.max(1)
    log1 = logprobabilites_local_2.max(1)
    
    log3 = log3.flatten()
    log2 = log2.flatten()
    log1 = log1.flatten()

    log3 = log3[valid_crop_samples]
    log2 = log2[valid_crop_samples]
    log1 = log1[valid_crop_samples]
    
    THRESHOLD = 0.5
    s3 = log3 > THRESHOLD
    s2 = (log3 < THRESHOLD) * (log2 > THRESHOLD)
    #s1 = (log3 < THRESHOLD) * (log2 < THRESHOLD) * (log1 > THRESHOLD)
    s1 = 1 - (s2 + s3)
    
    
    print(np.sum(s3)/s3.shape[0])
    print(np.sum(s2)/s3.shape[0])
    print(np.sum(s1)/s3.shape[0])    

    
    print(np.unique(targets))
    print(np.unique(targets_local_1))
    print(np.unique(targets_local_2))
        
    #Ignore unknown class class_id=0
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]
    targets_local_1_wo_unknown = targets_local_1[valid_crop_samples]
    predictions_local_1_wo_unknown = predictions_local_1[valid_crop_samples]
    targets_local_2_wo_unknown = targets_local_2[valid_crop_samples]
    predictions_local_2_wo_unknown = predictions_local_2[valid_crop_samples]

    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)

    print('Global model')
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))

    print('Local-1 model')    
    confusion_matrix = build_confusion_matrix(targets_local_1_wo_unknown, predictions_local_1_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    

    print('Local-2 model')        
    confusion_matrix = build_confusion_matrix(targets_local_2_wo_unknown, predictions_local_2_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    pix_acc_local_1 = np.sum( predictions_local_1_wo_unknown==targets_local_1_wo_unknown ) / predictions_local_1_wo_unknown.shape[0]
    pix_acc_local_2 = np.sum( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) / predictions_local_2_wo_unknown.shape[0]
    

    pix_acc_hybrid = (predictions_wo_unknown==targets_wo_unknown) * s3 + ( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) * s2 + ( predictions_local_1_wo_unknown==targets_local_1_wo_unknown )* s1
    pix_acc_hybrid = np.sum(pix_acc_hybrid)/pix_acc_hybrid.shape[0]
    print('Hybrid pix acc: ', pix_acc_hybrid)
    
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
    print('Pix acc = %.4f'%pix_acc)
    print('Pix acc - local 1 = %.4f'%pix_acc_local_1)
    print('Pix acc - local 2 = %.4f'%pix_acc_local_2)
    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy)
    print('Field acc = %.4f'%fieldwise_accuracy)
 
    print('Global model')
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
#    print('Local-1 model')    
#    confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
#    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
#    print('Local-2 model')    
#    confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
#    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
    #np.save('./cm.npy', confusion_matrix)
    
    return pix_acc




