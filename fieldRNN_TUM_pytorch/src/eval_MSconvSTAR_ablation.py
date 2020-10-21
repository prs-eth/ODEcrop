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
from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

def test(model, model_gt, dataloader):
    model.eval()

    logprobabilities_1 = list()
    logprobabilities_2 = list()
    logprobabilities_3 = list()

    targets_list_1 = list()
    targets_list_2 = list()
    targets_list_3 = list()

    gt_instance_list = list()
#    logprobabilities_refined = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        inputs, targets3, targets1, targets2, gt_instance = data

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        
        y1 = targets1.numpy()
        y2 = targets2.numpy()
        y3 = targets3.numpy()
      
        y_i = gt_instance.cpu().detach().numpy()
        z3, z1, z2 = model.forward(inputs)
        
        z3_refined = model_gt([z1.detach(), z2.detach(), z3.detach()])

        if type(z3_refined) == tuple:
            z3_refined = z3_refined[0]
            
        z1 = z1.cpu().detach().numpy()
        z2 = z2.cpu().detach().numpy()
        z3 = z3.cpu().detach().numpy()
        z3_refined = z3_refined.cpu().detach().numpy()
        
        #inputs_list.append(x)
        targets_list_1.append(y1)
        targets_list_2.append(y2)
        targets_list_3.append(y3)

        logprobabilities_1.append(z1)
        logprobabilities_2.append(z2)
        logprobabilities_3.append(z3)

        gt_instance_list.append(y_i)
#        logprobabilities_refined.append(z3_refined)
        
    return np.vstack(logprobabilities_1), np.vstack(logprobabilities_2), np.vstack(logprobabilities_3), np.concatenate(targets_list_1), np.concatenate(targets_list_2), np.concatenate(targets_list_3), np.vstack(gt_instance_list)



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
    
    labels = np.unique(targets)
    labels = labels.tolist()
    #nclasses = len(labels)
        
    cm = sklearn_cm(targets, predictions, labels=labels)
#    precision = precision_score(targets, predictions, labels=labels, average='macro')
#    recall = recall_score(targets, predictions, labels=labels, average='macro')
#    f1 = f1_score(targets, predictions, labels=labels, average='macro')
#    kappa = cohen_kappa_score(targets, predictions, labels=labels)
    #print('precision, recall, f1, kappa: ', precision, recall, f1, kappa)
    
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
    #print('Per-class acc:', cl_acc)
    return cl_acc



def evaluate_fieldwise(model, model_gt, dataset, batchsize=1, workers=0, viz=False, fold_num=5, p=0.6):
    model.eval()
    model_gt.eval()
    
    THRESHOLD = np.log(p)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites_1, logprobabilites_2, logprobabilites_3, targets_1, targets_2, targets_3, gt_instance = test(model, model_gt, dataloader)
    predictions_1 = logprobabilites_1.argmax(1)
    predictions_2 = logprobabilites_2.argmax(1)
    predictions_3 = logprobabilites_3.argmax(1)
    
    prob_1 = logprobabilites_1.max(1)
    prob_2 = logprobabilites_2.max(1)
    prob_3 = logprobabilites_3.max(1)
    
    
    prob_1 = prob_1.flatten()
    prob_2 = prob_2.flatten()
    prob_3 = prob_3.flatten()
    
    predictions_1 = predictions_1.flatten()
    predictions_2 = predictions_2.flatten()
    predictions_3 = predictions_3.flatten()

    targets_1 = targets_1.flatten()
    targets_2 = targets_2.flatten()
    targets_3 = targets_3.flatten()

    gt_instance = gt_instance.flatten()
            
    #Ignore unknown class class_id=0
    valid_crop_samples = targets_3 != 0
        
    
    targets_wo_unknown_1 = targets_1[valid_crop_samples]
    targets_wo_unknown_2 = targets_2[valid_crop_samples]
    targets_wo_unknown_3 = targets_3[valid_crop_samples]

    predictions_wo_unknown_1 = predictions_1[valid_crop_samples]
    predictions_wo_unknown_2 = predictions_2[valid_crop_samples]
    predictions_wo_unknown_3 = predictions_3[valid_crop_samples]

    prob_1 = prob_1[valid_crop_samples]
    prob_2 = prob_2[valid_crop_samples]
    prob_3 = prob_3[valid_crop_samples]

    gt_instance_wo_unknown = gt_instance[valid_crop_samples]


    v1 = prob_1 > THRESHOLD
    v2 = (prob_2 > THRESHOLD)
    v3 = prob_3 > THRESHOLD
    
    coverage_3 =  v3
    coverage_2 =  v2 * (1 - coverage_3)
    coverage_1 = v1 * (1 - coverage_2 - coverage_3)
    
    coverage_rate_3 = np.sum(coverage_3 ) / v1.shape[0]
    coverage_rate_2 = np.sum( coverage_3 + coverage_2 ) / v1.shape[0]
    coverage_rate_1 = np.sum( coverage_3 + coverage_2 + coverage_1 ) / v1.shape[0]
    
    print('Coverages: ', coverage_rate_3, coverage_rate_2, coverage_rate_1)
    
    c1 = predictions_wo_unknown_1==targets_wo_unknown_1
    c2 = predictions_wo_unknown_2==targets_wo_unknown_2
    c3 = predictions_wo_unknown_3==targets_wo_unknown_3
    
    
    c3_T = c3 * coverage_3
    c2_T = c3 * coverage_3 + c2 * coverage_2
    c1_T = c3 * coverage_3 + c2 * coverage_2 + + c1 * coverage_1
    
    pix_acc_1 = np.sum( c1_T ) / c3_T.shape[0]
    pix_acc_2 = np.sum( c2_T ) / c3_T.shape[0]
    pix_acc_3 = np.sum( c3_T ) / c3_T.shape[0]
    
    print('Acc: ',pix_acc_3, pix_acc_2, pix_acc_1 )  
    
    
    
#    c1 = c1[v1]
#    c2 = c2[v2]
#    c3 = c3[v3]
#
#    pix_acc_1 = np.sum( c1 ) / c1.shape[0]
#    print('Pix acc = %.4f'%pix_acc_1)
#
#    pix_acc_2 = np.sum( c2 ) / c2.shape[0]
#    print('Pix acc = %.4f'%pix_acc_2)
#    
#    pix_acc_3 = np.sum( c3 ) / c2.shape[0]
#    print('Pix acc = %.4f'%pix_acc_3)
    

#    prediction_wo_fieldwise_1 = np.zeros_like(predictions_wo_unknown_1)
#    prediction_wo_fieldwise_2 = np.zeros_like(predictions_wo_unknown_2)
#    prediction_wo_fieldwise_3 = np.zeros_like(predictions_wo_unknown_3)
#    
#    for i in np.unique(gt_instance_wo_unknown).tolist():
#        field_indexes =  gt_instance_wo_unknown==i 
#        
#        pred = predictions_wo_unknown_1[field_indexes]
#        pred = np.argmax( np.bincount(pred) )
#        prediction_wo_fieldwise_1[field_indexes] = pred
#    
#        pred = predictions_wo_unknown_2[field_indexes]
#        pred = np.argmax( np.bincount(pred) )
#        prediction_wo_fieldwise_2[field_indexes] = pred
#        
#        pred = predictions_wo_unknown_3[field_indexes]
#        pred = np.argmax( np.bincount(pred) )
#        prediction_wo_fieldwise_3[field_indexes] = pred
#    
#    
#    fieldwise_pix_accuracy_1 = np.sum( prediction_wo_fieldwise_1==targets_wo_unknown_1 ) / prediction_wo_fieldwise_1.shape[0]
#    fieldwise_pix_accuracy_2 = np.sum( prediction_wo_fieldwise_2==targets_wo_unknown_2 ) / prediction_wo_fieldwise_2.shape[0]
#    fieldwise_pix_accuracy_3 = np.sum( prediction_wo_fieldwise_3==targets_wo_unknown_3 ) / prediction_wo_fieldwise_3.shape[0]
#    
#    print('-'*20)
#    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy_1)
#    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy_2)
#    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy_3)
 
    
    return [coverage_rate_3, coverage_rate_2, coverage_rate_1], [pix_acc_3, pix_acc_2, pix_acc_1]




def evaluate_fieldwise_full_coverage(model, model_gt, dataset, batchsize=1, workers=0, viz=False, fold_num=5, p=0.6):
    model.eval()
    model_gt.eval()
    
    THRESHOLD = np.log(p)
    
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites_1, logprobabilites_2, logprobabilites_3, targets_1, targets_2, targets_3, gt_instance = test(model, model_gt, dataloader)
    predictions_1 = logprobabilites_1.argmax(1)
    predictions_2 = logprobabilites_2.argmax(1)
    predictions_3 = logprobabilites_3.argmax(1)
    
    prob_1 = logprobabilites_1.max(1)
    prob_2 = logprobabilites_2.max(1)
    prob_3 = logprobabilites_3.max(1)
    
    
    prob_1 = prob_1.flatten()
    prob_2 = prob_2.flatten()
    prob_3 = prob_3.flatten()
    
    predictions_1 = predictions_1.flatten()
    predictions_2 = predictions_2.flatten()
    predictions_3 = predictions_3.flatten()

    targets_1 = targets_1.flatten()
    targets_2 = targets_2.flatten()
    targets_3 = targets_3.flatten()

    gt_instance = gt_instance.flatten()
            
    #Ignore unknown class class_id=0
    valid_crop_samples = targets_3 != 0
        
    
    targets_wo_unknown_1 = targets_1[valid_crop_samples]
    targets_wo_unknown_2 = targets_2[valid_crop_samples]
    targets_wo_unknown_3 = targets_3[valid_crop_samples]

    predictions_wo_unknown_1 = predictions_1[valid_crop_samples]
    predictions_wo_unknown_2 = predictions_2[valid_crop_samples]
    predictions_wo_unknown_3 = predictions_3[valid_crop_samples]

    prob_1 = prob_1[valid_crop_samples]
    prob_2 = prob_2[valid_crop_samples]
    prob_3 = prob_3[valid_crop_samples]

    gt_instance_wo_unknown = gt_instance[valid_crop_samples]


    v1 = prob_1 > THRESHOLD
    v2 = prob_2 > THRESHOLD
    v3 = prob_3 > THRESHOLD
    
    coverage_3 =  v3
    coverage_2 =  v2 * (1 - coverage_3)
    coverage_1 = v1 * (1 - coverage_2 - coverage_3)
    
    coverage_rate_3 = np.sum(coverage_3 ) / v1.shape[0]
    coverage_rate_2 = np.sum( coverage_3 + coverage_2 ) / v1.shape[0]
    coverage_rate_1 = np.sum( coverage_3 + coverage_2 + coverage_1 ) / v1.shape[0]
    
    print('Coverages: ', coverage_rate_3, coverage_rate_2, coverage_rate_1)
    
    c1 = predictions_wo_unknown_1==targets_wo_unknown_1
    c2 = predictions_wo_unknown_2==targets_wo_unknown_2
    c3 = predictions_wo_unknown_3==targets_wo_unknown_3
    
    
    c3_T = c3 
    c2_T = c3 * coverage_3 + c2 * (1-coverage_3)
    c1_T = c3 * coverage_3 + c2 * coverage_2 + + c1 * (1-coverage_2-coverage_3)
    
    pix_acc_1 = np.sum( c1_T ) / c3_T.shape[0]
    pix_acc_2 = np.sum( c2_T ) / c3_T.shape[0]
    pix_acc_3 = np.sum( c3_T ) / c3.shape[0]
    
    
    
    print('Acc: ',pix_acc_3, pix_acc_2, pix_acc_1 )  
    
    return [coverage_rate_3, coverage_rate_2, coverage_rate_1], [pix_acc_3, pix_acc_2, pix_acc_1]


