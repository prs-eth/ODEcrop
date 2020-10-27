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

def test(model, dataloader):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    inputs_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):

        #inputs, targets = data
        input, _, _, target = data
          
        targets = torch.argmax(target,1)
        inputs = input.float()

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        z = model.forward(inputs)
        z = z.cpu().detach().numpy()
        
        inputs_list.append(x)
        targets_list.append(y)
        logprobabilities.append(z)
        
    return np.vstack(logprobabilities), np.vstack(inputs_list), np.concatenate(targets_list) # np.vstack(targets_list)

def test2(model, dataloader):
    model.eval()

    logprobabilities = list()
    targets_list = list()
    #inputs_list = list()
    gt_instance_list = list()

    for iteration, data in tqdm(enumerate(dataloader)):
        inputs, targets, _, _, gt_instance = data

        #Reshape the data
        inputs = inputs.permute(0,3,4,1,2)
        inputs = inputs.contiguous().view(-1, inputs.shape[3], inputs.shape[4])
        targets = targets.contiguous().view(-1)
        gt_instance = gt_instance.contiguous().view(-1)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        
        #x = inputs.cpu().detach().numpy()
        y = targets.cpu().detach().numpy()
        y_i = gt_instance.cpu().detach().numpy()
        z = model.forward(inputs)
        
        if type(z) == tuple:
            z = z[0]
        
        z = z.cpu().detach().numpy()
        
        #inputs_list.append(x)
        targets_list.append(y)
        logprobabilities.append(z)
        gt_instance_list.append(y_i)
        
    return np.vstack(logprobabilities), np.concatenate(targets_list), np.vstack(gt_instance_list)




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
    #print(cl_acc)
    
    return overall_accuracy, kappa, precision, recall, f1, cl_acc

def build_confusion_matrix(targets, predictions):
    
    labels = np.unique(targets)
    labels = labels.tolist()
    nclasses = len(labels)
    
    #cm, _, _ = np.histogram2d(targets, predictions, bins=nclasses)
    cm = sklearn_cm(targets, predictions, labels=labels)
    precision = precision_score(targets, predictions, labels=labels, average='macro')
    recall = recall_score(targets, predictions, labels=labels, average='macro')
    f1 = f1_score(targets, predictions, labels=labels, average='macro')
    kappa = cohen_kappa_score(targets, predictions, labels=labels)
    
    print('precision, recall, f1, kappa: ', precision, recall, f1, kappa)
    
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
    
def evaluate(model, dataset, batchsize=1, workers=0):
    label_list_local = range(19) #unknown, fieldcrop, grassland 
    label_list_local_1 = [0.,1,1,1,1,1,1,1,1,1,2,2,2,2,2] #unknown, fieldcrop, grassland 
    label_list_local_2 = [0.,1,2,1,2,3,3,3,3,4,5,5,6,6,6] #unknown, smallCreal, largeCreal, broadLeaf, veg, meadow, pastures
    
    #TODO: need fast tensordataloader? maybe not?
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, inputs, targets = test(model, dataloader)
    predictions = logprobabilites.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    
    #Ignore unknown class class_id=0
    valid_crop_samples = targets != 0
    targets_wo_unknown = targets#[valid_crop_samples]
    predictions_wo_unknown = predictions#[valid_crop_samples]
    
    predictions_local_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    predictions_local_1_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    predictions_local_2_wo_unknown = np.ones_like(predictions_wo_unknown)*888
    targets_local_wo_unknown = np.ones_like(targets_wo_unknown)*999
    targets_local_1_wo_unknown = np.ones_like(targets_wo_unknown)*999
    targets_local_2_wo_unknown = np.ones_like(targets_wo_unknown)*999
    
    for i in range(1,len(label_list_local_1)):

        predictions_local_wo_unknown[predictions_wo_unknown==i] = label_list_local[i]
        targets_local_wo_unknown[targets_wo_unknown==i] = label_list_local[i]

        #predictions_local_1_wo_unknown[predictions_wo_unknown==i] = label_list_local_1[i]
        #targets_local_1_wo_unknown[targets_wo_unknown==i] = label_list_local_1[i]

        #predictions_local_2_wo_unknown[predictions_wo_unknown==i] = label_list_local_2[i]
        #targets_local_2_wo_unknown[targets_wo_unknown==i] = label_list_local_2[i]
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    pix_acc_local_1 = np.sum( predictions_local_1_wo_unknown==targets_local_1_wo_unknown ) / predictions_local_1_wo_unknown.shape[0]
    pix_acc_local_2 = np.sum( predictions_local_2_wo_unknown==targets_local_2_wo_unknown ) / predictions_local_2_wo_unknown.shape[0]

    print('Pix acc = %.4f'%pix_acc)
    print('Pix acc - local 1 = %.4f'%pix_acc_local_1)
    print('Pix acc - local 2 = %.4f'%pix_acc_local_2)

    
    confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))

    overall_accuracy, kappa, precision, recall, f1, class_acc = confusion_matrix_to_accuraccies(confusion_matrix)
    
    return overall_accuracy


def evaluate_fieldwise(model, dataset, batchsize=1, workers=0, viz=False, fold_num=5, name='tr'):

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchsize, num_workers=workers)

    logprobabilites, targets, gt_instance = test2(model, dataloader)
    predictions = logprobabilites.argmax(1)
    
    predictions = predictions.flatten()
    targets = targets.flatten()
    gt_instance = gt_instance.flatten()
    
    #Ignore unknown class class_id=0
    if viz:
        valid_crop_samples = targets != 9999999999
    else:
        valid_crop_samples = targets != 0
    
    
    targets_wo_unknown = targets[valid_crop_samples]
    predictions_wo_unknown = predictions[valid_crop_samples]
    gt_instance_wo_unknown = gt_instance[valid_crop_samples]

    class_acc = confusion_matrix = build_confusion_matrix(targets_wo_unknown, predictions_wo_unknown)
    print_report(*confusion_matrix_to_accuraccies(confusion_matrix))
    
    pix_acc = np.sum( predictions_wo_unknown==targets_wo_unknown ) / predictions_wo_unknown.shape[0]
    print('Pix acc = %.4f'%pix_acc)

    prediction_wo_fieldwise = np.zeros_like(targets_wo_unknown)
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
        target_field[count] = target
        count+=1
    
    fieldwise_pix_accuracy = np.sum( prediction_wo_fieldwise==targets_wo_unknown ) / prediction_wo_fieldwise.shape[0]
    fieldwise_accuracy = np.sum( prediction_field==target_field ) / prediction_field.shape[0]
    # TODO: F1 score
    print('Fieldwise pix acc = %.4f'%fieldwise_pix_accuracy)
    print('Fieldwise acc = %.4f'%fieldwise_accuracy)
 
    #confusion_matrix = build_confusion_matrix(targets_wo_unknown, prediction_wo_fieldwise)
    #print_report(*confusion_matrix_to_accuraccies(confusion_matrix))    
    #np.save('./cm.npy', confusion_matrix)

    overall_accuracy, kappa, precision, recall, f1, class_acc = confusion_matrix_to_accuraccies(confusion_matrix)

    #Save for the visulization 
    if viz:
        predictions = predictions.reshape(-1,24,24)
        prediction_wo_fieldwise = prediction_wo_fieldwise.reshape(-1,24,24)
        targets = targets.reshape(-1,24,24)
        
        np.savez('./viz/' + name + '_' + str(fold_num) , targets=targets, predictions=predictions, predictions2=prediction_wo_fieldwise, cm=confusion_matrix)

    else:
        class_labels = dataset.label_list_glob
        class_names = dataset.label_list_glob_name
        existing_class_labels = np.unique(targets)[1:]
    
        for i in range(1,len(class_acc)):
            cur_ind = class_labels.index(existing_class_labels[i])
            name = class_names[int(cur_ind)]
            print(name,' %.4f'%class_acc[i])
    
    
    return fieldwise_pix_accuracy


