"""
author: Nando Metzger
metzgern@ethz.ch
"""

import os
from random import SystemRandom

import lib.utils as utils
from lib.utils import compute_loss_all_batches
from lib.utils import Bunch, get_optimizer, plot_confusion_matrix
from lib.construct import get_ODE_RNN_model, get_classic_RNN_model
from lib.ode_rnn import *
from lib.parse_datasets import parse_datasets

from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver

import torch.nn as nn
import torch
#from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from tqdm import tqdm
import pdb
import numpy as np
from hyperopt import STATUS_OK
import wandb

import pickle


def construct_and_train_model(config):
    # Create ODE-GRU model

    args = config["spec_config"][0]["args"]

    # namespace to dict
    argsdict = vars(args)

    print("")
    print("Testing hyperparameters:")
    for key in config.keys():
        if not key=='spec_config':
            argsdict[key] = config[key]
            print(key + " : " + str(argsdict[key]))
    print("")

    # namespace to dict
    args = Bunch(argsdict)

    # onrolle the other parameters:
    #Data_obj = config["spec_config"][0]["Data_obj"]
    file_name = config["spec_config"][0]["file_name"]
    experimentID = config["spec_config"][0]["experimentID"]
    input_command = config["spec_config"][0]["input_command"]
    Devices = config["spec_config"][0]["Devices"]
    num_gpus = config["spec_config"][0]["num_gpus"]
    num_seeds = config["spec_config"][0]["num_seeds"]

    num_gpus = max(num_gpus,1)
    if isinstance(args.batch_size, tuple):
        args.batch_size = int(args.batch_size[0])
    args.gru_units = int(args.gru_units)

    
    ##############################################################################
    ## set seed

    randID = int(SystemRandom().random()*10000)*1000
    ExperimentID = []
    for i in range(num_seeds):
        ExperimentID.append(randID + i)

    print("ExperimentID", ExperimentID)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    ##############################################################################
    # Dataset
    Data_obj = []
    for i, device in enumerate(Devices):
        Data_obj.append(parse_datasets(args, device))

    data_obj = Data_obj[0]

    input_dim = data_obj["input_dim"]
    classif_per_tp = False
    if ("classif_per_tp" in data_obj):
        # do classification per time point rather than on a time series as a whole
        classif_per_tp = data_obj["classif_per_tp"]

    if args.classif and (args.dataset == "hopper" or args.dataset == "periodic"):
        raise Exception("Classification task is not available for MuJoCo and 1d datasets")

    n_labels = 1
    if args.classif:
        if ("n_labels" in data_obj):
            n_labels = data_obj["n_labels"]
        else:
            raise Exception("Please provide number of labels for classification task")

    ##############################################################################
    # Create Model
    #pdb.set_trace()

    Model = []

    if True:
        for i in range(num_seeds):
            Model.append(get_ODE_RNN_model(args, Devices[0], input_dim, n_labels, classif_per_tp))
            
    if args.classic_rnn:
        for i in range(num_seeds):
            Model.append(get_classic_RNN_model(args, Devices[0], input_dim, n_labels, classif_per_tp))

    #print('ModelX:', Model)
    # "Magic" wandb model watcher
    wandb.watch(Model[0], "all")

    ##################################################################
    
    if args.tensorboard:
        #Validationwriter = []
        #Trainwriter = []
        for i in range(num_seeds):
            comment = '_'
            if args.classic_rnn:
                nntype = 'rnn'

            elif args.ode_rnn:
                nntype = 'ode'
            else:
                raise Exception("please select a model")
            
            RNNws_str = ""
            ODEws_str = ""
            bn_str = ""
            rs_str = ""
            if args.RNN_sharing:
                RNNws_str = "_RNNws"
            if args.ODE_sharing:
                ODEws_str = "_ODEws"
            if args.batchnorm:
                bn_str = "_BN"
            if args.resnet:
                rs_str = "_rs"

            comment = nntype + "_ns:" + str(args.n) + "_ba:" + str(args.batch_size) + "_ode-units:" + str(args.units) + "_gru-uts:" + str(args.gru_units) + "_lats:"+ str(args.latents) + "_rec-lay:" + str(args.rec_layers) + "_solver:" + str(args.ode_method) + "_seed:" +str(args.random_seed) + "_optim:" + str(args.optimizer) + "_stackin:" + str(args.stacking) + str(args.stack_order)+ ODEws_str + RNNws_str + bn_str + rs_str

            #validationtensorboard_dir = "runs/expID" + str(ExperimentID[i]) + "_VALID" + comment
            #Validationwriter.append( SummaryWriter(validationtensorboard_dir, comment=comment) )
            
            tensorboard_dir = "runs/expID" + str(ExperimentID[i]) + "_TRAIN" + comment
            #Trainwriter.append( SummaryWriter(tensorboard_dir, comment=comment) )
        
            print(tensorboard_dir)
            
    ##################################################################
    # Training
    
    Train_res = [None]*num_seeds
    Test_res = [None]*num_seeds
    Best_test_acc = [None]*num_seeds
    Best_test_acc_step = [None]*num_seeds

    #checkpoint = torch.load('/home/tmehmet/ODEcrop/saved_models/GRU.ckpt')
    checkpoint = torch.load('/home/tmehmet/ODEcrop/experiments/experiment_1659001_topscore.ckpt')
    Model[0].load_state_dict(checkpoint['state_dict'])

    for i in range(1):
        Train_res[i], Test_res[i], Best_test_acc[i], Best_test_acc_step[i] = test_it(
            [Model[i]],
            Data_obj,
            args,
            file_name,
            [ExperimentID[i]],
            #[Trainwriter[i]],
            #[Validationwriter[i]],
            input_command,
            [Devices[0]]
        )
    
    Test_acc = []
    for i in range(num_seeds):
        #pdb.set_trace()
        Test_acc.append(Test_res[i][0]["accuracy"])


    mean_test_acc = np.mean(Test_acc)
    var_test_acc = sum((abs(Test_acc - mean_test_acc)**2)/(num_seeds-1))

    mean_best_test_acc = np.mean(Best_test_acc)
    var_best_test_acc = sum((abs(Best_test_acc - mean_best_test_acc)**2)/(num_seeds-1))
    best_of_best_test_acc = np.max(Best_test_acc)
    best_of_best_test_acc_step = Best_test_acc_step[np.argmax(Best_test_acc)]

    # because it is fmin, we have to bring back some kind of loss, therefore 1-...
    return_dict = {
        'loss': 1-mean_best_test_acc,
        'loss_variance': var_best_test_acc,
        #'true_loss': 1-mean_test_acc,
        #'true_loss_variance':var_test_acc,
        'status': STATUS_OK,
        'num_seeds': num_seeds,
        'best_acc': best_of_best_test_acc,
        'best_peak_step': best_of_best_test_acc_step,
        'best_steps': Best_test_acc_step
    }

    print(return_dict)

    return return_dict

def test_it(
        Model,
        Data_obj,
        args,
        file_name,
        ExperimentID,
        #Trainwriter,
        #Validationwriter,
        input_command,
        Devices):

    """
    parameters:
        Model, #List of Models
        Data_obj, #List of Data_objects which live on different devices
        args,
        file_name,
        ExperimentID, #List of IDs
        trainwriter, #List of TFwriters
        validationwriter, #List of TFwriters
        input_command,
        Devices #List of devices
    """

    Ckpt_path = []
    Top_ckpt_path = []
    Best_test_acc = []
    Best_test_acc_step = []
    Logger = []
#    Optimizer = []
#    otherOptimizer = []
#    ODEOptimizer = []

    for i, device in enumerate(Devices):

        Ckpt_path.append( os.path.join(args.save, "experiment_" + str(ExperimentID[i]) + '.ckpt') )
        Top_ckpt_path.append( os.path.join(args.save, "experiment_" + str(ExperimentID[i]) + '_topscore.ckpt') )
        Best_test_acc.append(0)
        Best_test_acc_step.append(0)

        log_path = "logs/" + file_name + "_" + str(ExperimentID[i]) + ".log"
        if not os.path.exists("logs/"):
            utils.makedirs("logs/")
        Logger.append( utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__)) )
        Logger[i].info(input_command)
        
        

    num_batches = Data_obj[0]["n_train_batches"]
    #labels = Data_obj[0]["dataset_obj"].label_list

    #create empty lists
    num_gpus = len(Devices)
    train_res = [None] * num_gpus
    test_res = [None] * num_gpus
    label_dict = [None]* num_gpus

    # empty result placeholder
    somedict = {}
    test_res = [somedict]
    test_res[0]["accuracy"] = float(0)


    for itr in range(1):

        wait_until_kl_inc = 10
        if itr // num_batches < wait_until_kl_inc:
            kl_coef = 0.01
        else:
            kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

        if True :
            with torch.no_grad():

                for i, device in enumerate(Devices):
                    test_res[i], label_dict[i] = compute_loss_all_batches(Model[i], 
                        Data_obj[i]["test_dataloader"], args,
                        n_batches = Data_obj[i]["n_test_batches"],
                        experimentID = ExperimentID[i],
                        device = Devices[i],
                        n_traj_samples = 3, kl_coef = kl_coef)

                for i, device in enumerate(Devices):
    
                    # prepare GT labels and predictions
                    y_ref = label_dict[0]["correct_labels"].cpu()
                    y_pred = label_dict[0]["predict_labels"]


                    logdict = {
                        'validation': test_res[i]["accuracy"],
                        'validation_peak': Best_test_acc[i],
                        #'validation_peak_step': Best_test_acc_step[i],
                        
                        #'loss/validation': test_res[i]["loss"].detach(),
                        #'Confusionmatrix': conf_fig,

                        #'validation_cm' : sklearn_cm(y_ref, y_pred),
                        #'validation_precision': precision_score(y_ref, y_pred, average='macro'),
                        #'validation_recall': recall_score(y_ref, y_pred, average='macro'),
                        'validation_f1': f1_score(y_ref, y_pred, average='macro'),
                        'validation_kappa': cohen_kappa_score(y_ref, y_pred),
                    }
                    #wandb.log(logdict, step=itr*args.batch_size)
            print(logdict)
                    


    print(Best_test_acc[0], " at step ", Best_test_acc_step[0])


    
    return train_res, test_res, Best_test_acc[0], Best_test_acc_step[0]