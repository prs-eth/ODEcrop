###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
# Code adapted from Yulia Rubanova, Latent ordinary differential equations for irregularly-sampled time series
###########################

import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

import time
import datetime
import argparse
import numpy as np
import pandas as pd
from random import SystemRandom
#from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils
from lib.plotting import *

#from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver

from lib.utils import compute_loss_all_batches

# Nando's additional libraries
from tqdm import tqdm
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials

from lib.training import construct_and_train_model, train_it
from lib.utils import hyperopt_summary

import wandb

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n', '--num_samples', type=int, default=8000, help="Size of the dataset")
parser.add_argument('-validn',  type=int, default=21000, help="Size of the validation dataset")
parser.add_argument('--niters', type=int, default=1) # default=300
parser.add_argument('--lr',  type=float, default=0.00762, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=700)
parser.add_argument('--viz', default=True, action='store_true', help="NOT IN USE! Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="NOT IN USE! ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='crop', help="Dataset to load. Available: crop, swisscrop")
parser.add_argument('--hp_search', default=False, type=bool,  help="Decides if the validation set (hsearch=True) or the test set should be used.")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="NOT IN USE! Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

#only for swissdata:
parser.add_argument('--step', type=int, default=1, help="intervall used for skipping observations in swissdata")
parser.add_argument('--trunc', type=int, default=9, help="Feature truncation in swissdata")
parser.add_argument('--swissdatatype', type=str, default="2", help="blank (default), 2 (for more selective cloud handling), 2_toplabels for the most frequent labels. (only works if accordingly preprocessed) ")
parser.add_argument('--singlepix', default=False,  type=bool, help="Applies batchnormalization to the outputs of the RNN-cells")
parser.add_argument('--noskip', action='store_true', help="If the flag is step, the dataloader will not sort out cloudy frames, and all the frames will be taken for the training.")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="NOT IN USE! Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), gru_small, expdecay, lstm, star")
parser.add_argument('--ode-type', default="linear", help="ODE Function type. Available: linear (default, use --rec-layers and --units to adjust size), gru")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")
parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-vae', default=False, action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=70, help="Size of the latent state")
parser.add_argument('--stacking', type=int, default=1, help="Number of hidden layer-trajectories that should be stacked. Only valid, if --stack-order is unused")
parser.add_argument('--stack-order', nargs='*', help="a set of (separated by blank spaces): ode_rnn, gru, lstm, small_gru, star. Repetitions of same method are possible")
parser.add_argument('-ODEws', '--ODE-sharing', default=False,  type=bool, help="Ties the weight together across different ODE between RNN-ODE-layers. Only useful if --stacking is larger than 2")
parser.add_argument('-RNNws', '--RNN-sharing', default=False,  type=bool, help="Ties the weight together across different RNN between RNN-ODE-layers. Only useful if --stacking is larger than 2")
parser.add_argument('-BN', '--batchnorm', default=False,  type=bool, help="Applies batchnormalization to the outputs of the RNN-cells")
parser.add_argument('-RN', '--resnet', default=False,  type=bool, help="Turns the stacked latent-trajectories into stacked residual trajectories.")

parser.add_argument('--rec-layers', type=int, default=2, help="Number of layers in ODE func in recognition ODE") 
parser.add_argument('-u', '--units', type=int, default=255, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")
parser.add_argument('-RI', '--nornnimputation', default=False,  type=bool, help="If false (default), for the baseline models (vanilla RNNs), the models additionally imputes delta t, which is the time since the last observation.")
parser.add_argument('--use_pos_encod', default=False,  type=bool, help="If false (default). Using a 2-dim positional encoding concatinated to the RNN's hidden state. Only available for vanilla RNNs.")
parser.add_argument('--use_pos_encod2', default=False,  type=bool, help="If false (default). Using a 2-dim positional encoding concatinated to the RNN's hidden state. Only available for vanilla RNNs.")

parser.add_argument('--linear-classif', default=False, type=bool, help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--topper', default=False, type=bool, help="If a topper to transform the input before the first trajectory should be used")
parser.add_argument('--linear-topper', default=False, type=bool, help="If using a topper, use a linear classifier instead of 1-layer NN, else a 2-layer NN is used")

parser.add_argument('--rec-dims', type=int, default=100, help="Dimensionality of the recognition model (RNN).")
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', default="True", action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

parser.add_argument('--tensorboard',  action='store_true', default=True, help="monitor training with the help of tensorboard")
parser.add_argument('-v', type=int, default=2, help="Verbosity of training. 0:=silence, 1:= standard progressbar, 2:= progressbar with additional content")
parser.add_argument('--val_freq', type=int, default=100, help="Validate every ... batches")

parser.add_argument('--ode-method', type=str, default='euler',
					help="Method of the ODE-Integrator. One of: 'explicit_adams', fixed_adams', 'adams', 'tsit5', 'dopri5', 'bosh3', 'euler', 'midpoint', 'rk4' , 'adaptive_heun' ")
parser.add_argument('--optimizer', type=str, default='adamax',
					help="Chose from: adamax (default), adagrad, adadelta, adam, adaw, sparseadam, ASGD, RMSprop, rprop, SGD")
					# working: adamax, adagrad, adadelta, adam, adaw, ASGD, rprop, RMSprop
					# not working sparseadam(need sparse gradients), LBFGS(missing closure)
parser.add_argument('--lrdecay',  type=float, default=0.9995, help="For the Learning rate scheduler")

parser.add_argument('--trainsub',  type=float, default=1., help="Downsampling of the Training dataset. How many data points should be left. [0,1]")
parser.add_argument('--testsub',  type=float, default=1., help="Downsampling of the Testing dataset. How many data points should be left. [0,1]")

parser.add_argument('--num-seeds', type=int, default=1, help="Number of runs to average from. Default=1")
parser.add_argument('--num-search', type=int, default=1, help="Number of search steps to be executed")
parser.add_argument('--hparams', nargs='*', help="a set of (separated by blank spaces): rec_layers, units, latents, gru_units, optimizer, lr, batch_size, ode_method")


args = parser.parse_args()
args.n = args.num_samples

#print("I'm running on GPU") if torch.cuda.is_available() else print("I'm running on CPU")
num_gpus = torch.cuda.device_count()

if num_gpus> 0:
	
	print("I'm counting gpu's: ", num_gpus)
	print("Means I will train ", num_gpus , " models, with different random seeds")


Devices = []
if num_gpus>0:
	for ind in range(num_gpus):
		Devices.append("cuda:" + str(ind))

	print("My Devices: ", Devices)
else:
	Devices.append( torch.device("cuda:0" if torch.cuda.is_available() else "cpu") )

file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

# Wandb initialization after the trail is lauched!
import wandb
wandb.init(project="odecropclassification",config=args, sync_tensorboard=True, entity="cropteam", group=args.dataset)

#####################################################################################################

if __name__ == '__main__':

	experimentID = args.load

	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")
	utils.makedirs("vis/")

	#################################################################
	# Hyperparameter Optimization
	
	num_seeds = args.num_seeds

	# create a specification dictionary for training
	spec_config = {
			"args": args,
			#"Data_obj": Data_obj,
			#"args": args,
			"file_name": file_name,
			"experimentID": experimentID,
			"input_command": input_command,
			"Devices": Devices,
			"num_gpus": num_gpus,
			"num_seeds": args.num_seeds
		},
	
		
	hyper_config = {
		"spec_config": spec_config, # fixed argument space
	}

	# Hyperparameters: rec_layers, units, latents, gru_units, optimizer, lr, batch_size, ode-method
	if args.hparams is None:
		args.hparams = []

	if 'rec_layers' in args.hparams:
		hyper_config["rec_layers"] = hp.quniform('rec_layers', 1, 2, 1)
	
	if 'units' in args.hparams:
		hyper_config["units"] = hp.quniform('ode_units', 10, 350, 5) # default: 500?
	
	if 'latents' in args.hparams:
		hyper_config["latents"] = hp.quniform('latents', 20, 300, 5) # default: 100?

	if 'gru_units' in args.hparams:
		hyper_config["gru_units"] = hp.quniform('gru_units', 10, 300, 3) # default: 50?

	if 'optimizer' in args.hparams:
		optimizer_choice =  ['adam']  #['adamax', 'adagrad', 'adadelta', 'adam', 'adaw', 'ASGD', 'rprop', 'SGD', 'RMSprop'] RMSprop?
		print("optimizer choices: ", optimizer_choice)
		hyper_config["optimizer"] = hp.choice('optimizer', optimizer_choice)
	
	if 'lr' in args.hparams:
		hyper_config["lr"] = hp.loguniform('lr', np.log(0.004), np.log(0.011))
	
	if 'batch_size' in args.hparams:
		hyper_config["batch_size"] = hp.qloguniform('batch_size', np.log(150), np.log(1000), 10), 
	
	if 'ode_method' in args.hparams:
		solver_choice = ['dopri5'] #['explicit_adams', fixed_adams', 'adams', 'tsit5', 'dopri5', 'bosh3', 'euler', 'midpoint', 'rk4' , 'adaptive_heun']
		print("Solver choices: ", solver_choice)
		hyper_config["ode_method"] = hp.choice('ODE_solver', solver_choice)
	
	try:
		trials = Trials()
		best = fmin(construct_and_train_model,
			hyper_config,
			trials=trials,
			algo=tpe.suggest,
			max_evals=args.num_search)

	except KeyboardInterrupt:
		best=None

		print("ODE-RNN ", args.ode_rnn)
		print("Classic-RNN: ", args.classic_rnn)
		print("Stacked layers: ", args.stacking)
		print("Stacked layers: ", args.stack_order)
		print("ODE-type: ", args.ode_type)
		print("RNN-cell: ", args.rnn_cell)
		print("Weight-sharing ODE: ", args.ODE_sharing)
		print("Weight-sharing RNN: ", args.RNN_sharing)
		print("Using BN: ", args.batchnorm)
		print("defaut adapted LR's!!")

		if 'optimizer' in args.hparams:
			print("Optimizer choices: ", optimizer_choice)
		else:
			print("Used Optimizer:", args.optimizer)

		if 'ode_method' in args.hparams:
			print("ODE-Solver choices: ", solver_choice)
		else:
			print("Used Ode-Solver", args.ode_method)
			
		hyperopt_summary(trials)

	except Exception:

		print("ODE-RNN ", args.ode_rnn)
		print("Classic-RNN: ", args.classic_rnn)
		print("Stacked layers: ", args.stacking)
		print("Stacked layers: ", args.stack_order)
		print("ODE-type: ", args.ode_type)
		print("RNN-cell: ", args.rnn_cell)
		print("Weight-sharing ODE: ", args.ODE_sharing)
		print("Weight-sharing RNN: ", args.RNN_sharing)
		print("Using BN: ", args.batchnorm)
		print("defaut adapted LR's!!")

		if 'optimizer' in args.hparams:
			print("Optimizer choices: ", optimizer_choice)
		else:
			print("Used Optimizer:", args.optimizer)

		if 'ode_method' in args.hparams:
			print("ODE-Solver choices: ", solver_choice)
		else:
			print("Used Ode-Solver", args.ode_method)

		hyperopt_summary(trials)

		traceback.print_exc(file=sys.stdout)

	print("ODE-RNN ", args.ode_rnn)
	print("Classic-RNN: ", args.classic_rnn)
	print("Stacked layers: ", args.stacking)
	print("Stacked layers: ", args.stack_order)
	print("ODE-type: ", args.ode_type)
	print("RNN-cell: ", args.rnn_cell)
	print("Weight-sharing ODE: ", args.ODE_sharing)
	print("Weight-sharing RNN: ", args.RNN_sharing)
	print("Using BN: ", args.batchnorm)
	print("defaut adapted LR's!!")

	if 'optimizer' in args.hparams:
		print("Optimizer choices: ",optimizer_choice)
	else:
		print("Used Optimizer:", args.optimizer)

	if 'ode_method' in args.hparams:
		print("ODE-Solver choices: ", solver_choice)
	else:
		print("Used Ode-Solver", args.ode_method)

	hyperopt_summary(trials)

