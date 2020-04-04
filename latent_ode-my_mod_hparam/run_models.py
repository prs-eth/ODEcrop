###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova,
# Editor: Nando Metzger
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
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches

# Nando's additional libraries
from tqdm import tqdm
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials

from lib.training import construct_and_train_model, train_it
from lib.utils import hyperopt_summary

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=20000, help="Size of the dataset")
parser.add_argument('-validn',  type=int, default=4000, help="Size of the validation dataset")
parser.add_argument('--niters', type=int, default=1) # default=300
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=1000)
parser.add_argument('--viz', default=True, action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='crop', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")
parser.add_argument('--ode-rnn', default=True, action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-vae', default=False, action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=45, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=100, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=3, help="Number of layers in ODE func in recognition ODE") 
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=210, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=70, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', default="True", action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', default=False, action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")
parser.add_argument('--tensorboard',  action='store_true', default=True, help="monitor training with the help of tensorboard")
parser.add_argument('--ode-method', type=str, default='euler',
					help="Method of the ODE-Integrator. One of: 'explicit_adams', fixed_adams', 'adams', 'tsit5', 'dopri5', 'bosh3', 'euler', 'midpoint', 'rk4' , 'adaptive_heun' ")
parser.add_argument('--optimizer', type=str, default='adamax',
					help="Chose from: adamax (default), adagrad, adadelta, adam, adaw, sparseadam, ASGD, RMSprop, rprop, SGD")
					# working: adamax, adagrad, adadelta, adam, adaw, ASGD, rprop
					# not working sparseadam(need sparse gradients), LBFGS(missing closure), RMSprop(CE loss is NAN)

parser.add_argument('--num-seeds', type=int, default=3, help="Number of runs to average from. Default=3")
parser.add_argument('--num-search', type=int, default=1, help="Number of search steps to be executed")
parser.add_argument('--hparams', nargs='*', help="a set of: rec_layers, units, latents, gru_units, optimizer, lr, batch_size, ode_method")


args = parser.parse_args()

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

	##################################################################

	#Load checkpoint and evaluate the model
	if args.load is not None:
		#utils.get_ckpt_model(ckpt_path, model, device)
		utils.get_ckpt_model(top_ckpt_path, model, Devices[0])
		exit()

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

		#"rec_layers": hp.quniform('rec_layers', 1, 4, 1),
		#"units": hp.quniform('ode_units', 10, 400, 40), # default: 500
		#"latents": hp.quniform('latents', 15, 80, 5), # default: 35
		#"gru_units": hp.quniform('gru-units', 30, 120, 5), # default: 50
		#"optimizer": hp.choice('optimizer', optimizer_choice), 
		#"lr": hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
		#"batch_size": hp.qloguniform('batch_size', np.log(50), np.log(3000), 50), 
		#"random-seed":  hp.randint('seed', 5),
		#"ode-method": hp.choice('ODE_solver', solver_choice),
	}

	# Hyperparameters:
	# rec_layers, units, latents, gru_units, optimizer, lr, batch_size, ode-method

	if args.hparams is None:
		args.hparams = []

	if 'rec_layers' in args.hparams:
		hyper_config["rec_layers"] = hp.quniform('rec_layers', 1, 4, 1)
	
	if 'units' in args.hparams:
		hyper_config["units"] = hp.quniform('ode_units', 10, 400, 60) # default: 500
	
	if 'latents' in args.hparams:
		hyper_config["latents"] = hp.quniform('latents', 15, 100, 5) # default: 35

	if 'gru_units' in args.hparams:
		hyper_config["gru_units"] = hp.quniform('gru_units', 30, 120, 5) # default: 50

	if 'optimizer' in args.hparams:
		optimizer_choice =  ['adamax']  #['adamax', 'adagrad', 'adadelta', 'adam', 'adaw', 'ASGD', 'rprop', 'SGD'] RMSprop?
		print("optimizer choices: ", optimizer_choice)
		hyper_config["optimizer"] = hp.choice('optimizer', optimizer_choice)
	
	if 'lr' in args.hparams:
		hyper_config["lr"] = hp.loguniform('lr', np.log(0.0001), np.log(0.1))
	
	if 'batch_size' in args.hparams:
		hyper_config["batch_size"] = hp.qloguniform('batch_size', np.log(50), np.log(4000), 50), 
	
	if 'ode_method' in args.hparams:
		solver_choice = ['euler', 'dopri5'] #['explicit_adams', fixed_adams', 'adams', 'tsit5', 'dopri5', 'bosh3', 'euler', 'midpoint', 'rk4' , 'adaptive_heun']
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

		if 'optimizer' in args.hparams:
			print("Optimizer choices: ",optimizer_choice)
		if 'ode-method' in args.hparams:
			print("Solver choices: ", solver_choice)
		
		hyperopt_summary(trials)

	except Exception:
		hyperopt_summary(trials)
		traceback.print_exc(file=sys.stdout)

	if 'optimizer' in args.hparams:
		print("Optimizer choices: ",optimizer_choice)
	if 'ode-method' in args.hparams:
		print("Solver choices: ", solver_choice)
	hyperopt_summary(trials)

