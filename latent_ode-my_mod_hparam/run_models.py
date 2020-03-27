###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova,
# Editor: Nando Metzger
###########################

import os
import sys
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

import ray
from ray import tune, init
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from lib.training import construct_and_train_model, train_it


# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=300000, help="Size of the dataset")
parser.add_argument('-validn',  type=int, default=60000, help="Size of the validation dataset")
parser.add_argument('--niters', type=int, default=1) # default=300
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=2000)
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

parser.add_argument('-l', '--latents', type=int, default=15, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=100, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=4, help="Number of layers in ODE func in recognition ODE") 
parser.add_argument('--gen-layers', type=int, default=2, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=500, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=50, help="Number of units per layer in each of GRU update networks")

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


args = parser.parse_args()

#print("I'm running on GPU") if torch.cuda.is_available() else print("I'm running on CPU")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*10000000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	top_ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '_topscore.ckpt')
	best_test_acc = 0

	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, device)
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

	##################################################################
	# Create the model
	obsrv_std = 0.01
	if args.dataset == "hopper":
		obsrv_std = 1e-3 

	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.rnn_vae:
		if args.poisson: #not used
			print("Poisson process likelihood not implemented for RNN-VAE: ignoring --poisson")

		# Create RNN-VAE model
		model = RNN_VAE(input_dim, args.latents, 
			device = device, 
			rec_dims = args.rec_dims, 
			concat_mask = True, 
			obsrv_std = obsrv_std,
			z0_prior = z0_prior,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			n_units = args.units,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)


	elif args.classic_rnn: #not used
		if args.poisson:
			print("Poisson process likelihood not implemented for RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for standard RNN not implemented")
		# Create RNN model
		model = Classic_RNN(input_dim, args.latents, device, 
			concat_mask = True, obsrv_std = obsrv_std,
			n_units = args.units,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			linear_classifier = args.linear_classif,
			input_space_decay = args.input_decay,
			cell = args.rnn_cell,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
		
	elif args.ode_rnn: # using this thing
		# Create ODE-GRU model
		n_ode_gru_dims = args.latents
		method = args.ode_method
		#print(args.ode_method)
		
		if args.poisson:
			print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

		if args.extrap:
			raise Exception("Extrapolation for ODE-RNN not implemented")

		ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = n_ode_gru_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, method, args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
	
		model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
			z0_diffeq_solver = z0_diffeq_solver, n_gru_units = args.gru_units,
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet")
			).to(device)
	elif args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	##################################################################

	if args.viz:
		viz = Visualizations(device)
	
	##################################################################
	
	if args.tensorboard:
		comment = '_'
		if args.classic_rnn:
			nntype = 'rnn'

		elif args.ode_rnn:
			nntype = 'ode'

		comment = nntype + "_ns:" + str(args.n) + "_ba:" + str(args.batch_size) + "_uts:" + str(args.units) + "_gru-uts:" + str(args.gru_units) + "_lats:"+ str(args.latents) + "_rec-dims:" + str(args.rec_dims) + "_rec-lay:" + str(args.rec_layers) + "_solver" + str(args.ode_method) + "_seed" +str(args.random_seed)

		validationtensorboard_dir = "runs/expID" + str(experimentID) + "_VALID" + comment
		validationwriter = SummaryWriter(validationtensorboard_dir, comment=comment)
		
		tensorboard_dir = "runs/expID" + str(experimentID) + "_TRAIN" + comment
		trainwriter = SummaryWriter(tensorboard_dir, comment=comment)
		
	##################################################################
	
	#Load checkpoint and evaluate the model
	if args.load is not None:
		#utils.get_ckpt_model(ckpt_path, model, device)
		utils.get_ckpt_model(top_ckpt_path, model, device)
		exit()

	##################################################################
	# Training

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]

	############

	# initialize ray to run local
	init(local_mode=True)


	config = {
		"config": {
			"data_obj": data_obj,
			"args": args,
			"file_name": file_name,
			"optimizer": optimizer,
			"experimentID": experimentID,
			"trainwriter": trainwriter,
			"validationwriter": validationwriter
		}
		,
		"test_config": {
			"iterations": 100,
		},
				
	}

	space = {
		"rec_layers": (1, 6),
	}

	sched = AsyncHyperBandScheduler(time_attr="training_iteration", metric="mean_accuracy", mode="max")

	
	config = {
		#this dictionairy in dictionary does not work
		"spec_config":{
			"args": args,
			"data_obj": data_obj,
			"args": args,
			"file_name": file_name,
			"optimizer": optimizer,
			"experimentID": experimentID,
			"trainwriter": trainwriter,
			"validationwriter": validationwriter,
			"input_dim": input_dim
		},

		"rec_layers":  tune.sample_from(lambda _: np.random.choice(range(1,7)) ),
		"rec_dims":  tune.sample_from(lambda _: np.random.choice(range(1,100))),
	}
	
	#construct_and_train_model(config)

	search_alg = BayesOptSearch(
		space,
		max_concurrent=1,
		metric="mean_accuracy",
		mode="max",
		utility_kwargs={
			"kind": "ucb",
			"kappa": 2.5,
			"xi": 0.0
		})



	#construct_and_train_model(config)

	analysis = tune.run(
		construct_and_train_model,
		name=str(experimentID),
		#scheduler=sched,
		#search_alg=search_alg,

		stop={
			"training_iteration": 25
		},

		config=config
	)

	print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
	
	############

						
	validationwriter.close()
	trainwriter.close()
	
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)

