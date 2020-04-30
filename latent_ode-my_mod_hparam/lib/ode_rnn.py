###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline
from lib.base_models import create_classifier

import pdb


class ODE_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
		z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
		concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
		classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False,
		RNNcell = 'gru'):

		Baseline.__init__(self, input_dim, latent_dim, device = device, 
			obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		ode_rnn_encoder_dim = latent_dim

		self.ode_gru = Encoder_z0_ODE_RNN( 
			latent_dim = ode_rnn_encoder_dim, 
			input_dim = (input_dim) * 2, # input and the mask
			z0_diffeq_solver = z0_diffeq_solver, 
			n_gru_units = n_gru_units, 
			device = device,
			RNNcell = RNNcell).to(device)

		self.z0_diffeq_solver = z0_diffeq_solver

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)

		utils.init_network_weights(self.decoder)


	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = None, mode = None):

		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for ODE-RNN")

		# time_steps_to_predict and truth_time_steps should be the same 
		assert(len(truth_time_steps) == len(time_steps_to_predict))
		assert(mask is not None)
		
		data_and_mask = data
		if mask is not None:
			data_and_mask = torch.cat([data, mask],-1)

		_, _, latent_ys, _ = self.ode_gru.run_odernn(
			data_and_mask, truth_time_steps, run_backwards = False)
		
		latent_ys = latent_ys.permute(0,2,1,3)
		last_hidden = latent_ys[:,:,-1,:]

		#assert(torch.sum(int_lambda[0,0,-1,:] <= 0) == 0.)

		# do not calculate it, because we don't use it anymore
		#outputs = self.decoder(latent_ys)
		outputs = torch.zeros_like(data_and_mask)[None, :, :][:,:,:,:data_and_mask.shape[-1]//2]
		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]
		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (latent_ys[:,:,-1,:], 0.0, latent_ys[:,:,-1,:])}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(latent_ys)
			else:
				extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info


# Nando's modified function of a multilayer ODE-RNN
class ML_ODE_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
		z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
		concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
		classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False,
		RNNcell = 'gru', stacking = 1, linear_classifier = False,
		weight_sharing=False, include_topper=False, linear_topper=False,
		use_BN=True):

		Baseline.__init__(self, input_dim, latent_dim, device = device, 
			obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.stacking = stacking
		self.include_topper = include_topper
		ode_rnn_encoder_dim = latent_dim

		if weight_sharing:
			# If we want to perform weight sharing, we need to have a topper to reduce the dimensionality of the input to the latent dim
			self.include_topper = True
			input_dim_first = latent_dim
		
		else:
			input_dim_first = input_dim

		#need one Encoder_z0_ODE_RNN per layer.
		self.ode_gru =[]
		first_layer = True

		for _ in range(stacking):
			if first_layer:
				# input and the mask
				input_dimension = (input_dim_first)*2

				#reset boolean to enter other loop
				first_layer = False
			else:
				# otherwise we just take the latent dimension of the previous layer as the sequence
				# the input dimension is the dimension of the lantent dimension of the lower level trajectory plus the masking of this
				# therefore two times the latent_dimension.
				input_dimension = latent_dim*2

			#append a different ODE-RNN for every layer
			self.ode_gru.append(
				Encoder_z0_ODE_RNN( 
					latent_dim = ode_rnn_encoder_dim, 
					input_dim = input_dimension, 
					z0_diffeq_solver = z0_diffeq_solver, 
					n_gru_units = n_gru_units, 
					device = device,
					RNNcell = RNNcell,
					use_BN = use_BN
				).to(device)
			)

		# construct topper
		if self.include_topper:
			if linear_topper:
				self.topper = nn.Sequential(
					nn.Linear(input_dim, latent_dim),
					nn.Tanh(),).to(device)
			else:
				self.topper = nn.Sequential(
					nn.Linear(input_dim, 100),
					nn.Tanh(),
					nn.Linear(100, latent_dim),
					nn.Tanh(),).to(device)

		self.z0_diffeq_solver = z0_diffeq_solver

		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)

		utils.init_network_weights(self.decoder)

		z0_dim = latent_dim

		if use_binary_classif: 
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels),
					nn.Softmax(dim=(2))
					)
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)

		self.device = device


	def get_reconstruction(self, time_steps_to_predict, data, truth_time_steps, 
		mask = None, n_traj_samples = None, mode = None):

		if (len(truth_time_steps) != len(time_steps_to_predict)) or (torch.sum(time_steps_to_predict - truth_time_steps) != 0):
			raise Exception("Extrapolation mode not implemented for ODE-RNN")

		# time_steps_to_predict and truth_time_steps should be the same 
		assert(len(truth_time_steps) == len(time_steps_to_predict))
		assert(mask is not None)
		
		data_and_mask = data
		if mask is not None:
			data_and_mask = torch.cat([data, mask],-1)

		All_latent_ys = []
		first_layer = True
		
		n_traj, n_tp, n_dims = data_and_mask.size()

		# run for every layer
		for s in range(self.stacking):
			
			if first_layer:

				# if it is the first RNN-layer, transform the dimensionality of the input down using the topper NN
				if self.include_topper:
					pure_data = data_and_mask[:,:, :self.input_dim]
					mask2 = torch.sum( data_and_mask[:,:,self.input_dim:].bool() , dim=2).nonzero()

					data_topped = torch.zeros(n_traj, n_tp, self.latent_dim).to(self.device)

					data_topped[mask2[:,0], mask2[:,1]] = self.topper(pure_data[mask2[:,0], mask2[:,1]])

					new_mask = data_and_mask[:,:,self.input_dim:][:,:,0][:,:,None].repeat(1,1,self.latent_dim)

					# TODO: Batchnormalization here

					data_and_mask = torch.cat([data_topped, new_mask],-1)

				input_sequence = data_and_mask

				first_layer = False
			else:
				new_latent = latent_ys[0,:,:,:]
				latent_dim = new_latent.shape[-1]
				latent_mask = mask[:,:,0].unsqueeze(2).repeat(1, 1, latent_dim)
				
				#destroy latent trajectoy informations that are not observed.
				new_latent[~latent_mask.bool()] = 0
				
				input_sequence = torch.cat([new_latent, latent_mask], -1)

			# run one trajectory of ODE-RNN for every stacking-layer "s"
			_, _, latent_ys, _ = self.ode_gru[s].run_odernn(
				input_sequence, truth_time_steps, run_backwards = False)

			latent_ys = latent_ys.permute(0,2,1,3)
			All_latent_ys.append(latent_ys)

		# get the last hidden state of the last latent ode-rnn trajectory
		last_hidden = All_latent_ys[-1][:,:,-1,:]

		# do not calculate it, because we don't use it anymore
		#outputs = self.decoder(latent_ys)
		outputs = torch.zeros_like(data)[None, :, :]
		# Shift outputs for computing the loss -- we should compare the first output to the second data point, etc.
		first_point = data[:,0,:]

		outputs = utils.shift_outputs(outputs, first_point)

		extra_info = {"first_point": (All_latent_ys[-1][:,:,-1,:], 0.0, All_latent_ys[-1][:,:,-1,:])}

		if self.use_binary_classif:
			if self.classif_per_tp:
				extra_info["label_predictions"] = self.classifier(latent_ys)
			else:
				extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info
