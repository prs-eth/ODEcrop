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
from lib.constructODE import get_diffeq_solver

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import GRUCell, LSTMCell, RNNCellBase

from torch.distributions.normal import Normal
from torch.distributions import Independent
from torch.nn.parameter import Parameter
from lib.base_models import Baseline
from lib.base_models import create_classifier
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver

from lib.gru_ode import FullGRUODECell_Autonomous
from lib.RNNcells import STAR_unit, GRU_unit, GRU_standard_unit, LSTM_unit

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
# Nando's modified function of a multilayer ODE-RNN
class ML_ODE_RNN(Baseline):
	def __init__(self, input_dim, latent_dim, device = torch.device("cpu"),
		z0_diffeq_solver = None, n_gru_units = 100,  n_units = 100,
		concat_mask = False, obsrv_std = 0.1, use_binary_classif = False,
		classif_per_tp = False, n_labels = 1, train_classif_w_reconstr = False,
		RNNcell = 'gru_small', stacking = None, linear_classifier = False,
		ODE_sharing = True, RNN_sharing = False,
		include_topper = False, linear_topper = False,
		use_BN = True, resnet = False,
		ode_type="linear", ode_units=200, rec_layers=1, ode_method="dopri5",
		stack_order = None):

		Baseline.__init__(self, input_dim, latent_dim, device = device, 
			obsrv_std = obsrv_std, use_binary_classif = use_binary_classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = train_classif_w_reconstr)

		self.include_topper = include_topper
		self.resnet = resnet
		self.use_BN = use_BN
		ode_rnn_encoder_dim = latent_dim

		if ODE_sharing or RNN_sharing or self.resnet or self.include_topper:
			self.include_topper = True
			input_dim_first = latent_dim
		else:
			input_dim_first = input_dim

		if RNNcell=='lstm':
			ode_latents = int(latent_dim)*2
		else:
			ode_latents = int(latent_dim)

		#need one Encoder_z0_ODE_RNN per layer.
		self.ode_gru =[]
		self.z0_diffeq_solver =[]
		first_layer = True
		rnn_input = input_dim_first*2

		if stack_order is None: 
			stack_order = ["ode_rnn"]*stacking # a list of ode_rnn, star, gru, gru_small, lstm
		
		self.stacking = stacking
		if not (len(stack_order)==stacking): # stack_order argument must be as long as the stacking list
			print("Warning, the specified stacking order is not the same length as the number of stacked layers, taking stack-order as reference.")
			print("Stack-order: ", stack_order)
			print("Stacking argument: ", stacking)
			self.stacking = len(stack_order)

		# get the default ODE and RNN for the weightsharing
		# ODE stuff
		z0_diffeq_solver = get_diffeq_solver(ode_latents, ode_units, rec_layers, ode_method, ode_type="linear", device=device)
		
		# RNNcell
		if RNNcell=='gru':
			RNN_update = GRU_unit(latent_dim, rnn_input, n_units = n_gru_units, device=device).to(device)

		elif RNNcell=='gru_small':
			RNN_update = GRU_standard_unit(latent_dim, rnn_input, device=device).to(device)

		elif RNNcell=='lstm':
			RNN_update = LSTM_unit(latent_dim, rnn_input).to(device)

		elif RNNcell=="star":
			RNN_update = STAR_unit(latent_dim, rnn_input, n_units = n_gru_units).to(device)

		else:
			raise Exception("Invalid RNN-cell type. Hint: expdecay not available for ODE-RNN")


		# Put the layers it into the model
		for s in range(self.stacking):
			
			use_ODE = (stack_order[s]=="ode_rnn")

			if first_layer:
				# input and the mask
				layer_input_dimension = (input_dim_first)*2
				first_layer = False
				
			else:
				# otherwise we just take the latent dimension of the previous layer as the sequence
				layer_input_dimension = latent_dim*2

			# append the same z0_ODE-RNN for every layer
			
			if not RNN_sharing:
				
				if not use_ODE:
					vertical_rnn_input = layer_input_dimension + 2 # +2 for delta t and it's mask
					thisRNNcell = stack_order[s]
				else:
					vertical_rnn_input = layer_input_dimension
					thisRNNcell = RNNcell

				if thisRNNcell=='gru':
					#pdb.set_trace()
					RNN_update = GRU_unit(latent_dim, vertical_rnn_input, n_units = n_gru_units, device=device).to(device)

				elif thisRNNcell=='gru_small':
					RNN_update = GRU_standard_unit(latent_dim, vertical_rnn_input, device=device).to(device)

				elif thisRNNcell=='lstm':
					# two times latent dimension because of the cell state!
					RNN_update = LSTM_unit(latent_dim*2, vertical_rnn_input).to(device)

				elif thisRNNcell=="star":
					RNN_update = STAR_unit(latent_dim, vertical_rnn_input, n_units = n_gru_units).to(device)

				else:
					raise Exception("Invalid RNN-cell type. Hint: expdecay not available for ODE-RNN")

			if not ODE_sharing:

				if RNNcell=='lstm':
					ode_latents = int(latent_dim)*2
				else:
					ode_latents = int(latent_dim)

				z0_diffeq_solver = get_diffeq_solver(ode_latents, ode_units, rec_layers, ode_method, ode_type="linear", device=device)
				

			self.Encoder0 = Encoder_z0_ODE_RNN( 
				latent_dim = ode_rnn_encoder_dim, 
				input_dim = layer_input_dimension, 
				z0_diffeq_solver = z0_diffeq_solver, 
				n_gru_units = n_gru_units, 
				device = device,
				RNN_update = RNN_update,
				use_BN = use_BN,
				use_ODE = use_ODE
			).to(device)

			self.ode_gru.append( self.Encoder0 )
		
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
			
			utils.init_network_weights(self.topper)

			self.topper_bn = nn.BatchNorm1d(latent_dim)

		"""
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, n_units),
			nn.Tanh(),
			nn.Linear(n_units, input_dim),)
		utils.init_network_weights(self.decoder)
		"""

		z0_dim = latent_dim

		# get the end-of-sequence classifier
		if use_binary_classif: 
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels),
					nn.Softmax(dim=(2))
					)
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)

			if self.use_BN:
				self.bn_lasthidden = nn.BatchNorm1d(latent_dim)

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

					# create tensor of topperoutput, and fill in the corresponding locations
					data_topped = torch.zeros(n_traj, n_tp, self.latent_dim).to(self.device)
					if self.use_BN:
						data_topped[mask2[:,0], mask2[:,1]] = self.topper_bn( self.topper(pure_data[mask2[:,0], mask2[:,1]]) )
					else:
						data_topped[mask2[:,0], mask2[:,1]] = self.topper(pure_data[mask2[:,0], mask2[:,1]])

					# create mask with the new size
					new_mask = data_and_mask[:,:,self.input_dim:][:,:,0][:,:,None].repeat(1,1,self.latent_dim)

					#replace the data_and_mask
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

			# add the output as a residual, if it is a ResNet
			if self.resnet:
				latent_ys = latent_ys + input_sequence.unsqueeze(0)[:,:,:,:self.latent_dim]

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
				if self.use_BN:
					last_hidden = self.bn_lasthidden(last_hidden.squeeze()).unsqueeze(0)
				extra_info["label_predictions"] = self.classifier(last_hidden).squeeze(-1)

		# outputs shape: [n_traj_samples, n_traj, n_tp, n_dims]
		return outputs, extra_info