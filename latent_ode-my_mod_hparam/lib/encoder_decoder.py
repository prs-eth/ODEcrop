###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
import math
from torch.nn.modules.rnn import LSTM, GRU
from torch.distributions import Categorical, Normal

import lib.utils as utils
import lib.utils as utils
from lib.utils import get_device

from lib.RNNcells import STAR_unit, GRU_unit, GRU_standard_unit, LSTM_unit

import pdb


class Encoder_z0_RNN(nn.Module):
	def __init__(self, latent_dim, input_dim, lstm_output_size = 20, 
		use_delta_t = True, device = torch.device("cpu")):
		
		super(Encoder_z0_RNN, self).__init__()
	
		self.gru_rnn_output_size = lstm_output_size
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.use_delta_t = use_delta_t

		self.hiddens_to_z0 = nn.Sequential(
		   nn.Linear(self.gru_rnn_output_size, 50),
		   nn.Tanh(),
		   nn.Linear(50, latent_dim * 2),)

		utils.init_network_weights(self.hiddens_to_z0)

		input_dim = self.input_dim

		if use_delta_t:
			self.input_dim += 1
		self.gru_rnn = GRU(self.input_dim, self.gru_rnn_output_size).to(device)

	def forward(self, data, time_steps, run_backwards = True):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		# data shape: [n_traj, n_tp, n_dims]
		# shape required for rnn: (seq_len, batch, input_size)
		# t0: not used here
		n_traj = data.size(0)

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		data = data.permute(1,0,2) 

		if run_backwards:
			# Look at data in the reverse order: from later points to the first
			data = utils.reverse(data)

		if self.use_delta_t:
			delta_t = time_steps[1:] - time_steps[:-1]
			if run_backwards:
				# we are going backwards in time with
				delta_t = utils.reverse(delta_t)
			# append zero delta t in the end
			delta_t = torch.cat((delta_t, torch.zeros(1).to(self.device)))
			delta_t = delta_t.unsqueeze(1).repeat((1,n_traj)).unsqueeze(-1)
			data = torch.cat((delta_t, data),-1)

		outputs, _ = self.gru_rnn(data)

		# LSTM output shape: (seq_len, batch, num_directions * hidden_size)
		last_output = outputs[-1]

		self.extra_info ={"rnn_outputs": outputs, "time_points": time_steps}

		mean, std = utils.split_last_dim(self.hiddens_to_z0(last_output))
		std = std.abs()

		assert(not torch.isnan(mean).any())
		assert(not torch.isnan(std).any())

		return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_ODE_RNN(nn.Module):
	# Derive z0 by running ode backwards.
	# For every y_i we have two versions: encoded from data and derived from ODE by running it backwards from t_i+1 to t_i
	# Compute a weighted sum of y_i from data and y_i from ode. Use weighted y_i as an initial value for ODE runing from t_i to t_i-1
	# Continue until we get to z0
	def __init__(self, latent_dim, input_dim, z0_diffeq_solver = None, 
		z0_dim = None, RNN_update = None, 
		n_gru_units = 100, device = torch.device("cpu"),
		RNNcell = 'gru', use_BN=True):
		
		super(Encoder_z0_ODE_RNN, self).__init__()

		if z0_dim is None:
			self.z0_dim = latent_dim
		else:
			self.z0_dim = z0_dim

		self.RNNcell = RNNcell

		rnn_input = input_dim

		if RNN_update is None:

			if self.RNNcell=='gru':
				self.RNN_update = GRU_unit(latent_dim, rnn_input, n_units = n_gru_units, device=device).to(device)

			elif self.RNNcell=='gru_small':
				self.RNN_update = GRU_standard_unit(latent_dim, rnn_input, device=device).to(device)

			elif self.RNNcell=='lstm':
				self.RNN_update = LSTM_unit(latent_dim, rnn_input).to(device)

			elif self.RNNcell=="star":
				self.RNN_update = STAR_unit(latent_dim, rnn_input, n_units = n_gru_units).to(device)

			else:
				raise Exception("Invalid RNN-cell type. Hint: expdecay not available for ODE-RNN")

		else:
			self.RNN_update = RNN_update

		self.ode_bn0 = nn.BatchNorm1d(latent_dim)
		self.ode_bn1 = nn.BatchNorm1d(latent_dim)

		self.use_BN = use_BN
		self.z0_diffeq_solver = z0_diffeq_solver
		self.latent_dim = latent_dim
		self.input_dim = input_dim
		self.device = device
		self.extra_info = None

		self.transform_z0 = nn.Sequential(
			nn.Linear(latent_dim * 2, 100),
			nn.Tanh(),
			nn.Linear(100, self.z0_dim * 2),)
		utils.init_network_weights(self.transform_z0)

		self.output_bn = nn.BatchNorm1d(latent_dim)


	def forward(self, data, time_steps, run_backwards = True, save_info = False):
		# data, time_steps -- observations and their time stamps
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 
		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		n_traj, n_tp, n_dims = data.size()
		if len(time_steps) == 1:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)
			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(self.device)

			xi = data[:,0,:].unsqueeze(0)

			last_yi, last_yi_std = self.RNN_update(prev_y, prev_std, xi)
			extra_info = None
		else:
			
			last_yi, last_yi_std, _, extra_info = self.run_odernn(
				data, time_steps, run_backwards = run_backwards,
				save_info = save_info)
			
		means_z0 = last_yi.reshape(1, n_traj, self.latent_dim)
		std_z0 = last_yi_std.reshape(1, n_traj, self.latent_dim)

		mean_z0, std_z0 = utils.split_last_dim( self.transform_z0( torch.cat((means_z0, std_z0), -1)))
		std_z0 = std_z0.abs()
		if save_info:
			self.extra_info = extra_info

		return mean_z0, std_z0


	def run_odernn(self, data, time_steps, 
		run_backwards = True, save_info = False):
		# IMPORTANT: assumes that 'data' already has mask concatenated to it 

		n_traj, n_tp, n_dims = data.size()
		extra_info = []

		t0 = time_steps[-1]
		if run_backwards:
			t0 = time_steps[0]

		device = get_device(data)

		if self.RNNcell=='lstm':
			prev_h = torch.zeros((1, n_traj, self.latent_dim//2)).to(device)
			prev_h_std = torch.zeros((1, n_traj, self.latent_dim//2)).to(device)

			ci = torch.zeros((1, n_traj, self.latent_dim//2)).to(device)
			ci_std = torch.zeros((1, n_traj, self.latent_dim//2)).to(device)
			
			#concatinate cell state and hidden state
			prev_y = torch.cat([prev_h, ci], -1)
			prev_std = torch.cat([prev_h_std, ci_std], -1)
		else:
			prev_y = torch.zeros((1, n_traj, self.latent_dim)).to(device)
			prev_std = torch.zeros((1, n_traj, self.latent_dim)).to(device)
		
		prev_t, t_i = time_steps[-1] + 0.01,  time_steps[-1]

		interval_length = time_steps[-1] - time_steps[0]
		minimum_step = interval_length / 50 # maybe have to modify minimum time step

		#print("minimum step: {}".format(minimum_step))

		assert(not torch.isnan(data).any())
		assert(not torch.isnan(time_steps).any())

		latent_ys = []

		# Run ODE backwards and combine the y(t) estimates using gating
		time_points_iter = range(0, len(time_steps))
		if run_backwards:
			time_points_iter = reversed(time_points_iter)

		for i in time_points_iter:

			#TODO: Include inplementationin case of no ODE function

			if (prev_t - t_i) < minimum_step:
				time_points = torch.stack((prev_t, t_i))
				inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

				assert(not torch.isnan(inc).any())

				ode_sol = prev_y + inc
				ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

				assert(not torch.isnan(ode_sol).any())
			else:
				n_intermediate_tp = max(2, ((prev_t - t_i) / minimum_step).int()) # get steps in between

				time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp)
				ode_sol = self.z0_diffeq_solver(prev_y, time_points)

				assert(not torch.isnan(ode_sol).any())

			if torch.mean(ode_sol[:, :, 0, :]  - prev_y) >= 0.001:
				print("Error: first point of the ODE is not equal to initial value")
				print(torch.mean(ode_sol[:, :, 0, :]  - prev_y))
				exit()
			#assert(torch.mean(ode_sol[:, :, 0, :]  - prev_y) < 0.001)

			yi_ode = ode_sol[:, :, -1, :]
			xi = data[:,i,:].unsqueeze(0)
			
			if self.RNNcell=='lstm':
				h_i_ode = yi_ode[:,:,:self.latent_dim//2]
				c_i_ode = yi_ode[:,:,self.latent_dim//2:]
				h_c_lstm = (h_i_ode, c_i_ode)

				# actually this is a LSTM update here:
				outi, yi_std = self.RNN_update(h_c_lstm, prev_std, xi)

				# the RNN cell is a LSTM and outi:=(yi,ci), we only need h as latent dim
				yi = torch.cat([outi[0], outi[1]], -1)
			else:

				# GRU-unit: the output is directly the hidden state
				yi, yi_std = self.RNN_update(yi_ode, prev_std, xi)

			prev_y, prev_std = yi, yi_std
			prev_t, t_i = time_steps[i],  time_steps[i-1]

			latent_ys.append(yi)

			if save_info:
				d = {"yi_ode": yi_ode.detach(), #"yi_from_data": yi_from_data,
					 "yi": yi.detach(), "yi_std": yi_std.detach(), 
					 "time_points": time_points.detach(), "ode_sol": ode_sol.detach()}
				extra_info.append(d)

		latent_ys = torch.stack(latent_ys, 1)

		#BatchNormalization for the outputs
		if self.use_BN:
			latent_ys = self.output_bn(latent_ys.squeeze().permute(0,2,1)).permute(0,2,1).unsqueeze(0)
			#print(self.output_bn.running_mean)
			#print(self.output_bn.running_var)

		assert(not torch.isnan(yi).any())
		assert(not torch.isnan(yi_std).any())

		return yi, yi_std, latent_ys, extra_info



class Decoder(nn.Module):
	def __init__(self, latent_dim, input_dim):
		super(Decoder, self).__init__()
		# decode data from latent space where we are solving an ODE back to the data space

		decoder = nn.Sequential(
		   nn.Linear(latent_dim, input_dim),)

		utils.init_network_weights(decoder)	
		self.decoder = decoder

	def forward(self, data):
		return self.decoder(data)


