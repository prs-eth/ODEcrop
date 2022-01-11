###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
###########################

import lib.utils as utils

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
#from torch.nn import Parameter
#from torch import Tensor

import numpy as np
import pdb

import math
	

class STAR_unit(nn.Module):

	"""
	Code based on:
	Gating Revisited: Deep Multi-layer RNNs That Can Be Trained
	https://arxiv.org/abs/1911.11033
	"""

	def __init__(self, hidden_size, input_size,
		n_units=0, bias=True, use_BN=False):
		super(STAR_unit, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.bias = bias
		
		if n_units==0:
			self.x_K = nn.Linear(input_size,  hidden_size, bias=bias)
			self.x_z = nn.Linear(input_size,  hidden_size, bias=bias)
			self.h_K = nn.Linear(hidden_size,  hidden_size, bias=bias)

			init.orthogonal_(self.x_K.weight) 
			init.orthogonal_(self.x_z.weight)
			init.orthogonal_(self.h_K.weight)

			self.x_K.bias.data.fill_(0.)
			self.x_z.bias.data.fill_(0.)
			#self.h_K.bias.data.fill_(0.)
		else:

			self.x_K = nn.Sequential(
				nn.Linear(input_size, n_units),
				nn.Tanh(),
				nn.Linear(n_units, hidden_size))
			utils.init_network_weights(self.x_K, initype="ortho")

			self.x_z = nn.Sequential(
				nn.Linear(input_size, n_units),
				nn.Tanh(),
				nn.Linear(n_units, hidden_size))
			utils.init_network_weights(self.x_z, initype="ortho")

			self.h_K = nn.Sequential(
				nn.Linear(hidden_size, n_units),
				nn.Tanh(),
				nn.Linear(n_units, hidden_size))
			utils.init_network_weights(self.h_K, initype="ortho")
		
		self.use_BN = use_BN

		if self.use_BN:
			self.bn_x_K = nn.BatchNorm1d(hidden_size)
			self.bn_x_z = nn.BatchNorm1d(hidden_size)
			self.bn_h_K = nn.BatchNorm1d(hidden_size)

#		init.kaiming_normal_(self.x_K.weight) 
#		init.kaiming_normal_(self.x_z.weight)
#		init.kaiming_normal_(self.h_K.weight)

		#bias_f= np.log(np.random.uniform(1,784,hidden_size))
		#bias_f = torch.Tensor(bias_f)  
		#self.bias_K = Variable(bias_f.cuda(), requires_grad=True)

	def forward(self, hidden, y_std, x, masked_update=True):
		
		#getting the mask
		n_data_dims = x.size(-1)//2
		mask = x[:, :, n_data_dims:]
		utils.check_mask(x[:, :, :n_data_dims], mask)
		mask = (torch.sum(mask, -1, keepdim = True) > 0).float()


		gate_x_K = self.x_K(x) 			# return size torch.Size([1, batch_size, latent_dim])
		gate_x_z = self.x_z(x) 			# return size torch.Size([1, batch_size, latent_dim])
		gate_h_K = self.h_K(hidden)		# return size torch.Size([1, batch_size, latent_dim])
		
		gate_x_K = gate_x_K.squeeze()
		gate_x_z = gate_x_z.squeeze()
		gate_h_K = gate_h_K.squeeze()

		if self.use_BN:
			if torch.sum(mask.float())>1:
				gate_x_K[mask.squeeze().bool()] = self.bn_x_K(gate_x_K[mask.squeeze().bool()])
				gate_x_z[mask.squeeze().bool()] = self.bn_x_z(gate_x_z[mask.squeeze().bool()])
				gate_h_K[mask.squeeze().bool()] = self.bn_h_K(gate_h_K[mask.squeeze().bool()])

		K_gain = torch.sigmoid(gate_x_K + gate_h_K)
		z = torch.tanh(gate_x_z)
		
		h_new = hidden + K_gain * ( z - hidden) 
		h_new = torch.tanh(h_new)
		
		# Masked update
		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			

			assert(not torch.isnan(mask).any())

			h_new = mask * h_new + (1-mask) * hidden

			if torch.isnan(h_new).any():
				print("new_y is nan!")
				print(mask)
				print(hidden)
				print(h_new)
				exit()

		return h_new, y_std


# GRU description: 
# http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/
class GRU_unit(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		n_units = 100,
		device = torch.device("cpu")):
		super(GRU_unit, self).__init__()

		if update_gate is None:
			self.update_gate = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate

		if reset_gate is None:
			self.reset_gate = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			self.new_state_net = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, n_units),
			   nn.Tanh(),
			   nn.Linear(n_units, latent_dim * 2))
			utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, y_mean, y_std, x, masked_update = True):
		
		y_concat = torch.cat([y_mean, y_std, x], -1)

		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
		#pdb.set_trace()
		new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		new_state_std = new_state_std.abs()

		new_y = (1-update_gate) * new_state + update_gate * y_mean
		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

		assert(not torch.isnan(new_y).any())

		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			utils.check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_y = mask * new_y + (1-mask) * y_mean
			new_y_std = mask * new_y_std + (1-mask) * y_std

			if torch.isnan(new_y).any():
				print("new_y is nan!")
				print(mask)
				print(y_mean)
				print(prev_new_y)
				exit()

		new_y_std = new_y_std.abs()
		return new_y, new_y_std



class GRU_standard_unit(nn.Module):
	def __init__(self, latent_dim, input_dim, 
		update_gate = None,
		reset_gate = None,
		new_state_net = None,
		device = torch.device("cpu")):
		super(GRU_standard_unit, self).__init__()

		if update_gate is None:
			self.update_gate = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.update_gate)
		else: 
			self.update_gate  = update_gate

		if reset_gate is None:
			self.reset_gate = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, latent_dim),
			   nn.Sigmoid())
			utils.init_network_weights(self.reset_gate)
		else: 
			self.reset_gate  = reset_gate

		if new_state_net is None:
			self.new_state_net = nn.Sequential(
			   nn.Linear(latent_dim * 2 + input_dim, latent_dim * 2),
			   )
			utils.init_network_weights(self.new_state_net)
		else: 
			self.new_state_net  = new_state_net


	def forward(self, y_mean, y_std, x, masked_update = True):
		
		y_concat = torch.cat([y_mean, y_std, x], -1)
		
		update_gate = self.update_gate(y_concat)
		reset_gate = self.reset_gate(y_concat)
		concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)
		
		new_state, new_state_std = utils.split_last_dim(self.new_state_net(concat))
		new_state_std = new_state_std.abs()

		new_y = (1-update_gate) * new_state + update_gate * y_mean
		new_y_std = (1-update_gate) * new_state_std + update_gate * y_std

		assert(not torch.isnan(new_y).any())

		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			utils.check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_y = mask * new_y + (1-mask) * y_mean
			new_y_std = mask * new_y_std + (1-mask) * y_std

			if torch.isnan(new_y).any():
				print("new_y is nan!")
				print(mask)
				print(y_mean)
				print(prev_new_y)
				exit()

		new_y_std = new_y_std.abs()
		return new_y, new_y_std




# Implement a LSTM cell
# LSTM-unit code inspired by Ricard's answer to https://stackoverflow.com/questions/50168224/does-a-clean-and-extendable-lstm-implementation-exists-in-pytorch
class LSTM_unit(nn.Module):

	def __init__(self, hidden_size, input_size):
		super(LSTM_unit, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size//2
		self.i2h = nn.Linear(input_size, 4 * self.hidden_size)
		self.h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
		self.reset_parameters()

	def reset_parameters(self):
		std = 1.0 / math.sqrt(self.hidden_size)
		for w in self.parameters():
			w.data.uniform_(-std, std)

	def forward(self, y_mean, y_std, x, masked_update = True):
		#forward(self, x, hidden):

		h, c = y_mean
		h = h.view(h.size(1), -1)
		c = c.view(c.size(1), -1)
		x_short = x.view(x.size(1), -1)

		# Linear mappings
		preact = self.i2h(x_short) + self.h2h(h)

		# activations
		gates = preact[:, :3 * self.hidden_size].sigmoid()
		g_t = preact[:, 3 * self.hidden_size:].tanh()
		i_t = gates[:, :self.hidden_size]
		f_t = gates[:, self.hidden_size:2 * self.hidden_size]
		o_t = gates[:, -self.hidden_size:]

		c_t = torch.mul(c, f_t) + torch.mul(i_t, g_t)

		h_t = torch.mul(o_t, c_t.tanh())

		new_h = h_t.view(1, h_t.size(0), -1)
		new_c = c_t.view(1, c_t.size(0), -1)
		
		new_y = (new_h, new_c)

		if masked_update:
			# IMPORTANT: assumes that x contains both data and mask
			# update only the hidden states for hidden state only if at least one feature is present for the current time point
			n_data_dims = x.size(-1)//2
			mask = x[:, :, n_data_dims:]
			#pdb.set_trace()
			utils.check_mask(x[:, :, :n_data_dims], mask)
			
			mask = (torch.sum(mask, -1, keepdim = True) > 0).float()

			assert(not torch.isnan(mask).any())

			new_h = mask * new_h + (1-mask) * y_mean[0]
			new_c = mask * new_c + (1-mask) * y_mean[1]

			new_y = (new_h, new_c)

			if torch.isnan(new_h).any():
				print("new_y is nan!")
				print(mask)
				print(new_h)
				print(y_mean)
				exit()
			if torch.isnan(new_c).any():
				print("new_y is nan!")
				print(mask)
				print(new_c)
				print(y_mean)
				exit()
		
		# just return a dummy tensor, since it is not used later for this project.
		new_y_std = y_std

		#return h_t, (h_t, c_t)
		return new_y, new_y_std
