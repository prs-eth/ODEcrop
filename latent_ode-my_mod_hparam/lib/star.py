"""
author: Nando Metzger
metzgern@ethz.ch

Code based on:
Gating Revisited: Deep Multi-layer RNNs That Can Be Trained
https://arxiv.org/abs/1911.11033
"""

import lib.utils as utils

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
#from torch.nn import Parameter
#from torch import Tensor

import numpy as np
import pdb
	

class STAR_unit(nn.Module):

	def __init__(self, hidden_size, input_size,
		n_units=0, bias=True):
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
			
		
#		init.kaiming_normal_(self.x_K.weight) 
#		init.kaiming_normal_(self.x_z.weight)
#		init.kaiming_normal_(self.h_K.weight)

		#bias_f= np.log(np.random.uniform(1,784,hidden_size))
		#bias_f = torch.Tensor(bias_f)  
		#self.bias_K = Variable(bias_f.cuda(), requires_grad=True)

		
	def forward(self, hidden, y_std, x):
				
		gate_x_K = self.x_K(x) 			# return size torch.Size([1, batch_size, latent_dim])
		gate_x_z = self.x_z(x) 			# return size torch.Size([1, batch_size, latent_dim])
		gate_h_K = self.h_K(hidden)		# return size torch.Size([1, batch_size, latent_dim])
		
		gate_x_K = gate_x_K.squeeze()
		gate_x_z = gate_x_z.squeeze()
		gate_h_K = gate_h_K.squeeze()
		
		K_gain = torch.sigmoid(gate_x_K + gate_h_K)
		z = torch.tanh(gate_x_z)
		
		h_new = hidden + K_gain * ( z - hidden) 
		h_new = torch.tanh(h_new)
		
		return h_new, y_std