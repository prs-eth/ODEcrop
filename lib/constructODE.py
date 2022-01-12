###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
###########################

import lib.utils as utils

from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver

import torch.nn as nn
import torch

from lib.ODEcells import FullGRUODECell_Autonomous, FullSTARODECell_Autonomous, FullSTARODECell_Autonomous_2



def get_diffeq_solver(ode_latents, ode_units, rec_layers,
		ode_method, ode_type="linear",
		device = torch.device("cpu"),
		gates=None):

	if ode_type=="linear":
		ode_func_net = utils.create_net(ode_latents, ode_latents, 
		n_layers = int(rec_layers), n_units = int(ode_units), nonlinear = nn.Tanh)
	elif ode_type=="gru":
		ode_func_net = FullGRUODECell_Autonomous(ode_latents, bias=True)
	elif ode_type=="star":
		ode_func_net = FullSTARODECell_Autonomous(ode_latents, bias=True)
	elif ode_type=="star2":
		ode_func_net = FullSTARODECell_Autonomous_2(ode_latents, int(ode_units), gates[0], gates[1], gates[2], bias=True)
	else:
		raise Exception("Invalid ODE-type. Choose linear or gru.")

	rec_ode_func = ODEFunc( input_dim = 0, latent_dim = ode_latents,
		ode_func_net = ode_func_net, device = device).to(device)

	z0_diffeq_solver = DiffeqSolver(0, rec_ode_func, ode_method, ode_latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	return z0_diffeq_solver

