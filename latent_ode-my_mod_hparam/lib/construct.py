"""
author: Nando Metzger
metzgern@ethz.ch
"""

import lib.utils as utils 
from lib.ode_rnn import *

from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver




def  get_ODE_RNN_model(args, device, input_dim, n_labels, classif_per_tp):

	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	n_ode_gru_dims = int(args.latents)
	method = args.ode_method

	if args.poisson:
		print("Poisson process likelihood not implemented for ODE-RNN: ignoring --poisson")

	if args.extrap:
		raise Exception("Extrapolation for ODE-RNN not implemented")

	ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
		n_layers = int(args.rec_layers), n_units = int(args.units), nonlinear = nn.Tanh)

	rec_ode_func = ODEFunc(
		input_dim = input_dim, 
		latent_dim = n_ode_gru_dims,
		ode_func_net = ode_func_net,
		device = device).to(device)

	z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, method, int(args.latents), 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	model = ODE_RNN(input_dim, n_ode_gru_dims, device = device, 
		z0_diffeq_solver = z0_diffeq_solver, n_gru_units = int(args.gru_units),
		concat_mask = True, obsrv_std = obsrv_std,
		use_binary_classif = args.classif,
		classif_per_tp = classif_per_tp,
		n_labels = n_labels,
		train_classif_w_reconstr = (args.dataset == "physionet")
		).to(device)

	return model