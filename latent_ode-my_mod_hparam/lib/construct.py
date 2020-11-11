"""
author: Nando Metzger
metzgern@ethz.ch
"""

import lib.utils as utils 
from lib.ode_rnn import *
from lib.rnn_baselines import *

from lib.ode_func import ODEFunc
from lib.gru_ode import FullGRUODECell_Autonomous
from lib.diffeq_solver import DiffeqSolver

import pdb

def get_ODE_RNN_model(args, device, input_dim, n_labels, classif_per_tp):

	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	n_ode_gru_dims = int(args.latents)

	
	if args.stacking>=1:
		model = ML_ODE_RNN(input_dim, n_ode_gru_dims, device = device,
			 n_gru_units = int(args.gru_units),
			concat_mask = True, obsrv_std = obsrv_std,
			use_binary_classif = args.classif,
			classif_per_tp = classif_per_tp,
			n_labels = n_labels,
			train_classif_w_reconstr = (args.dataset == "physionet"),
			RNNcell = args.rnn_cell,
			stacking = args.stacking, stack_order = args.stack_order,
			ODE_sharing = args.ODE_sharing, RNN_sharing = args.RNN_sharing,
			include_topper = args.topper, linear_topper = args.linear_topper,
			use_BN = args.batchnorm,
			resnet = args.resnet,
			ode_type=args.ode_type, ode_units = args.units, rec_layers = args.rec_layers, ode_method = args.ode_method,
			nornnimputation=args.nornnimputation
		).to(device)
	else:
		raise Exception("Number of stacked layers must be greater or equal to 1.")

	return model


def get_classic_RNN_model(args, device, input_dim, n_labels, classif_per_tp):
	
	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)

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
		train_classif_w_reconstr = (args.dataset == "physionet"),
		).to(device)

		
	return model