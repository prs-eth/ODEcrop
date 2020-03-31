"""
author: Nando Metzger
metzgern@ethz.ch
"""

import os
from random import SystemRandom

import lib.utils as utils
from lib.utils import compute_loss_all_batches
from lib.utils import Bunch
from lib.construct import get_ODE_RNN_model
from lib.ode_rnn import *

from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from tqdm import tqdm
import pdb


def construct_and_train_model(config):
	# Create ODE-GRU model

	args = config["spec_config"][0]["args"]

	# namespace to dict
	argsdict = vars(args)

	for key in config.keys():
		if not key=='spec_config':
			argsdict[key] = config[key]

	# namespace to dict
	args = Bunch(argsdict)

	# onrolle the other parameters:
	data_obj = config["spec_config"][0]["data_obj"]
	file_name = config["spec_config"][0]["file_name"]
	experimentID = config["spec_config"][0]["experimentID"]
	input_command = config["spec_config"][0]["input_command"]
	device = config["spec_config"][0]["device"]

	##############################################################################

	# set seed
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*10000000)


	##############################################################################

	input_dim = data_obj["input_dim"]
	pdb.set_trace()
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


	##############################################################################
	# Create Model

	model = get_ODE_RNN_model(args, device, input_dim, n_labels, classif_per_tp)


	##################################################################
	
	if args.tensorboard:
		comment = '_'
		if args.classic_rnn:
			nntype = 'rnn'

		elif args.ode_rnn:
			nntype = 'ode'

		comment = nntype + "_ns:" + str(args.n) + "_ba:" + str(args.batch_size) + "_ode-units:" + str(args.units) + "_gru-uts:" + str(args.gru_units) + "_lats:"+ str(args.latents) + "_rec-lay:" + str(args.rec_layers) + "_solver:" + str(args.ode_method) + "_seed:" +str(args.random_seed) + "_optim:" +str(args.optimizer)

		validationtensorboard_dir = "runs/expID" + str(experimentID) + "_VALID" + comment
		validationwriter = SummaryWriter(validationtensorboard_dir, comment=comment)
		
		tensorboard_dir = "runs/expID" + str(experimentID) + "_TRAIN" + comment
		trainwriter = SummaryWriter(tensorboard_dir, comment=comment)
	
	print(tensorboard_dir)
	##################################################################

	##################################################################
	# Training

	num_batches = data_obj["n_train_batches"]

	train_res, test_res = train_it(
		model,
		data_obj,
		args,
		file_name,
		experimentID,
		trainwriter,
		validationwriter,
		input_command,
		device
	)

	train_res_2, test_res_2 = train_it(
		model,
		data_obj,
		args,
		file_name,
		experimentID,
		trainwriter,
		validationwriter,
		input_command,
		device
	)


	# because it is fmin, we have to bring back some kind of loss, therefore 1-...
	pdb.set_trace()


	return 1-test_res["accuracy"]

def train_it(
		model,
		data_obj,
		args,
		file_name,
		experimentID,
		trainwriter,
		validationwriter,
		input_command,
		device):

	"""
	parameters:
		model,
		data_obj,
		args,
		file_name,
		experimentID,
		trainwriter,
		validationwriter,
		device
	"""

	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
	top_ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '_topscore.ckpt')
	best_test_acc = 0

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	#logger.info(input_command)
	
	optimizer = get_optimizer(args, model.parameters())

	num_batches = data_obj["n_train_batches"]

	
	for itr in (range(1, num_batches * (args.niters) + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.01
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
		
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		train_res = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
		train_res["loss"].backward()
		optimizer.step()

		n_iters_to_viz = 0.333
		vizualization_intervall =(round(n_iters_to_viz * num_batches - 0.499999) if round(n_iters_to_viz * num_batches - 0.499999)>0 else 1)
		if (itr!=1) and (itr % round(n_iters_to_viz * num_batches - 0.499999)== 0) :
			
			with torch.no_grad():

				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef)

				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])

				#logger.info("Experiment " + str(experimentID))
				#logger.info(message)
				#logger.info("KL coef: {}".format(kl_coef))
				#logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				#logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				# write training numbers
				if "accuracy" in train_res:
					#logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))
					trainwriter.add_scalar('Classification_accuracy', train_res["accuracy"], itr*args.batch_size)
				
				if "loss" in train_res:
					trainwriter.add_scalar('loss', train_res["loss"].detach(), itr*args.batch_size)
				
				if "ce_loss" in train_res:
					trainwriter.add_scalar('CE_loss', train_res["ce_loss"].detach(), itr*args.batch_size)
				
				if "mse" in train_res:
					trainwriter.add_scalar('MSE', train_res["mse"], itr*args.batch_size)
				
				if "pois_likelihood" in train_res:
					trainwriter.add_scalar('Poisson_likelihood', train_res["pois_likelihood"], itr*args.batch_size)
				
				#write test numbers
				if "auc" in test_res:
					#logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))
					validationwriter.add_scalar('Classification_AUC', test_res["auc"], itr*args.batch_size)
					
				if "mse" in test_res:
					#logger.info("Test MSE: {:.4f}".format(test_res["mse"]))
					validationwriter.add_scalar('MSE', test_res["mse"], itr*args.batch_size)
					
				if "accuracy" in test_res:
					#logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))
					validationwriter.add_scalar('Classification_accuracy', test_res["accuracy"], itr*args.batch_size)

				if "pois_likelihood" in test_res:
					#logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))
					validationwriter.add_scalar('Poisson_likelihood', test_res["pois_likelihood"], itr*args.batch_size)
				
				if "loss" in train_res:
					validationwriter.add_scalar('loss', test_res["loss"].detach(), itr*args.batch_size)
				
				if "ce_loss" in test_res:
					#logger.info("CE loss: {}".format(test_res["ce_loss"]))
					validationwriter.add_scalar('CE_loss', test_res["ce_loss"], itr*args.batch_size)
	
				#logger.info("-----------------------------------------------------------------------------------")

				torch.save({
					'args': args,
					'state_dict': model.state_dict(),
				}, ckpt_path)

				if test_res["accuracy"] > best_test_acc:
					best_test_acc = test_res["accuracy"]
					torch.save({
						'args': args,
						'state_dict': model.state_dict(),
					}, top_ckpt_path)
	
	return train_res, test_res