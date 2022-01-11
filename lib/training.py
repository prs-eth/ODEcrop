###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
###########################

import os
from random import SystemRandom

import lib.utils as utils
from lib.utils import compute_loss_all_batches
from lib.utils import Bunch, get_optimizer, plot_confusion_matrix
from lib.construct import get_ODE_RNN_model#, get_classic_RNN_model
from lib.ode_rnn import *
from lib.parse_datasets import parse_datasets

from lib.latent_vis import get_pca_fig

from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from sklearn.metrics import confusion_matrix as sklearn_cm
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score

from tqdm import tqdm
import pdb
import numpy as np
from hyperopt import STATUS_OK
import wandb

import pickle



def construct_and_train_model(config):
	# Create ODE-GRU model

	args = config["spec_config"][0]["args"]

	# namespace to dict
	argsdict = vars(args)

	print("")
	print("Testing hyperparameters:")
	for key in config.keys():
		if not key=='spec_config':
			argsdict[key] = config[key]
			print(key + " : " + str(argsdict[key]))
	print("")

	# namespace to dict
	args = Bunch(argsdict)

	# onrolle the other parameters:
	#Data_obj = config["spec_config"][0]["Data_obj"]
	file_name = config["spec_config"][0]["file_name"]
	experimentID = config["spec_config"][0]["experimentID"]
	input_command = config["spec_config"][0]["input_command"]
	Devices = config["spec_config"][0]["Devices"]
	num_gpus = config["spec_config"][0]["num_gpus"]
	num_seeds = config["spec_config"][0]["num_seeds"]

	num_gpus = max(num_gpus,1)
	if isinstance(args.batch_size, tuple):
		args.batch_size = int(args.batch_size[0])
	args.gru_units = int(args.gru_units)

	
	##############################################################################
	## set seed

	randID = int(SystemRandom().random()*10000)*1000
	ExperimentID = []
	for i in range(num_seeds):
		ExperimentID.append(randID + i)

	print("ExperimentID", ExperimentID)
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	##############################################################################
	## Weights & Biases

	wandb.init(project="odecropclassification",config=args, sync_tensorboard=True, entity="cropteam", group=args.dataset)

	##############################################################################
	# Dataset
	Data_obj = []
	for i, device in enumerate(Devices):
		Data_obj.append(parse_datasets(args, device))

	data_obj = Data_obj[0]

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

	##############################################################################
	# Create Model

	Model = []
	if args.ode_rnn:
		for i in range(num_seeds):
			Model.append(get_ODE_RNN_model(args, Devices[0], input_dim, n_labels, classif_per_tp))
	else:
		raise Exception("only the model with flag '--ode-rnn' is supported. Add this to the command.")

	#if args.classic_rnn:
	#	for i in range(num_seeds):
	#		Model.append(get_classic_RNN_model(args, Devices[0], input_dim, n_labels, classif_per_tp))

	# "Magic" wandb model watcher
	wandb.watch(Model[0], "all")

	##################################################################
	
	# Tensorboard not used anymore.
	if args.tensorboard:
		Validationwriter = []
		#Trainwriter = []
		for i in range(num_seeds):
			comment = '_'
			if args.classic_rnn:
				nntype = 'rnn'

			elif args.ode_rnn:
				nntype = 'ode'
			else:
				raise Exception("please select a model")
			
			RNNws_str = ""
			ODEws_str = ""
			bn_str = ""
			rs_str = ""
			if args.RNN_sharing:
				RNNws_str = "_RNNws"
			if args.ODE_sharing:
				ODEws_str = "_ODEws"
			if args.batchnorm:
				bn_str = "_BN"
			if args.resnet:
				rs_str = "_rs"

			comment = nntype + "_ns:" + str(args.n) + "_ba:" + str(args.batch_size) + "_ode-units:" + str(args.units) + "_gru-uts:" + str(args.gru_units) + "_lats:"+ str(args.latents) + "_rec-lay:" + str(args.rec_layers) + "_solver:" + str(args.ode_method) + "_seed:" +str(args.random_seed) + "_optim:" + str(args.optimizer) + "_stackin:" + str(args.stacking) + str(args.stack_order)+ ODEws_str + RNNws_str + bn_str + rs_str

			validationtensorboard_dir = "runs/expID" + str(ExperimentID[i]) + "_VALID" + comment
			Validationwriter.append( SummaryWriter(validationtensorboard_dir, comment=comment) )
			
			tensorboard_dir = "runs/expID" + str(ExperimentID[i]) + "_TRAIN" + comment
			#Trainwriter.append( SummaryWriter(tensorboard_dir, comment=comment) )
		
			print(tensorboard_dir)
			
	##################################################################
	# Training
	
	Train_res = [None]*num_seeds
	Test_res = [None]*num_seeds
	Best_test_acc = [None]*num_seeds
	Best_test_acc_step = [None]*num_seeds

	for i in range(num_seeds):
		Train_res[i], Test_res[i], Best_test_acc[i], Best_test_acc_step[i] = train_it(
			[Model[i]],
			Data_obj,
			args,
			file_name,
			[ExperimentID[i]],
			#[Trainwriter[i]],
			[Validationwriter[i]],
			input_command,
			[Devices[0]]
		)
	
	Test_acc = []
	Train_acc = []
	for i in range(num_seeds):
		#pdb.set_trace()
		Test_acc.append(Test_res[i][0]["accuracy"])
		Train_acc.append(Train_res[i][0]["accuracy"])


	mean_test_acc = np.mean(Test_acc)
	var_test_acc = sum((abs(Test_acc - mean_test_acc)**2)/(num_seeds-1))

	mean_best_test_acc = np.mean(Best_test_acc)
	var_best_test_acc = sum((abs(Best_test_acc - mean_best_test_acc)**2)/(num_seeds-1))
	best_of_best_test_acc = np.max(Best_test_acc)
	best_of_best_test_acc_step = Best_test_acc_step[np.argmax(Best_test_acc)]

	mean_train_acc = np.mean(Train_acc)
	var_train_acc = sum((abs(Train_acc - mean_train_acc)**2)/(num_seeds-1))
	
	# because it is fmin, we have to bring back some kind of loss, therefore 1-...
	return_dict = {
		'loss': 1-mean_best_test_acc,
		'loss_variance': var_best_test_acc,
		#'true_loss': 1-mean_test_acc,
		#'true_loss_variance':var_test_acc,
		'status': STATUS_OK,
		'num_seeds': num_seeds,
		'best_acc': best_of_best_test_acc,
		'best_peak_step': best_of_best_test_acc_step,
		'best_steps': Best_test_acc_step
	}

	print(return_dict)

	return return_dict

def train_it(
		Model,
		Data_obj,
		args,
		file_name,
		ExperimentID,
		#Trainwriter,
		Validationwriter,
		input_command,
		Devices):

	"""
	parameters:
		Model, #List of Models
		Data_obj, #List of Data_objects which live on different devices
		args,
		file_name,
		ExperimentID, #List of IDs
		trainwriter, #List of TFwriters
		validationwriter, #List of TFwriters
		input_command,
		Devices #List of devices
	"""

	Ckpt_path = []
	Top_ckpt_path = []
	Best_test_acc = []
	Best_test_acc_step = []
	Logger = []
	Optimizer = []
	otherOptimizer = []
	ODEOptimizer = []

	for i, device in enumerate(Devices):

		Ckpt_path.append( os.path.join(args.save, "experiment_" + str(ExperimentID[i]) + '.ckpt') )
		Top_ckpt_path.append( os.path.join(args.save, "experiment_" + str(ExperimentID[i]) + '_topscore.ckpt') )
		Best_test_acc.append(0)
		Best_test_acc_step.append(0)

		log_path = "logs/" + file_name + "_" + str(ExperimentID[i]) + ".log"
		if not os.path.exists("logs/"):
			utils.makedirs("logs/")
		Logger.append( utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__)) )
		Logger[i].info(input_command)
		
		Optimizer.append( get_optimizer(args.optimizer, args.lr, Model[i].parameters() ) )


	num_batches = Data_obj[0]["n_train_batches"]
	labels = Data_obj[0]["dataset_obj"].label_list

	#create empty lists for results and similar
	num_gpus = len(Devices)
	train_res = [None] * num_gpus
	batch_dict = [None] * num_gpus
	test_res = [None] * num_gpus
	label_dict = [None]* num_gpus

	# empty result placeholder
	somedict = {}
	test_res = [somedict]
	test_res[0]["accuracy"] = float(0)

	if args.v==1 or args.v==2:
		pbar = tqdm(range(1, num_batches * (args.niters) + 1), position=0, leave=True, ncols=160)
	else:
		pbar = range(1, num_batches * (args.niters) + 1)
	
	for itr in pbar:
	
		for i, device in enumerate(Devices):
			Optimizer[i].zero_grad()
		for i, device in enumerate(Devices):
			# default decay_rate = 0.999, lowest= args.lr/10 	# original
			# decay_rate = 0.9995, lowest = args.lr / 50 		# new
			utils.update_learning_rate(Optimizer[i], decay_rate = args.lrdecay, lowest = args.lr/1000)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.01
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
		
		for i, device in enumerate(Devices):
			batch_dict[i] = utils.get_next_batch(Data_obj[i]["train_dataloader"])
		
		for i, device in enumerate(Devices):
			train_res[i] = Model[i].compute_all_losses(batch_dict[i], n_traj_samples = 3, kl_coef = kl_coef)
		
		for i, device in enumerate(Devices):
			train_res[i]["loss"].backward()
		
		for i, device in enumerate(Devices):
			Optimizer[i].step()

		n_iters_to_viz = 0.333
		if args.dataset=="swisscrop":
			n_iters_to_viz /= 20
		
		if (itr!=0)	and (itr % args.val_freq) ==0 :
			with torch.no_grad():
				
				# Calculate labels and loss on test data
				for i, device in enumerate(Devices):
					test_res[i], label_dict[i] = compute_loss_all_batches(Model[i], 
						Data_obj[i]["test_dataloader"], args,
						n_batches = Data_obj[i]["n_test_batches"],
						experimentID = ExperimentID[i],
						device = Devices[i],
						n_traj_samples = 3, kl_coef = kl_coef)

				for i, device in enumerate(Devices):
					
					#make confusion matrix
					cm, conf_fig = plot_confusion_matrix(label_dict[0]["correct_labels"],label_dict[0]["predict_labels"], Data_obj[0]["dataset_obj"].label_list, tensor_name='dev/cm')
					Validationwriter[i].add_figure("Validation_Confusionmatrix", conf_fig, itr*args.batch_size)
					
					# prepare GT labels and predictions
					y_ref_train = torch.argmax(train_res[0]['label_predictions'], dim=2).squeeze().cpu()
					y_pred_train = torch.argmax(batch_dict[0]['labels'], dim=1).cpu()
					y_ref = label_dict[0]["correct_labels"].cpu()
					y_pred = label_dict[0]["predict_labels"]

					# prepare GT labels and predictions
					y_ref_train = torch.argmax(train_res[0]['label_predictions'], dim=2).squeeze().cpu()
					y_pred_train = torch.argmax(batch_dict[0]['labels'], dim=1).cpu()
					y_ref = label_dict[0]["correct_labels"].cpu()
					y_pred = label_dict[0]["predict_labels"]

					#Make checkpoint
					torch.save({
						'args': args,
						'state_dict': Model[i].state_dict(),
					}, Ckpt_path[i])

					if test_res[i]["accuracy"] > Best_test_acc[i]:
						Best_test_acc[i] = test_res[i]["accuracy"]
						Best_test_acc_step[i] = itr*args.batch_size
						torch.save({
							'args': args,
							'state_dict': Model[i].state_dict(),
							'cm': cm
						}, Top_ckpt_path[i])


						#utils.plot_confusion_matrix2(y_ref, y_pred, Data_obj[0]["dataset_obj"].label_list, ExperimentID[i])
						# Save trajectory here
						#if not test_res[i]["PCA_traj"] is None:
						#	with open( os.path.join('vis', 'traj_dict' + str(ExperimentID[i]) + '.pickle' ), 'wb') as handle:
						#		pickle.dump(test_res[i]["PCA_traj"], handle, protocol=pickle.HIGHEST_PROTOCOL)

					# make PCA visualization
					if "PCA_traj" in test_res[0]:
						#PCA_fig = get_pca_fig(test_res[0]["PCA_traj"]["PCA_trajs1"])
						PCA_fig = None
					else:
						PCA_fig = None

					logdict = {
						'Classification_accuracy/train': train_res[i]["accuracy"],
						'Classification_accuracy/validation': test_res[i]["accuracy"],
						'Classification_accuracy/validation_peak': Best_test_acc[i],
						'Classification_accuracy/validation_peak_step': Best_test_acc_step[i],
						
						'loss/train': train_res[i]["loss"].detach(),
						'loss/validation': test_res[i]["loss"].detach(),

						'Other_metrics/train_cm' : sklearn_cm(y_ref_train, y_pred_train),
						'Other_metrics/train_precision': precision_score(y_ref_train, y_pred_train, average='macro'),
						'Other_metrics/train_recall': recall_score(y_ref_train, y_pred_train, average='macro'),
						'Other_metrics/train_f1': f1_score(y_ref_train, y_pred_train, average='macro'),
						'Other_metrics/train_kappa': cohen_kappa_score(y_ref_train, y_pred_train),

						'Other_metrics/validation_cm' : sklearn_cm(y_ref, y_pred),
						'Other_metrics/validation_precision': precision_score(y_ref, y_pred, average='macro'),
						'Other_metrics/validation_recall': recall_score(y_ref, y_pred, average='macro'),
						'Other_metrics/validation_f1': f1_score(y_ref, y_pred, average='macro'),
						'Other_metrics/validation_kappa': cohen_kappa_score(y_ref, y_pred),
					}

					if "PCA_traj" in test_res[0]:
						pass
						#logdict['Visualization/latent_trajectory'] = wandb.Image( get_pca_fig(test_res[0]["PCA_traj"]) )
					
					wandb.log(logdict, step=itr*args.batch_size)
					
					# wandb.sklearn.plot_confusion_matrix(y_ref, y_pred, labels)
		# Write training loss and accuracy after every batch (Only recommanded for debugging)
		fine_train_writer = False
		if fine_train_writer:
			if "loss" in train_res[i]:
				Validationwriter[i].add_scalar('loss/train', train_res[i]["loss"].detach(), itr*args.batch_size)
			if "accuracy" in train_res[i]:
				Validationwriter[i].add_scalar('Classification_accuracy/train', train_res[i]["accuracy"], itr*args.batch_size)
					

		#update progressbar
		if args.v==2:
			pbar.set_description(
				"Train Ac: {:.3f} %  |  Test Ac: {:.3f} %, Peak Test Ac.: {:.3f} % (at {} batches)  |".format(
					train_res[0]["accuracy"]*100,
					test_res[0]["accuracy"]*100,
					Best_test_acc[i]*100,
					Best_test_acc_step[0]//args.batch_size)
			)

		#empty all training variables
		#train_res = [None] * num_gpus
		batch_dict = [None] * num_gpus
		#test_res = [None] * num_gpus
		label_dict = [None]* num_gpus

	print(Best_test_acc[0], " at step ", Best_test_acc_step[0])
	return train_res, test_res, Best_test_acc[0], Best_test_acc_step[0]