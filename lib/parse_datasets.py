###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
# Code adapted from Yulia Rubanova, Latent ordinary differential equations for irregularly-sampled time series
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import FastTensorDataLoader
from lib.diffeq_solver import DiffeqSolver
from torch.distributions import uniform

from torch.utils.data import DataLoader

from crop_classification import Crops, variable_time_collate_fn_crop
from swisscrop_classification import SwissCrops

from sklearn import model_selection
import random

#Nando's packages
import pdb

#####################################################################################################
def parse_datasets(args, device):

	def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
		batch = torch.stack(batch)
		data_dict = {
			"data": batch, 
			"time_steps": time_steps}

		data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
		return data_dict

	dataset_name = args.dataset

	n_total_tp = args.timepoints + args.extrap
	max_t_extrap = args.max_t / args.timepoints * n_total_tp

	##################################################################
	# Crop Classification TUM
		
	if dataset_name == "crop":

		#Implemented tensorformat
		list_form = False
		num_workers = 1

		# set to False in order to speed up the process
		automatic_batching = False

		#turn this boolean to true in order to get access to the larger "evaluation" dataset used for validation
		eval_as_test = True

		root = r'data/Crops'
		scratch_root1 = r'/scratch/Nando/ODEcrop/Crops'
		scratch_root2 = r'/cluster/scratch/metzgern/ODEcrop/Crops'
		if os.path.exists(scratch_root1):
			root = scratch_root1
		elif os.path.exists(scratch_root2):
			root = scratch_root2
		print("dataroot: " + root)#cloud_thresh

		if args.hp_search:
			
			train_dataset_obj = Crops(root, mode="train_from_train", args=args, noskip=False,
									download=True, device = device, list_form = list_form)
			test_dataset_obj = Crops(root, mode="validation_from_train", args=args, noskip=False,
									download=True, device = device, list_form = list_form)
			eval_dataset_obj = Crops(root, mode="validation_from_train", args=args, noskip=False,
									download=True, device = device, list_form = list_form)

			validation_batch_size = args.batch_size
		else:
			train_dataset_obj = Crops(root, mode="train", args=args, noskip=False,
									download=True, device = device, list_form = list_form)
			test_dataset_obj = Crops(root, mode="test", args=args, noskip=False,
									download=True, device = device, list_form = list_form)
			eval_dataset_obj = Crops(root, mode="eval", args=args, noskip=False, 
									download=True, device = device,  list_form = list_form)


			validation_batch_size = 10000
		
		n_samples = min(args.n, len(train_dataset_obj))
		n_eval_samples = min( float("inf"), len(eval_dataset_obj)) #TODO set it back to inf
		n_test_samples = min( float("inf"), len(test_dataset_obj))
		
		#should I read the data into memory? takes about 4 minutes for the whole dataset!
		#not recommended for debugging with large datasets, so better set it to false
		#read_to_mem = list_form #defualt True 
		if list_form:
			train_data = train_dataset_obj[:n_samples]
			test_data = test_dataset_obj[:n_test_samples]
			eval_data = eval_dataset_obj[:n_eval_samples]
		else:
			train_data = train_dataset_obj
			test_data = test_dataset_obj
			eval_data = eval_dataset_obj
			
		if list_form:
			vals, tt, mask, labels = train_dataset_obj[0]
		else:
			a_train_dict = train_dataset_obj[0]
			vals = a_train_dict["observed_data"]
			tt = a_train_dict["observed_tp"]
			mask = a_train_dict["observed_mask"]
			labels = a_train_dict["labels"]

		batch_size = min(args.batch_size, args.n)

		#evaluation batch sizes. #Must be tuned to increase efficency of evaluation
		#validation_batch_size = 10000 # size 30000 is 10s per batch
		test_batch_size = min(n_test_samples, validation_batch_size)
		eval_batch_size = min(n_eval_samples, validation_batch_size)
		
		#create the dataloader
		if list_form:
			train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False, 
				collate_fn= lambda batch: variable_time_collate_fn_crop(batch, args, device, data_type="train", list_form=list_form))
			test_dataloader = DataLoader(test_data, batch_size = test_batch_size, shuffle=False, 
				collate_fn= lambda batch: variable_time_collate_fn_crop(batch, args, device, data_type="test", list_form=list_form))
			eval_dataloader = DataLoader(eval_data, batch_size = eval_batch_size, shuffle=False, 
				collate_fn= lambda batch: variable_time_collate_fn_crop(batch, args, device, data_type="eval", list_form=list_form))
		
		else: #else tensor format is used
			if automatic_batching:
				train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle=False, num_workers=num_workers)
				test_dataloader = DataLoader(test_data, batch_size = test_batch_size, shuffle=False, num_workers=num_workers)
				eval_dataloader = DataLoader(eval_data, batch_size = eval_batch_size, shuffle=False, num_workers=num_workers)

			else: #else manual batching is used
				#recommendation: set shuffle to False, the underlying hd5y structure is than more efficient
				# because it can make use of the countagious blocks of data.
				perc = 1.0
				early_prediction = int(26*perc)
				train_dataloader = FastTensorDataLoader(train_data, batch_size=batch_size, shuffle=False, early_prediction=early_prediction, subsamp=args.trainsub, use_pos_encod2=args.use_pos_encod2)
				test_dataloader = FastTensorDataLoader(test_data, batch_size=test_batch_size, shuffle=False, use_pos_encod2=args.use_pos_encod2)
				eval_dataloader = FastTensorDataLoader(eval_data, batch_size=eval_batch_size, shuffle=False, early_prediction=early_prediction, subsamp=args.testsub, use_pos_encod2=args.use_pos_encod2)
			
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader), #changed to validate on the evalutation set #attention, might be another naming convention...
					"eval_dataloader": utils.inf_generator(eval_dataloader), #attention, might be another naming convention...
					"input_dim": vals.size(-1),
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"n_eval_batches": len(eval_dataloader),
					"classif_per_tp": False, # We want to classify the whole sequence!!. Standard: True, #optional
					"n_labels": labels.size(-1)}
		
		print("")
		print("Trainingdataset:")
		print(data_objects["dataset_obj"])

		if eval_as_test:
			data_objects["test_dataloader"] = utils.inf_generator(eval_dataloader)
			data_objects["n_test_batches"] = len(eval_dataloader)
			print("Using Evaluationdataset:")
			print(eval_dataset_obj)

		else:
			print("Using Testdataset:")
			print(test_dataset_obj)

		return data_objects
	
	##################################################################
	###########	 SWISS Crop Classification	 ####################
	
	if dataset_name == "swisscrop":
		
		# Search for a dataroot
		
		root = r'data/SwissCrops'
		scratch_root1 = r'/cluster/scratch/metzgern/ODEcrop/Swisscrop'
		scratch_root2 = r'/scratch/Nando/ODEcrop/Swisscrop'
		if os.path.exists(scratch_root1):
			root = scratch_root1
			print(scratch_root1)
		elif os.path.exists(scratch_root2):
			# Leonhard cluster case
			root = scratch_root2
			print(scratch_root2)
		print("dataroot: " + root)

		# Search for a dataroot
		root = r'data/SwissCrops'
		scratch_root1 = r'/cluster/scratch/metzgern/ODEcrop/Swisscrop'
		scratch_root2 = r'/scratch/Nando/ODEcrop/Swisscrop'
		if os.path.exists(scratch_root1):
			root = scratch_root1
			print(scratch_root1)
		elif os.path.exists(scratch_root2):
			# Leonhard cluster case
			root = scratch_root2
			print(scratch_root2)
		print("dataroot: " + root)

		if args.hp_search:
			train_dataset_obj = SwissCrops(root, mode="train_from_train", device=device,  noskip=args.noskip,
											step=args.step, trunc=args.trunc, nsamples=args.n,
											datatype=args.swissdatatype, singlepix=args.singlepix)
			test_dataset_obj = SwissCrops(root, mode="validation_from_train", device=device,  noskip=args.noskip,
											step=args.step, trunc=args.trunc, nsamples=args.validn,
											datatype=args.swissdatatype, singlepix=args.singlepix) 
		else:
				
			train_dataset_obj = SwissCrops(root, mode="train", device=device,  noskip=args.noskip,
											step=args.step, trunc=args.trunc, nsamples=args.n,
											datatype=args.swissdatatype, singlepix=args.singlepix)
			test_dataset_obj = SwissCrops(root, mode="test", device=device,  noskip=args.noskip,
											step=args.step, trunc=args.trunc, nsamples=args.validn,
											datatype=args.swissdatatype, singlepix=args.singlepix) 
		
		n_samples = min(args.n, len(train_dataset_obj))
		n_test_samples = min( float("inf"), len(test_dataset_obj))
		
		#evaluation batch sizes. #Must be tuned to increase efficency of evaluation
		validation_batch_size = 5000 # size 30000 is 10s per batch, also depending on server connection
		train_batch_size = min(args.batch_size, args.n)
		test_batch_size = min(n_test_samples, validation_batch_size)

		a_train_dict = train_dataset_obj[0]
		vals = a_train_dict["observed_data"]
		tt = a_train_dict["observed_tp"]
		mask = a_train_dict["observed_mask"]
		labels = a_train_dict["labels"]
		
		train_dataloader = FastTensorDataLoader(train_dataset_obj, batch_size=train_batch_size, subsamp=args.trainsub, use_pos_encod2=args.use_pos_encod2)
		test_dataloader = FastTensorDataLoader(test_dataset_obj, batch_size=test_batch_size, subsamp=args.testsub, use_pos_encod2=args.use_pos_encod2)

		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader), 
					"input_dim": vals.size(-1),
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, # We want to classify the whole sequence!!. Standard: True, #optional
					"n_labels": train_dataloader.nclasses+1} #plus one, because there is one class that summerizes all the other classes--> "other" is "0"
		
		return data_objects

	########### 1d datasets ###########

	# Sampling args.timepoints time points in the interval [0, args.max_t]
	# Sample points for both training sequence and explapolation (test)
	distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
	time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
	time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
	time_steps_extrap = torch.sort(time_steps_extrap)[0]

	dataset_obj = None
	##################################################################
	# Sample a periodic function
	if dataset_name == "periodic":
		dataset_obj = Periodic_1d(
			init_freq = None, init_amplitude = 1.,
			final_amplitude = 1., final_freq = None, 
			z0 = 1.)

	##################################################################

	if dataset_obj is None:
		raise Exception("Unknown dataset: {}".format(dataset_name))

	dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n, 
		noise_weight = args.noise_weight)

	# Process small datasets
	dataset = dataset.to(device)
	time_steps_extrap = time_steps_extrap.to(device)

	train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

	n_samples = len(dataset)
	input_dim = dataset.size(-1)

	batch_size = min(args.batch_size, args.n)
	train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
	test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
		collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))
	
	data_objects = {#"dataset_obj": dataset_obj, 
				"train_dataloader": utils.inf_generator(train_dataloader), 
				"test_dataloader": utils.inf_generator(test_dataloader),
				"input_dim": input_dim,
				"n_train_batches": len(train_dataloader),
				"n_test_batches": len(test_dataloader)}

	return data_objects


