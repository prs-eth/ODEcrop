###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.utils import FastTensorDataLoader
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader

from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity
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
	# MuJoCo dataset
	if dataset_name == "hopper":
		dataset_obj = HopperPhysics(root='data', download=True, generate=False, device = device)
		dataset = dataset_obj.get_dataset()[:args.n]
		dataset = dataset.to(device)

		n_tp_data = dataset[:].shape[1]

		# Time steps that are used later on for exrapolation
		time_steps = torch.arange(start=0, end = n_tp_data, step=1).float().to(device)
		time_steps = time_steps / len(time_steps)

		dataset = dataset.to(device)
		time_steps = time_steps.to(device)

		if not args.extrap:
			# Creating dataset for interpolation
			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			n_traj = len(dataset)
			n_tp_data = dataset.shape[1]
			n_reduced_tp = args.timepoints

			# sample time points from different parts of the timeline, 
			# so that the model learns from different parts of hopper trajectory
			start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
			end_ind = start_ind + n_reduced_tp
			sliced = []
			for i in range(n_traj):
				  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
			dataset = torch.stack(sliced).to(device)
			time_steps = time_steps[:n_reduced_tp]

		# Split into train and test by the time sequences
		train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

		n_samples = len(dataset)
		input_dim = dataset.size(-1)

		batch_size = min(args.batch_size, args.n)
		train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "train"))
		test_dataloader = DataLoader(test_y, batch_size = n_samples, shuffle=False,
			collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "test"))
		
		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader)}
		return data_objects

	##################################################################
	# Physionet dataset

	if dataset_name == "physionet":
		train_dataset_obj = PhysioNet('data/physionet', train=True, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps
		test_dataset_obj = PhysioNet('data/physionet', train=False, 
										quantization = args.quantization,
										download=True, n_samples = min(10000, args.n), 
										device = device)

		# Combine and shuffle samples from physionet Train and physionet Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]
		
		
		if not args.classif:
			# Concatenate samples from original Train and Test sets
			# Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
			total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8, 
			random_state = 42, shuffle = True)

		record_id, tt, vals, mask, labels = train_data[0]

		n_samples = len(total_dataset)
		input_dim = vals.size(-1)

		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		data_min, data_max = get_data_min_max(total_dataset)

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))

		attr_names = train_dataset_obj.params
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": 1} #optional
		return data_objects

	##################################################################
	# Human activity dataset

	if dataset_name == "activity":
		n_samples =  min(10000, args.n)
		dataset_obj = PersonActivity('data/PersonActivity', 
							download=True, n_samples =  n_samples, device = device)
		print(dataset_obj)
		# Use custom collate_fn to combine samples with arbitrary time observations.
		# Returns the dataset along with mask and time steps

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8, 
			random_state = 42, shuffle = True)

		train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
		test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

		record_id, tt, vals, mask, labels = train_data[0]
		input_dim = vals.size(-1)

		batch_size = min(min(len(dataset_obj), args.batch_size), args.n)
		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
		test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))

		data_objects = {"dataset_obj": dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": True, #optional
					"n_labels": labels.size(-1)}

		return data_objects
		
	##################################################################
	# Crop Classification
		
	if dataset_name == "crop":

		#Implemented tensorformat
		list_form = False
		num_workers = 1

		# set to False in order to speed up the process
		automatic_batching = False

		#turn this boolean to true in order to get access to the larger "evaluation" dataset used for validation
		eval_as_test = True

		root = 'data/Crops'
		scratch_root = '/scratch/Nando/ODEcrop/Crops'
		if os.path.exists(scratch_root):
			root = scratch_root

		train_dataset_obj = Crops(root, mode="train", args=args,
									download=True, device = device, list_form = list_form)
		test_dataset_obj = Crops(root, mode="test", args=args, 
									download=True, device = device, list_form = list_form)
		
		eval_dataset_obj = Crops(root, mode="eval", args=args, 
									download=True, device = device,  list_form = list_form)
		
		
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
		validation_batch_size = 10000 # size 30000 is 10s per batch
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
				train_dataloader = FastTensorDataLoader(train_data, batch_size=batch_size, shuffle=False)
				test_dataloader = FastTensorDataLoader(test_data, batch_size=test_batch_size, shuffle=False)
				eval_dataloader = FastTensorDataLoader(eval_data, batch_size=eval_batch_size, shuffle=False)
			
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
	###########     SWISS Crop Classification     ####################
	
	if dataset_name == "swisscrop":

		root = 'data/SwissCrops'
		scratch_root = '/scratch/Nando/ODEcrop/SwissCrops'
		if os.path.exists(scratch_root):
			root = scratch_root

		train_dataset_obj = SwissCrops(root, mode="train", device=device,
										step=args.step, trunc=args.trunc, nsamples=args.n,
										datatype=args.swissdatatype)
		test_dataset_obj = SwissCrops('data/SwissCrops', mode="test", device=device,
										step=args.step, trunc=args.trunc, nsamples=args.validn,
										datatype=args.swissdatatype) 
		
		n_samples = min(args.n, len(train_dataset_obj))
		n_test_samples = min( float("inf"), len(test_dataset_obj))
		
		#evaluation batch sizes. #Must be tuned to increase efficency of evaluation
		validation_batch_size = 10000 # size 30000 is 10s per batch, also depending on server connection
		train_batch_size = min(args.batch_size, args.n)
		test_batch_size = min(n_test_samples, validation_batch_size)

		a_train_dict = train_dataset_obj[0]
		vals = a_train_dict["observed_data"]
		tt = a_train_dict["observed_tp"]
		mask = a_train_dict["observed_mask"]
		labels = a_train_dict["labels"]
		
		train_dataloader = FastTensorDataLoader(train_dataset_obj, batch_size=train_batch_size)
		test_dataloader = FastTensorDataLoader(test_dataset_obj, batch_size=test_batch_size)

		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader), 
					"input_dim": vals.size(-1),
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"classif_per_tp": False, # We want to classify the whole sequence!!. Standard: True, #optional
					"n_labels": labels.size(-1)}

		"""
		print("")
		print("Trainingdataset:")
		print(data_objects["dataset_obj"])

		print("Using Testdataset:")
		print(test_dataset_obj)
		"""
		
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


