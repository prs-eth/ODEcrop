###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import logging
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime
from tqdm import tqdm

import h5py
import pdb
import torch.optim as optim

from textwrap import wrap
import re
import itertools
#import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix
from lib.latent_vis import get_pca_traj

import matplotlib.pyplot as plt 

from lib.latent_vis import get_pca_traj


def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	#logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def dump_pickle(data, filename):
	with open(filename, 'wb') as pkl_file:
		pickle.dump(data, pkl_file)

def load_pickle(filename):
	with open(filename, 'rb') as pkl_file:
		filecontent = pickle.load(pkl_file)
	return filecontent

def make_dataset(dataset_type = "spiral",**kwargs):
	if dataset_type == "spiral":
		data_path = "data/spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	elif dataset_type == "chiralspiral":
		data_path = "data/chiral-spirals.pickle"
		dataset = load_pickle(data_path)["dataset"]
		chiralities = load_pickle(data_path)["chiralities"]
	else:
		raise Exception("Unknown dataset type " + dataset_type)
	return dataset, chiralities


def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res


def init_network_weights(net, std = 0.1, initype="ortho"):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			if initype=="ortho":
				nn.init.orthogonal_(m.weight)
			else:
				nn.init.normal_(m.weight, mean=0, std=std)

			#nn.init.xavier_uniform_(m.weight, gain=1.0)
			#nn.init.kaiming_uniform_(m.weight, gain=1.0)
			#nn.init.xavier_normal_(m.weight, gain=1.0)
			#nn.init.kaiming_normal_(m.weight, gain=1.0)
			#nn.init.sparse_(m.weight, sparsity=0.1, std=0.01)
			
			if not m.bias is None:
				nn.init.constant_(m.bias, val=0)


def flatten(x, dim):
	return x.reshape(x.size()[:dim] + (-1, ))


def subsample_timepoints(data, time_steps, mask, n_tp_to_sample = None):
	# n_tp_to_sample: number of time points to subsample. If not None, sample exactly n_tp_to_sample points
	if n_tp_to_sample is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)


	if n_tp_to_sample > 1:
		# Subsample exact number of points
		assert(n_tp_to_sample <= n_tp_in_batch)
		n_tp_to_sample = int(n_tp_to_sample)

		for i in range(data.size(0)):
			missing_idx = sorted(np.random.choice(np.arange(n_tp_in_batch), n_tp_in_batch - n_tp_to_sample, replace = False))

			data[i, missing_idx] = 0.
			if mask is not None:
				mask[i, missing_idx] = 0.
	
	elif (n_tp_to_sample <= 1) and (n_tp_to_sample > 0):
		# Subsample percentage of points from each time series
		percentage_tp_to_sample = n_tp_to_sample
		for i in range(data.size(0)):
			# take mask for current training sample and sum over all features -- figure out which time points don't have any measurements at all in this batch
			current_mask = mask[i].sum(-1).cpu()
			non_missing_tp = np.where(current_mask > 0)[0]
			n_tp_current = len(non_missing_tp)
			n_to_sample = int(n_tp_current * percentage_tp_to_sample)
			subsampled_idx = sorted(np.random.choice(non_missing_tp, n_to_sample, replace = False))
			tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

			data[i, tp_to_set_to_zero] = 0.
			if mask is not None:
				mask[i, tp_to_set_to_zero] = 0.

	return data, time_steps, mask


def cut_out_timepoints(data, time_steps, mask, n_points_to_cut = None):
	# n_points_to_cut: number of consecutive time points to cut out
	if n_points_to_cut is None:
		return data, time_steps, mask
	n_tp_in_batch = len(time_steps)

	if n_points_to_cut < 1:
		raise Exception("Number of time points to cut out must be > 1")

	assert(n_points_to_cut <= n_tp_in_batch)
	n_points_to_cut = int(n_points_to_cut)

	for i in range(data.size(0)):
		start = np.random.choice(np.arange(5, n_tp_in_batch - n_points_to_cut-5), replace = False)

		data[i, start : (start + n_points_to_cut)] = 0.
		if mask is not None:
			mask[i, start : (start + n_points_to_cut)] = 0.

	return data, time_steps, mask


def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()


def split_train_test(data, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]
	return data_train, data_test

def split_train_test_data_and_time(data, time_steps, train_fraq = 0.8):
	n_samples = data.size(0)
	data_train = data[:int(n_samples * train_fraq)]
	data_test = data[int(n_samples * train_fraq):]

	assert(len(time_steps.size()) == 2)
	train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
	test_time_steps = time_steps[:, int(n_samples * train_fraq):]

	return data_train, data_test, train_time_steps, test_time_steps



def get_next_batch(dataloader):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()

	# remove the time points where there are no observations in this batch
	non_missing_tp = torch.sum(data_dict["observed_data"],(0,2)) != 0.
	batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
	if len(data_dict["observed_tp"].shape)==2:
		batch_dict["observed_tp"] = data_dict["observed_tp"][0,non_missing_tp]
	else:
		batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

	# print("observed data")
	# print(batch_dict["observed_data"].size())

	if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
		batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

	batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
	if len(data_dict["observed_tp"].shape)==2:
		batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][0]
	else:
		batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

	non_missing_tp = torch.sum(data_dict["data_to_predict"],(0,2)) != 0.
	batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
	if len(data_dict["observed_tp"].shape)==2:
		batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][0,non_missing_tp]
	else:
		batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]
	# print("data_to_predict")
	# print(batch_dict["data_to_predict"].size())

	if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
		batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		batch_dict["labels"] = data_dict["labels"]
		
		#Nando's Modification below: why don't cut the labels as well?
		#if args.dataset == "crop":
		batch_dict["labels"] = data_dict["labels"]#[:, non_missing_tp]

	batch_dict["mode"] = data_dict["mode"]
	return batch_dict



def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path)
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr


def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]


def create_net(n_inputs, n_outputs, n_layers = 1, 
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)


def get_item_from_pickle(pickle_file, item_name):
	from_pickle = load_pickle(pickle_file)
	if item_name in from_pickle:
		return from_pickle[item_name]
	return None


def get_dict_template():
	return {"observed_data": None,
			"observed_tp": None,
			"data_to_predict": None,
			"tp_to_predict": None,
			"observed_mask": None,
			"mask_predicted_data": None,
			"labels": None
			}


def normalize_data(data):
	reshaped = data.reshape(-1, data.size(-1))

	att_min = torch.min(reshaped, 0)[0]
	att_max = torch.max(reshaped, 0)[0]

	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / att_max
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	return data_norm, att_min, att_max


def normalize_masked_data(data, mask, att_min, att_max):
	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / att_max
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	# set masked out elements back to zero 
	data_norm[mask == 0] = 0

	return data_norm, att_min, att_max


def shift_outputs(outputs, first_datapoint = None):
	outputs = outputs[:,:,:-1,:]

	if first_datapoint is not None:
		n_traj, n_dims = first_datapoint.size()
		first_datapoint = first_datapoint.reshape(1, n_traj, 1, n_dims)
		outputs = torch.cat((first_datapoint, outputs), 2)
	return outputs


def split_data_extrap(data_dict, dataset = ""):
	device = get_device(data_dict["data"])

	n_observed_tp = data_dict["data"].size(1) // 2
	if dataset == "hopper":
		n_observed_tp = data_dict["data"].size(1) // 3

	split_dict = {"observed_data": data_dict["data"][:,:n_observed_tp,:].clone(),
				"observed_tp": data_dict["time_steps"][:n_observed_tp].clone(),
				"data_to_predict": data_dict["data"][:,n_observed_tp:,:].clone(),
				"tp_to_predict": data_dict["time_steps"][n_observed_tp:].clone()} # we want to predict the sucessive values

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if ("mask" in data_dict) and (data_dict["mask"] is not None):
		split_dict["observed_mask"] = data_dict["mask"][:, :n_observed_tp].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"][:, n_observed_tp:].clone()

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		split_dict["labels"] = data_dict["labels"].clone()

	split_dict["mode"] = "extrap"
	return split_dict





def split_data_interp(data_dict):
	device = get_device(data_dict["data"])

	split_dict = {"observed_data": data_dict["data"].clone(),
				"observed_tp": data_dict["time_steps"].clone(),
				"data_to_predict": data_dict["data"].clone(),
				"tp_to_predict": data_dict["time_steps"].clone()} # we want to predict values in between

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if "mask" in data_dict and data_dict["mask"] is not None:
		split_dict["observed_mask"] = data_dict["mask"].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"].clone()

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		split_dict["labels"] = data_dict["labels"].clone()

	split_dict["mode"] = "interp"
	return split_dict



def add_mask(data_dict):
	data = data_dict["observed_data"]
	mask = data_dict["observed_mask"]

	if mask is None:
		mask = torch.ones_like(data).to(get_device(data))

	data_dict["observed_mask"] = mask
	return data_dict


def subsample_observed_data(data_dict, n_tp_to_sample = None, n_points_to_cut = None):
	# n_tp_to_sample -- if not None, randomly subsample the time points. The resulting timeline has n_tp_to_sample points
	# n_points_to_cut -- if not None, cut out consecutive points on the timeline.  The resulting timeline has (N - n_points_to_cut) points

	if n_tp_to_sample is not None:
		# Randomly subsample time points
		data, time_steps, mask = subsample_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_tp_to_sample = n_tp_to_sample)

	if n_points_to_cut is not None:
		# Remove consecutive time points
		data, time_steps, mask = cut_out_timepoints(
			data_dict["observed_data"].clone(), 
			time_steps = data_dict["observed_tp"].clone(), 
			mask = (data_dict["observed_mask"].clone() if data_dict["observed_mask"] is not None else None),
			n_points_to_cut = n_points_to_cut)

	new_data_dict = {}
	for key in data_dict.keys():
		new_data_dict[key] = data_dict[key]

	new_data_dict["observed_data"] = data.clone()
	new_data_dict["observed_tp"] = time_steps.clone()
	new_data_dict["observed_mask"] = mask.clone()

	if n_points_to_cut is not None:
		# Cut the section in the data to predict as well
		# Used only for the demo on the periodic function
		new_data_dict["data_to_predict"] = data.clone()
		new_data_dict["tp_to_predict"] = time_steps.clone()
		new_data_dict["mask_predicted_data"] = mask.clone()

	return new_data_dict


def split_and_subsample_batch(data_dict, args, data_type = "train"):
	if data_type == "train":
		# Training set
		if args.extrap:
			processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
		else:
			processed_dict = split_data_interp(data_dict)

	else:
		# Test set
		if args.extrap:
			processed_dict = split_data_extrap(data_dict, dataset = args.dataset)
		else:
			processed_dict = split_data_interp(data_dict)

	# add mask
	processed_dict = add_mask(processed_dict)

	# Subsample points or cut out the whole section of the timeline
	if (args.sample_tp is not None) or (args.cut_tp is not None):
		processed_dict = subsample_observed_data(processed_dict, 
			n_tp_to_sample = args.sample_tp, 
			n_points_to_cut = args.cut_tp)

	# if (args.sample_tp is not None):
	# 	processed_dict = subsample_observed_data(processed_dict, 
	# 		n_tp_to_sample = args.sample_tp)
	return processed_dict




def compute_loss_all_batches(model,
	test_dataloader, args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1., 
	max_samples_for_eval = None):

	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["mse"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	total["pois_likelihood"] = 0
	total["ce_loss"] = 0

	n_test_batches = 0
	
	classif_predictions = torch.Tensor([]).to(device)
	all_test_labels =  torch.Tensor([]).to(device)
	hard_test_labels =  torch.Tensor([]).long().to(device)
	hard_classif_predictions = torch.Tensor([]).long().to(device)

	save_latents = 10

	for i in tqdm(range(n_batches)):
		#pdb.set_trace()
		batch_dict = get_next_batch(test_dataloader)

		results  = model.compute_all_losses(batch_dict,
			n_traj_samples = n_traj_samples, kl_coef = kl_coef, save_latents=save_latents)

		if args.classif:
			n_labels = model.n_labels #batch_dict["labels"].size(-1)
			n_traj_samples = results["label_predictions"].size(0)

			classif_predictions = torch.cat((classif_predictions, 
				results["label_predictions"].reshape(n_traj_samples, -1, n_labels)), 1)
			all_test_labels = torch.cat((all_test_labels, 
				batch_dict["labels"].reshape(-1, n_labels)), 0)
			
			hard_classif_predictions = torch.cat((hard_classif_predictions, 
				results["label_predictions"].max(-1)[1]  ), 1)
			hard_test_labels = torch.cat((hard_test_labels, 
				batch_dict["labels"].max(-1)[1]  ), 0)
		
		for key in total.keys(): 
			if key in results:
				var = results[key]
				if isinstance(var, torch.Tensor):
					var = var.detach()
				total[key] += var

		# extract latent trajetories
		if first and plot_latent:
			PCA_traj = get_pca_traj(results["latent_info"][0], num_PCA=5, num_train_PCA=10, PCA_dim=1)
			first, testing = False, False

		
		n_test_batches += 1

		if save_latents>0:

			PCA_traj = {"PCA_trajs1": get_pca_traj(results["latent_info"][0], PCA_dim=1),
						"PCA_trajs2": get_pca_traj(results["latent_info"][0], PCA_dim=2),
						"PCA_trajs3": get_pca_traj(results["latent_info"][0], PCA_dim=3) }

			save_latents = 0
			stored_latents = True

		# for speed
		results = []
		if max_samples_for_eval is not None:
			if n_batches * batch_size >= max_samples_for_eval:
				break

	
	if n_test_batches > 0:
		for key, value in total.items():
			total[key] = total[key] / n_test_batches
 
	if args.classif:
		if args.dataset == "physionet":
			#all_test_labels = all_test_labels.reshape(-1)
			# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
			all_test_labels = all_test_labels.repeat(n_traj_samples,1,1)


			idx_not_nan = ~torch.isnan(all_test_labels) # Nando's edit, original was: idx_not_nan = 1 - torch.isnan(all_test_labels)
			classif_predictions = classif_predictions[idx_not_nan]
			all_test_labels = all_test_labels[idx_not_nan]

			dirname = "plots/" + str(experimentID) + "/"
			os.makedirs(dirname, exist_ok=True)
			
			total["auc"] = 0.
			if torch.sum(all_test_labels) != 0.:
				print("Number of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
				print("Number of examples with mortality 1: {}".format(torch.sum(all_test_labels == 1.)))

				# Cannot compute AUC with only 1 class
				total["auc"] = sk.metrics.roc_auc_score(all_test_labels.cpu().numpy().reshape(-1), 
					classif_predictions.cpu().numpy().reshape(-1))
			else:
				print("Warning: Couldn't compute AUC -- all examples are from the same class")
		
		if args.dataset == "activity":
			all_test_labels = all_test_labels.repeat(n_traj_samples,1,1)

			labeled_tp = torch.sum(all_test_labels, -1) > 0.

			all_test_labels = all_test_labels[labeled_tp]
			classif_predictions = classif_predictions[labeled_tp]

			# classif_predictions and all_test_labels are in on-hot-encoding -- convert to class ids
			_, pred_class_id = torch.max(classif_predictions, -1)
			_, class_labels = torch.max(all_test_labels, -1)

			pred_class_id = pred_class_id.reshape(-1) 

			correct_labels = class_labels.cpu().numpy()
			predict_labels =  pred_class_id.cpu().numpy()

			total["accuracy"] = sk.metrics.accuracy_score(
					class_labels.cpu().numpy(), 
					pred_class_id.cpu().numpy())
			
		if args.dataset in ["crop", "swisscrop"]:
			hard_test_labels = hard_test_labels.repeat(n_traj_samples,1,1)
			
			idx_not_nan = ~torch.isnan(hard_test_labels)[0,0] # Nando's edit: idx_not_nan = 1 - torch.isnan(all_test_labels)
			hard_classif_predictions = hard_classif_predictions[0,idx_not_nan]
			hard_test_labels = hard_test_labels[0,0,idx_not_nan]
			
			#print("Number of labeled examples: {}".format(len(hard_test_labels.reshape(-1))))
			correct_labels = hard_test_labels.cpu()
			predict_labels =  hard_classif_predictions.cpu().numpy()
			total["accuracy"] = sk.metrics.accuracy_score(
					correct_labels, 
					predict_labels)
	
	classif_predictions = []
	all_test_labels =  []
	hard_test_labels =  []
	hard_classif_predictions = []
	
	if stored_latents:
		total["PCA_traj"] = PCA_traj

	return total, {"correct_labels": correct_labels, "predict_labels": predict_labels}

def check_mask(data, mask):
	#check that "mask" argument indeed contains a mask for data
	n_zeros = torch.sum(mask == 0.).cpu().numpy()
	n_ones = torch.sum(mask == 1.).cpu().numpy()

	# mask should contain only zeros and ones
	assert((n_zeros + n_ones) == np.prod(list(mask.size())))

	# all masked out elements should be zeros
	assert(torch.sum(data[mask == 0.] != 0.) == 0)


# Experimental of Nando:
class FastTensorDataLoader:
	"""
	A DataLoader-like object for a set of tensors that can be much faster than
	TensorDataset + DataLoader because dataloader grabs individual indices of
	the dataset and calls cat (slow).
	"""
	def __init__(self, dataset, batch_size=32, shuffle=False, batch_shuffle=True, early_prediction=0,
		subsamp=1.):
		"""
		Initialize a FastTensorDataLoader.

		:param *dataset: hdf5 dataset. Eg. Crop dataset
		:param batch_size: batch size to load.
		:param shuffle: if True, shuffle the data *in-place* whenever an iterator is created out of this object.
			Recommendation: set shuffle to False. Then: the underlying hd5y is than more efficient,
			because id can make use of the contiguous blocks of data.

		:returns: A FastTensorDataLoader.
		"""
		self.dataset = dataset
		self.hdf5dataloader = self.dataset.hdf5dataloader

		self.dataset_len = len(dataset)
		self.dataset_true_len = dataset.true_len__()
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_shuffle = batch_shuffle
		self.timestamps = h5py.File(os.path.join(self.dataset.processed_folder, self.dataset.time_file), "r")["tt"][:]
		self.noskip = dataset.noskip
		self.subsamp = subsamp

		#
		if type(dataset).__name__=='SwissCrops':
			self.remapping = True
			self.reflistglob = self.dataset.labellistglob13
			self.nclasses = 13
		else:
			self.remapping = False

		# prepare skipping of steps and truncation of features
		self.early_prediction = early_prediction
		
		if hasattr(self.dataset, 'step'):
			self.step = self.dataset.step
		else:
			self.step = 1
		
		if hasattr(self.dataset, 'trunc'):
			trunc = self.dataset.trunc
		else:
			trunc = self.dataset.features//self.dataset.nb**2
		
		self.feature_trunc = self.dataset.nb**2*trunc
		
		# Calculate # batches
		n_batches, remainder = divmod(self.dataset_len, self.batch_size)
		if remainder > 0: 
			n_batches += 1 # what hapens to the last one => not full right?
		self.n_batches = n_batches

		# Calculate batches for full dataset
		n_true_batches, true_remainder = divmod(self.dataset_true_len, self.batch_size)
		if true_remainder > 0: 
			n_true_batches += 1 # what hapens to the last one => not full right?
		self.n_true_batches = n_true_batches

		# initialize iterator state
		self.true_batch_indices = np.arange(self.n_true_batches)
		if self.batch_shuffle:
			np.random.seed(1996)
			np.random.shuffle(self.true_batch_indices)
		self.subsampled_batch_indices = self.true_batch_indices[:self.n_batches]

		self.singlepix = self.dataset.singlepix
		if self.singlepix:
			a = np.zeros(9, dtype=bool)
			a[4] = 1
			kronmask = np.kron(np.ones(9,dtype=bool),a)
			self.kronmask = kronmask[:self.feature_trunc]


	def __iter__(self):
		# reset iterator state
		if self.shuffle:
			self.indices = np.random.permutation(self.dataset_len)
		else:
			self.indices = None
		
		#self.batch_indices = np.arange(self.n_batches)
		self.batch_indices = self.subsampled_batch_indices
		
		if self.batch_shuffle and self.dataset.mode=="train":
			np.random.shuffle(self.batch_indices)
		
		#if self.batch_shuffle:
		#	np.random.shuffle(self.batch_indices)

		self.bi = 0
		self.i = 0
		return self

	def __next__(self):
		if self.i >= self.dataset_len: #start from beginning again 
			raise StopIteration 
			self.i = 0
		
		if self.bi >= self.n_batches: #start from beginning again
			raise StopIteration 
			self.bi = 0

		if self.indices is not None:
			indices = np.sort(self.indices[self.i:self.i+self.batch_size])
			#batch = torch.index_select(t, 0, indices)
			#pdb.set_trace()
			data = torch.from_numpy( self.hdf5dataloader["data"][indices] ).float().to(self.dataset.device)
			time_stamps = torch.from_numpy( self.timestamps ).to(self.dataset.device)
			mask = torch.from_numpy(  self.hdf5dataloader["mask"][indices] ).float().to(self.dataset.device)
			labels = torch.from_numpy( self.hdf5dataloader["labels"][indices] ).float().to(self.dataset.device)

			data_dict = {
				"data": data, 
				"time_steps": time_stamps,
				"mask": mask,
				"labels": labels}
		else:

			start = self.batch_indices[self.bi]*self.batch_size
			stop = start + self.batch_size
			
			data = torch.from_numpy( self.hdf5dataloader["data"][start:stop] ).float()#.to(self.dataset.device)
			time_stamps = torch.from_numpy( self.timestamps )#.to(self.dataset.device)
			mask = torch.from_numpy(self.hdf5dataloader["mask"][start:stop] ).float()#.to(self.dataset.device)
			labels = torch.from_numpy( self.hdf5dataloader["labels"][start:stop] ).float()#.to(self.dataset.device)

			if self.singlepix:
				data_dict = {
					"data": data[:,::self.step,self.kronmask].to(self.dataset.device), 
					"time_steps": time_stamps[::self.step].to(self.dataset.device),
					"mask": mask[:,::self.step,self.kronmask].to(self.dataset.device),
					"labels": labels}
			else:
				data_dict = {
					"data": data[:,::self.step,:self.feature_trunc].to(self.dataset.device), 
					"time_steps": time_stamps[::self.step].to(self.dataset.device),
					"mask": mask[:,::self.step,:self.feature_trunc].to(self.dataset.device),
					"labels": labels}

		if self.subsamp>0 and self.subsamp<1:
			max_len = data_dict["mask"].shape[1]
			features = data_dict["mask"].shape[2]
			validinds = [torch.nonzero(torch.sum(seq,1)) for seq in data_dict["mask"]] 
			newinds = [ inds[torch.multinomial(torch.ones(len(inds)), max(int(len(inds)*self.subsamp), 1), replacement=False )] for inds in validinds]
			data_dict["mask"] = torch.stack([ torch.zeros(max_len, dtype=torch.float32, device=self.dataset.device).scatter_(0, torch.squeeze(inds), 1) for inds in newinds]).unsqueeze(2).repeat(1,1,features)
			
			
		if self.early_prediction > 0:
			filter_rest =  torch.zeros_like(data_dict["mask"]) 
			filter_rest[:,:self.early_prediction,:] = 1
			data_dict["mask"] = data_dict["mask"] * filter_rest

		#if self.noskip:
			# Mark every frame as observed (needed for some experiments)
			#data_dict["mask"] = torch.ones_like(data_dict["mask"])
		
		#perform remapping for Swisscrops
		if self.remapping:
			targetind = torch.argmax(data_dict["labels"],1)#.numpy()
			#target = torch.zeros_like(targetind)

			for i in range(len(self.dataset.labellistglob)):
				#delete the label if it is not within the k most frequent classes k={13,23}
				if not (self.dataset.labellist[i] in self.dataset.labellist13):
					targetind[targetind == self.dataset.labellistglob[i]] = 0
			
			# Reduce range of labels
			uniquelabels = np.unique(self.reflistglob)
			for i in range(self.nclasses):
				targetind[targetind == uniquelabels[i]] = i+1

			#Convert back to one hot
			labels = torch.zeros((self.batch_size, self.nclasses+1))
			labels[np.arange(self.batch_size),targetind] = 1

			data_dict["labels"] = labels.to(self.dataset.device)

		else:
			data_dict["labels"] = data_dict["labels"].to(self.dataset.device)

		data_dict = split_and_subsample_batch(data_dict, self.dataset.args, data_type = self.dataset.mode)
				
		self.i += self.batch_size
		self.bi +=1
		return data_dict

	def __len__(self):
		return self.n_batches


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


def hyperopt_summary(trials):

	best_res = 0
	print("")
	print("---------------------------------------------------------------------------------------------------------------------------------------------")
	print('TRIAL PROTOCOL:')
	print("---------------------------------------------------------------------------------------------------------------------------------------------")

	for i, trial in enumerate(trials.trials[:]):
		message = 'Trial: {:04d}  |  #runs: {}  |  Mean Acc.: {:.3f} %  |  Std: {:.3f} %  |  Best Acc.: {:.3f} % (Peak: {} samples)  |  Hparams: {}  |'.format(
			i+1, 
			trial["result"]["num_seeds"],
			(1-trial["result"]["loss"])*100,
			np.sqrt(trial["result"]["loss_variance"])*100,
			(trial["result"]["best_acc"])*100,
			trial["result"]["best_peak_step"],
			trial["misc"]["vals"] )

		
		print(message)

		if best_res<1-trial["result"]["loss"]:
			best_res = 1-trial["result"]["loss"]
			best_var = trial["result"]["loss_variance"]
			best_param = trial["misc"]["vals"]
			best_of_best_res = trial["result"]["best_acc"]

	print("---------------------------------------------------------------------------------------------------------------------------------------------")
	message2 = "Best configuration mean: {:.3f} % (+-{:.3f}) (best run: {:.3f} %) , with Hyperparmeters: {}".format(
		best_res*100,
		np.sqrt(best_var)*100,
		best_of_best_res*100,
		best_param
	)
	print(message2)
	print("---------------------------------------------------------------------------------------------------------------------------------------------")



def get_optimizer(optimizer, lr, params):

	if optimizer == 'adagrad':
		optimizer = torch.optim.Adagrad(params, lr=lr*5, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
	elif optimizer == 'adadelta':
		optimizer = optim.Adadelta(params, lr=lr*100*5, rho=0.9, eps=1e-06, weight_decay=0)
	elif optimizer == 'adam':
		optimizer = optim.Adam(params, lr=lr/10*5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	elif optimizer == 'adaw':
		optimizer = optim.AdamW(params, lr=lr/10*5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
	elif optimizer == 'sparseadam':
		optimizer = optim.SparseAdam(params/10*5, lr=lr, betas=(0.9, 0.999), eps=1e-08)
	elif optimizer == 'ASGD':
		optimizer = optim.ASGD(params, lr=lr*5, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
	elif optimizer == 'LBFGS':
		optimizer = optim.LBFGS(params, lr=lr*100*5) 
	elif optimizer == 'RMSprop':
		optimizer = optim.RMSprop(params, lr=lr*5)
	elif optimizer == 'rprop':
		optimizer = optim.Rprop(params, lr=lr*5)
	elif optimizer == 'SGD':
		optimizer = optim.SGD(params, lr=lr*5, momentum=0, dampening=0, weight_decay=0, nesterov=False)
	elif optimizer == 'adamax': #standard: adamax
		optimizer = optim.Adamax(params, lr=lr) # best lr=0.01, standard is lr=0.002, mutiply every other by factor 5 as well
	else:
		raise Exception("Optimizer not supported. Please change it!")

	return optimizer

def plot_confusion_matrix2(target_test, pred_test, valid_labels_names, ExperimentID):

	from sklearn.metrics import confusion_matrix
	#from sklearn.metrics import ConfusionMatrixDisplay
	import seaborn as sn
	import pandas as pd
	import matplotlib.pyplot as plt

	cm = confusion_matrix(target_test, pred_test, normalize='true')

	df_cm = pd.DataFrame(cm, index = [i for i in valid_labels_names],
					columns = [i for i in valid_labels_names])
	plt.figure(figsize = (15,10))
	sn.heatmap(df_cm, annot=False,  vmin=0, vmax=1, cmap='Blues')
	#plt.xlabel('True label')
	#plt.ylabel('Predicted label')
	#plt.title('Confusion matrix')
	plt.savefig('vis/cm' + str(ExperimentID) + '.pdf', bbox_inches='tight')

	#plt.close()


def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
	''' 
	Parameters:
		correct_labels                  : These are your true classification categories.
		predict_labels                  : These are you predicted classification categories
		labels                          : This is a lit of labels which will be used to display the axix labels
		title='Confusion matrix'        : Title for your matrix
		tensor_name = 'MyFigure/image'  : Name for the output summay tensor

	Returns:
		summary: TensorFlow summary 

	Other itema to note:
		- Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
		- Currently, some of the ticks dont line up due to rotations.
	'''
	cm = confusion_matrix(correct_labels, predict_labels)#, labels=labels)
	if normalize:
		cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
		cm = np.nan_to_num(cm, copy=True)
		cm = cm.astype('int')

	np.set_printoptions(precision=2)
	###fig, ax = matplotlib.figure.Figure()

	fig = matplotlib.figure.Figure(figsize=(6, 6), dpi=320, facecolor='w', edgecolor='k')
	ax = fig.add_subplot(1, 1, 1)
	im = ax.imshow(cm, cmap='Oranges')

	classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
	classes = ['\n'.join(wrap(l, 40)) for l in classes]

	tick_marks = np.arange(len(classes))

	ax.set_xlabel('Predicted', fontsize=7)
	ax.set_xticks(tick_marks)
	c = ax.set_xticklabels(classes, fontsize=4, rotation=-90,  ha='center')
	ax.xaxis.set_label_position('bottom')
	ax.xaxis.tick_bottom()

	ax.set_ylabel('True Label', fontsize=7)
	ax.set_yticks(tick_marks)
	ax.set_yticklabels(classes, fontsize=4, va ='center')
	ax.yaxis.set_label_position('left')
	ax.yaxis.tick_left()
	ax.set_ylim(len(cm)-0.5, -0.5)

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=3.5, verticalalignment='center', color= "black")
	fig.set_tight_layout(True)
	#summary = tfplot.figure.to_summary(fig, tag=tensor_name)
	return 1, fig




