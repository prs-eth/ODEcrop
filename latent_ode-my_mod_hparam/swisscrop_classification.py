
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mai 19 14:28:46 2020

@author: metzgern
"""


import os
import h5py
import torch

import numpy as np
import csv
from tqdm import tqdm
from datetime import datetime

import lib.utils as utils
from lib.utils import FastTensorDataLoader

import pdb

class SwissCrops(object):

	# Complete list
	label = ['0_unknown', 'Barley', 'Beets', 'Berries', 'Biodiversity', 'Chestnut', 'Fallow', 'Field bean', 'Forest', 'Gardens',
		 'Grain', 'Hedge', 'Hemp', 'Hops', 'Linen', 'Maize', 'Meadow', 'MixedCrop', 'Multiple', 'Oat', 'Orchards', 'Pasture',
		 'Potatoes', 'Rapeseed', 'Rye', 'Sorghum', 'Soy', 'Spelt', 'Sugar_beets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat',
		 'unknownclass1', 'unknownclass2', 'unknownclass3']

	#Updated list after selection of most frequent ones....
	label = ['Meadow','Potatoes', 'Pasture', 'Maize', 'Sugar_beets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'WinterBarley', 'WinterRapeseed', 'WinterWheat']
	label_dict = {k: i for i, k in enumerate(label)}
	reverse_label_dict = {v: k for k, v in label_dict.items()}

	def __init__(self, root, mode='train', device = torch.device("cpu"),
		neighbourhood=3, cloud_thresh=0.05,
		nsamples=float("inf"),args=None,
		step=1, trunc=9, datatype="2_toplabels",
		singlepix=False, noskip=False):
		
		self.datatype = datatype

		self.normalize = True
		self.shuffle = True
		self.singlepix = singlepix

		self.root = root
		self.nb = neighbourhood
		self.cloud_thresh = cloud_thresh
		self.device = device
		self.n = nsamples
		self.mode = mode
		self.noskip = noskip
		if noskip:
			raise Exception("--noskip option not supported for swissdata.")			

		if args==None:
			argsdict = {"dataset": "swisscrop", "sample_tp": None, "cut_tp": None, "extrap": False}
			self.args = utils.Bunch(argsdict)
		
		# calculated from 50k samples
		#self.means = [0.40220731, 0.2304579, 0.21944561, 0.22120122, 0.00414104, 0.00608051, 0.00555058, 0.00306677, 0.00378373]
		#self.stds = [0.24774854, 0.29837374, 0.3176923, 0.29580569, 0.00475051, 0.00396885, 0.00412216, 0.00274612, 0.00241172]
		
		# define de previously calculated global training mean and std...
		self.means = [0.4071655 , 0.2441012 , 0.23429523, 0.23402453, 0.00432794, 0.00615292, 0.00566292, 0.00306609, 0.00367624]
		self.stds = [0.24994541, 0.30625425, 0.32668449, 0.30204761, 0.00490984, 0.00411067, 0.00426914, 0.0027143 , 0.00221963]

		if not self.check_exists():
			self.process_data()

		if mode=="train":
			data_file = self.train_file
		elif mode=="test":
			data_file = self.test_file

		if not os.path.exists(data_file):
			print("haven't found " + data_file + " . Starting to preprocess the whole dataset...")
			#self.process_data()
		
		self.hdf5dataloader = h5py.File(data_file, "r", rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7)
		self.nsamples = self.hdf5dataloader["data"].shape[0]
		self.nfeatures = self.hdf5dataloader["data"].shape[2]

		# get timestamps
		if not os.path.exists( os.path.join(self.processed_folder, self.time_file)):
			self.read_date_file()

		self.timestamps =  h5py.File(os.path.join(self.processed_folder, self.time_file), "r")["tt"][:]
		assert(self.timestamps.size==self.hdf5dataloader["data"].shape[1])

		self.features = self.hdf5dataloader["data"].shape[2]
		
		#selective features and timestamps
		self.step = step # skippage of timesteps
		self.trunc = trunc # feature truncation
		self.feature_trunc = trunc*self.nb**2
		#self.mask = np.kron(np.hstack([np.ones(self.trunc),np.zeros(  self.nfeatures//self.nb**2 - self.trunc)]), np.ones(9))


	def process_data(self):
	
		os.makedirs(self.processed_folder, exist_ok=True)

		train_dataset = Dataset("data/SwissCrops/", 0.,'train')
		raw_train_samples = len(train_dataset)

		test_dataset = Dataset("data/SwissCrops/", 0.,'test')
		raw_test_samples = len(test_dataset)

		num_invalid_obs = 0

		# Calculate the number of trainingsamples
		raw_batch = (24 - int(self.nb/2)*2)**2
		ntrainsamples = raw_batch * raw_train_samples
		ntestsamples =  raw_batch * raw_test_samples

		# Inplement index aliasing to perform shuffling
		trainindices = np.arange(ntrainsamples)
		testindices = np.arange(ntestsamples)
		
		## TRAIN DATASET ##
		shuffle_chucks = 20 # 30: limit of 64GB RAM, 60: limit of 32GB RAM
		splits = np.array_split(trainindices, shuffle_chucks)

		# get measures
		X, target, target_local_1, target_local_2, cloud_cover = train_dataset[0]

		raw_features = X.shape[1]
		nfeatures = raw_features* self.nb**2
		seq_length = X.shape[0]

		ntargetclasses = train_dataset.n_classes
		ntargetclasses_l1 = train_dataset.n_classes_local_1
		ntargetclasses_l2 = train_dataset.n_classes_local_2
		
		# Open a hdf5 files and create arrays
		hdf5_file_train = h5py.File(self.train_file , mode='w', rdcc_nbytes=1024**2*16000, rdcc_nslots=1e7, libver='latest')
		hdf5_file_train.create_dataset("data", (ntrainsamples, seq_length, nfeatures), np.float16, chunks=(1500, seq_length, nfeatures) )
		hdf5_file_train.create_dataset("mask", (ntrainsamples, seq_length, nfeatures), np.bool,  chunks=(1500, seq_length, nfeatures) )
		hdf5_file_train.create_dataset("labels", (ntrainsamples,  ntargetclasses), np.int8,  chunks=(1500, ntargetclasses ))
		hdf5_file_train.create_dataset("labels_local1", (ntrainsamples, ntargetclasses_l1), np.int8,  chunks=(1500, ntargetclasses_l1) )
		hdf5_file_train.create_dataset("labels_local2", (ntrainsamples, ntargetclasses_l2), np.int8,  chunks=(1500, ntargetclasses_l1) )
		
		#prepare first splitblock
		X_merge = np.zeros( (len(splits[0]), seq_length, nfeatures) , dtype=np.float16) 
		mask_merge = np.ones( (len(splits[0]), seq_length, nfeatures) , dtype=bool)
		target_merge = np.ones( (len(splits[0]),  ntargetclasses) , dtype=np.int8)
		target_l1_merge = np.ones( (len(splits[0]),  ntargetclasses_l1) , dtype=np.int8)
		target_l2_merge = np.ones( (len(splits[0]),  ntargetclasses_l2) , dtype=np.int8)

		missing = 0
		observed = 0
		first_batch = True # will be changed after the first batch
		accum_counter = 0
		split_counter = 0

		n_valid = 0
		summation = np.zeros( (raw_features) )
		sq_summation = np.zeros( (raw_features) )

		#for idx in tqdm(range(raw_train_samples)):
		for idx in tqdm(range(raw_train_samples)):
			X, target, target_local_1, target_local_2, cloud_cover = train_dataset[idx]

			# check if data can be cropped
			cloud_mask = cloud_cover>self.cloud_thresh
			invalid_obs = np.sum(cloud_mask,axis=0)==0

			sub_shape = (self.nb, self.nb)
			view_shape = tuple(np.subtract(invalid_obs.shape, sub_shape) + 1) + sub_shape
			strides = invalid_obs.strides + invalid_obs.strides
			sub_invalid = np.lib.stride_tricks.as_strided(invalid_obs,view_shape,strides)

			# store the number of invalid observations
			num_invalid_obs += np.sum ( (np.sum(sub_invalid, axis=(2,3))!=0) )
			assert(num_invalid_obs==0)

			# Prepare for running mean and std calculation
			valid_ind = np.nonzero( (~cloud_mask)[:,np.newaxis] )
			valid_data = X[valid_ind[0],:,valid_ind[2],valid_ind[3]]
			summation += valid_data.sum(0)
			sq_summation += (valid_data**2).sum(0)
			n_valid += (valid_data**2).shape[0]

			if self.normalize:
				norm_data = (valid_data-self.means)/self.stds
				X[valid_ind[0],:,valid_ind[2],valid_ind[3]] = norm_data

			#prepare mask for later
			sub_shape = (seq_length, self.nb, self.nb)
			view_shape = tuple(np.subtract(cloud_mask.shape, sub_shape) + 1) + sub_shape
			strides = cloud_mask.strides + cloud_mask.strides
			sub_cloud = np.lib.stride_tricks.as_strided(cloud_mask,view_shape,strides)

			ravel_mask = sub_cloud.reshape(raw_batch, seq_length, self.nb**2)
			cloud_mask = np.tile(ravel_mask, (1,1, raw_features))
			mask = ~cloud_mask
			 
			# Subtile the features
			sub_shape = (seq_length, raw_features, self.nb, self.nb)
			view_shape = tuple(np.subtract(X.shape, sub_shape) + 1) + sub_shape
			strides = X.strides + X.strides
			sub_X = np.lib.stride_tricks.as_strided(X,view_shape,strides)

			ravel_X = sub_X.reshape(raw_batch, sub_X.shape[4], nfeatures )

			# subconvolove Targets
			sub_shape = (self.nb, self.nb)
			view_shape = tuple(np.subtract(target.shape, sub_shape) + 1) + sub_shape
			strides = target.strides + target.strides

			sub_target = np.lib.stride_tricks.as_strided(target,view_shape,strides)
			sub_target_local_1 = np.lib.stride_tricks.as_strided(target_local_1,view_shape,strides)
			sub_target_local_2 = np.lib.stride_tricks.as_strided(target_local_2,view_shape,strides)

			ravel_mask = sub_invalid.reshape(raw_batch, 1, self.nb**2)
			ravel_target = sub_target[:,:,self.nb//2, self.nb//2].reshape(-1)
			ravel_target_local_1 = sub_target_local_1[:,:,self.nb//2, self.nb//2].reshape(-1)
			ravel_target_local_2 = sub_target_local_2[:,:,self.nb//2, self.nb//2].reshape(-1)[:]
			
			# bring to one-hot format
			OH_target = np.zeros((ravel_target.size, ntargetclasses))
			OH_target[np.arange(ravel_target.size),ravel_target] = 1

			OH_target_local_1 = np.zeros((ravel_target_local_1.size, ntargetclasses_l1))
			OH_target_local_1[np.arange(ravel_target_local_1.size),ravel_target_local_1] = 1

			OH_target_local_2 = np.zeros((ravel_target_local_2.size, ntargetclasses_l2))
			OH_target_local_2[np.arange(ravel_target_local_2.size),ravel_target_local_2] = 1

			# if only one pixel in a neighbourhood is corrupted, we don't use it=> set complete mask of this (sample, timestep) as unobserved
			mask = np.tile( (mask.sum(2)==nfeatures)[:,:,np.newaxis] , (1,1,nfeatures))

			# "destroy" data, that is corrputed by bad weather. We will never use it!
			ravel_X[~mask] = 0

			#for statistics
			missing += np.sum(mask == 0.)
			observed += np.sum(mask == 1.)

			# Accummulate data before writing it to file
			# fill in HDF5 file
			if first_batch:
				start_ix = 0
				stop_ix = raw_batch
				first_batch = False
			else:
				start_ix = stop_ix
				stop_ix += raw_batch

			if stop_ix<len(splits[split_counter]):
				#write to memory file
				X_merge[start_ix:stop_ix] = ravel_X
				mask_merge[start_ix:stop_ix] = mask
				target_merge[start_ix:stop_ix] = OH_target
				target_l1_merge[start_ix:stop_ix] = OH_target_local_1
				target_l2_merge[start_ix:stop_ix] = OH_target_local_2

			else:
				#Write to file, if merge is big enough
				#determine th amount of overdose
				overdose = stop_ix - len(splits[split_counter])
				validdose = raw_batch - overdose 

				# add to memory only how much fits in it
				X_merge[start_ix:] = ravel_X[:validdose]
				mask_merge[start_ix:] = mask[:validdose]
				target_merge[start_ix:] = OH_target[:validdose]
				target_l1_merge[start_ix:] = OH_target_local_1[:validdose]
				target_l2_merge[start_ix:] = OH_target_local_2[:validdose]
				
				#shuffle the blocks
				self.shuffle = True
				if self.shuffle:
					merge_ind = np.arange(len(splits[split_counter]))
					np.random.shuffle(merge_ind)

					X_merge_write = X_merge[merge_ind]
					mask_merge_write = mask_merge[merge_ind]
					target_merge_write = target_merge[merge_ind]
					target_l1_merge_write = target_l1_merge[merge_ind]
					target_l2_merge_write = target_l2_merge[merge_ind]
				else:
					X_merge_write = X_merge
					mask_merge_write = mask_merge
					target_merge_write = target_merge
					target_l1_merge_write = target_l1_merge
					target_l2_merge_write = target_l2_merge

				#fill in data to hdf5 file
				sorted_indices = splits[split_counter]
				
				hdf5_file_train["data"][sorted_indices[0]:sorted_indices[-1]+1, ...] = X_merge_write
				hdf5_file_train["mask"][sorted_indices[0]:sorted_indices[-1]+1, ...] = mask_merge_write
				hdf5_file_train["labels"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_merge_write
				hdf5_file_train["labels_local1"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_l1_merge_write
				hdf5_file_train["labels_local2"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_l2_merge_write
				
				accum_counter = 0
				split_counter += 1

				#prepare next merge variable
				if split_counter<len(splits):
					X_merge = np.zeros( (len(splits[split_counter]), seq_length, nfeatures) , dtype=np.float16) 
					mask_merge = np.ones( (len(splits[split_counter]), seq_length, nfeatures) , dtype=bool)
					target_merge = np.ones( (len(splits[split_counter]),  ntargetclasses) , dtype=np.int8)
					target_l1_merge = np.ones( (len(splits[split_counter]),  ntargetclasses_l1) , dtype=np.int8)
					target_l2_merge = np.ones( (len(splits[split_counter]),  ntargetclasses_l2) , dtype=np.int8)

					# fill in the overdose from the current split/chunck
					start_ix = 0
					stop_ix = overdose

					X_merge[start_ix:stop_ix] = ravel_X[validdose:]
					mask_merge[start_ix:stop_ix] = mask[validdose:]
					target_merge[start_ix:stop_ix] = OH_target[validdose:]
					target_l1_merge[start_ix:stop_ix] = OH_target_local_1[validdose:]
					target_l2_merge[start_ix:stop_ix] = OH_target_local_2[validdose:]

			accum_counter += 1
		
		print("found ", num_invalid_obs, " invalid Neighbourhood-Observations in training data")
		assert(num_invalid_obs==0)

		## TEST DATASET ##
		shuffle_chucks = 25 #15 # 30: limit of 64GB RAM, 60: limit of 32GB RAM
		splits = np.array_split(testindices, shuffle_chucks)
	
		hdf5_file_test = h5py.File(self.test_file, mode='w', rdcc_nbytes =1024**2*24000, rdcc_nslots=1e7, libver='latest')
		hdf5_file_test.create_dataset("data", (ntestsamples, seq_length, nfeatures), np.float16, chunks=(10000, seq_length, nfeatures) )
		hdf5_file_test.create_dataset("mask", (ntestsamples, seq_length, nfeatures), np.bool,  chunks=(10000, seq_length, nfeatures) )
		hdf5_file_test.create_dataset("labels", (ntestsamples, ntargetclasses), np.int8,  chunks=(10000, ntargetclasses) )
		hdf5_file_test.create_dataset("labels_local1", (ntestsamples, ntargetclasses_l1), np.int8,  chunks=(10000, ntargetclasses_l1) )
		hdf5_file_test.create_dataset("labels_local2", (ntestsamples, ntargetclasses_l2), np.int8,  chunks=(10000, ntargetclasses_l2) )
		
		#prepare first splitblock
		X_merge = np.zeros( (len(splits[0]), seq_length, nfeatures) , dtype=np.float16) 
		mask_merge = np.ones( (len(splits[0]), seq_length, nfeatures) , dtype=bool)
		target_merge = np.ones( (len(splits[0]),  ntargetclasses) , dtype=np.int8)
		target_l1_merge = np.ones( (len(splits[0]),  ntargetclasses_l1) , dtype=np.int8)
		target_l2_merge = np.ones( (len(splits[0]),  ntargetclasses_l2) , dtype=np.int8)

		missing = 0
		observed = 0
		first_batch = True # will be changed after the first batch
		accum_counter = 0
		split_counter = 0

		#for idx in tqdm(range(raw_test_samples)):
		for idx in tqdm(range(raw_test_samples)):
			X, target, target_local_1, target_local_2, cloud_cover = test_dataset[idx]

			# check if data can be cropped
			cloud_mask = cloud_cover>self.cloud_thresh
			invalid_obs = np.sum(cloud_mask,axis=0)==0

			sub_shape = (self.nb, self.nb)
			view_shape = tuple(np.subtract(invalid_obs.shape, sub_shape) + 1) + sub_shape
			strides = invalid_obs.strides + invalid_obs.strides
			sub_invalid = np.lib.stride_tricks.as_strided(invalid_obs,view_shape,strides)

			# store the number of invalid observations
			num_invalid_obs += np.sum ( (np.sum(sub_invalid, axis=(2,3))!=0) )
			assert(num_invalid_obs==0)

			# Prepare for running mean and std calculation
			valid_ind = np.nonzero( (~cloud_mask)[:,np.newaxis] )
			valid_data = X[valid_ind[0],:,valid_ind[2],valid_ind[3]]

			if self.normalize:
				norm_data = (valid_data-self.means)/self.stds
				X[valid_ind[0],:,valid_ind[2],valid_ind[3]] = norm_data

			#prepare mask for later
			sub_shape = (seq_length, self.nb, self.nb)
			view_shape = tuple(np.subtract(cloud_mask.shape, sub_shape) + 1) + sub_shape
			strides = cloud_mask.strides + cloud_mask.strides
			sub_cloud = np.lib.stride_tricks.as_strided(cloud_mask,view_shape,strides)

			ravel_mask = sub_cloud.reshape(raw_batch, seq_length, self.nb**2)
			cloud_mask = np.tile(ravel_mask, (1,1, raw_features))
			mask = ~cloud_mask
			 
			# Subtile the features
			sub_shape = (seq_length, raw_features, self.nb, self.nb)
			view_shape = tuple(np.subtract(X.shape, sub_shape) + 1) + sub_shape
			strides = X.strides + X.strides
			sub_X = np.lib.stride_tricks.as_strided(X,view_shape,strides)

			ravel_X = sub_X.reshape(raw_batch, sub_X.shape[4], nfeatures )

			# subconvolove Targets
			sub_shape = (self.nb, self.nb)
			view_shape = tuple(np.subtract(target.shape, sub_shape) + 1) + sub_shape
			strides = target.strides + target.strides

			sub_target = np.lib.stride_tricks.as_strided(target,view_shape,strides)
			sub_target_local_1 = np.lib.stride_tricks.as_strided(target_local_1,view_shape,strides)
			sub_target_local_2 = np.lib.stride_tricks.as_strided(target_local_2,view_shape,strides)

			ravel_mask = sub_invalid.reshape(raw_batch, 1, self.nb**2)
			ravel_target = sub_target[:,:,self.nb//2, self.nb//2].reshape(-1)
			ravel_target_local_1 = sub_target_local_1[:,:,self.nb//2, self.nb//2].reshape(-1)
			ravel_target_local_2 = sub_target_local_2[:,:,self.nb//2, self.nb//2].reshape(-1)[:]
			
			# bring to one-hot format
			OH_target = np.zeros((ravel_target.size, ntargetclasses))
			OH_target[np.arange(ravel_target.size),ravel_target] = 1

			OH_target_local_1 = np.zeros((ravel_target_local_1.size, ntargetclasses_l1))
			OH_target_local_1[np.arange(ravel_target_local_1.size),ravel_target_local_1] = 1

			OH_target_local_2 = np.zeros((ravel_target_local_2.size, ntargetclasses_l2))
			OH_target_local_2[np.arange(ravel_target_local_2.size),ravel_target_local_2] = 1

			# if only one pixel in a neighbourhood is corrupted, we don't use it=> set complete mask of this (sample, timestep) as unobserved
			mask = np.tile( (mask.sum(2)==nfeatures)[:,:,np.newaxis] , (1,1,nfeatures))

			# "destroy" data, that is corrputed by bad weather. We will never use it!
			ravel_X[~mask] = 0

			#for statistics
			missing += np.sum(mask == 0.)
			observed += np.sum(mask == 1.)

			# Accummulate data before writing it to file
			# fill in HDF5 file
			if first_batch:
				start_ix = 0
				stop_ix = raw_batch
				first_batch = False
			else:
				start_ix = stop_ix
				stop_ix += raw_batch

			if stop_ix<len(splits[split_counter]):
				#write to memory file
				X_merge[start_ix:stop_ix] = ravel_X
				mask_merge[start_ix:stop_ix] = mask
				target_merge[start_ix:stop_ix] = OH_target
				target_l1_merge[start_ix:stop_ix] = OH_target_local_1
				target_l2_merge[start_ix:stop_ix] = OH_target_local_2

			else:
				#Write to file, if merge is big enough
				#determine th amount of overdose
				overdose = stop_ix - len(splits[split_counter])
				validdose = raw_batch - overdose 

				# add to memory only how much fits in it
				X_merge[start_ix:] = ravel_X[:validdose]
				mask_merge[start_ix:] = mask[:validdose]
				target_merge[start_ix:] = OH_target[:validdose]
				target_l1_merge[start_ix:] = OH_target_local_1[:validdose]
				target_l2_merge[start_ix:] = OH_target_local_2[:validdose]

				#shuffle the blocks
				shuffle_test = True
				if shuffle_test:
					merge_ind = np.arange(len(splits[split_counter]))
					np.random.shuffle(merge_ind)

					X_merge = X_merge[merge_ind]
					mask_merge = mask_merge[merge_ind]
					target_merge = target_merge[merge_ind]
					target_l1_merge = target_l1_merge[merge_ind]
					target_l2_merge = target_l2_merge[merge_ind]

				#fill in data to hdf5 file
				sorted_indices = splits[split_counter]
				
				hdf5_file_test["data"][sorted_indices[0]:sorted_indices[-1]+1, ...] = X_merge
				hdf5_file_test["mask"][sorted_indices[0]:sorted_indices[-1]+1, ...] = mask_merge
				hdf5_file_test["labels"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_merge
				hdf5_file_test["labels_local1"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_l1_merge
				hdf5_file_test["labels_local2"][sorted_indices[0]:sorted_indices[-1]+1, ...] = target_l2_merge
				
				accum_counter = 0
				split_counter += 1

				#prepare next merge variable
				if split_counter<len(splits):
					X_merge = np.zeros( (len(splits[split_counter]), seq_length, nfeatures) , dtype=np.float16) 
					mask_merge = np.ones( (len(splits[split_counter]), seq_length, nfeatures) , dtype=bool)
					target_merge = np.ones( (len(splits[split_counter]),  ntargetclasses) , dtype=np.int8)
					target_l1_merge = np.ones( (len(splits[split_counter]),  ntargetclasses_l1) , dtype=np.int8)
					target_l2_merge = np.ones( (len(splits[split_counter]),  ntargetclasses_l2) , dtype=np.int8)

					# fill in the overdose from the current split/chunck
					start_ix = 0
					stop_ix = overdose

					X_merge[start_ix:stop_ix] = ravel_X[validdose:]
					mask_merge[start_ix:stop_ix] = mask[validdose:]
					target_merge[start_ix:stop_ix] = OH_target[validdose:]
					target_l1_merge[start_ix:stop_ix] = OH_target_local_1[validdose:]
					target_l2_merge[start_ix:stop_ix] = OH_target_local_2[validdose:]

			accum_counter += 1

		print("found ", num_invalid_obs, " invalid Neighbourhood-Observations in validation data")
		assert(num_invalid_obs==0)

		print("Valid observations: ", (observed/(observed+missing))*100, "%")

		# Calculate mean and std on train
		showmeanstd=True
		if showmeanstd:
			
			print("Calculating mean and standard deviation of training dataset ...")
			training_mean2 = summation/n_valid
			training_std2 = np.sqrt( sq_summation/n_valid - training_mean2**2 )

			print("Means: ", training_mean2)
			print("Std: ", training_std2)
				
		hdf5_file_train.close()
		hdf5_file_test.close()

		print("Preprocessing finished")

	def read_date_file(self):
		
		# read file and strip the \n 
		lines = [line.rstrip('\n') for line in open(self.raw_time_file)]
					
		# define time formate
		dates = [datetime(int(line[:4]),int(line[4:6]),int(line[6:8]))  for line in lines ]
		ref_date = dates[0]

		# calculate time difference to the reference and save in numpy variable
		times = np.asarray([(date-ref_date).days for date in dates])
		tt = (times)

		#normalize it to one
		tt = times/times[-1]
		
		timestamps_hdf5 = h5py.File(os.path.join(self.processed_folder, self.time_file), 'w')
		timestamps_hdf5.create_dataset('tt', data=tt)
		timestamps_hdf5.close()

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def raw_file(self):
		return os.path.join(self.raw_folder, "train_set_24x24_debug.hdf5")

	@property
	def train_file(self):
		return os.path.join(self.processed_folder, "train_set_3x3_processed_13_" + self.datatype + ".hdf5")

	@property
	def test_file(self):
		return os.path.join(self.processed_folder, "test_set_3x3_processed_13_" + self.datatype + ".hdf5")

	@property
	def raw_time_file(self):
		return os.path.join(self.root, 'raw_dates.txt')

	@property
	def time_file(self):
		#name only without path for consitency with other datasets
		return 'raw_dates.hdf5'
	
	def get_label(self, record_id):
		return self.label_dict[record_id]
	
	def get_label_name(self, record_id):
		return self.reverse_label_dict[record_id]
	
	@property
	def label_list(self):
		return self.label
	
	def check_exists(self):
		exist_train = os.path.exists( self.train_file )
		exist_test = os.path.exists( self.test_file )
		
		if not (exist_train and exist_test):
			return False
		return True

	def __len__(self):
		# returns the number of samples that are actually used
		if self.mode=="train":
			return min(self.n, self.hdf5dataloader["data"].shape[0])
		else:
			return min(self.n, self.hdf5dataloader["data"].shape[0])
	
	def true_len__(self):
		# returns the number of samples of the entire dataset
		if self.mode=="train":
			return self.hdf5dataloader["data"].shape[0]
		else:
			return self.hdf5dataloader["data"].shape[0]

	def __getitem__(self, index):
		"""
		Class
		For slicing and dataloading, it is suggested to use the FastDataLoader class. It makes loading way faster and includes shuffling and batching.
		"""
		if isinstance(index, slice):
			print("Warning: Slicing of hdf5 data can be slow")
			output = []
			start = 0 if index.start is None else index.start
			step = 1 if index.start is None else index.step

			data = torch.from_numpy( self.hdf5dataloader["data"][start:index.stop:step] ).float().to(self.device)
			time_stamps = torch.from_numpy( self.timestamps ).to(self.device)
			mask = torch.from_numpy(  self.hdf5dataloader["mask"][start:index.stop:step] ).float().to(self.device)
			labels = torch.from_numpy( self.hdf5dataloader["labels"][start:index.stop:step] ).float().to(self.device)
			
			#make it a dictionary to replace the collate function....
			data_dict = {
				"data": data, 
				"time_steps": time_stamps,
				"mask": mask,
				"labels": labels}

			data_dict = utils.split_and_subsample_batch(data_dict, self.args, data_type = self.mode)
			
			return data_dict
		else:
			data = torch.from_numpy( self.hdf5dataloader["data"][index] ).float().to(self.device)
			time_stamps = torch.from_numpy( self.timestamps ).to(self.device)
			mask = torch.from_numpy(self.hdf5dataloader["mask"][index] ).float().to(self.device)
			labels = torch.from_numpy( self.hdf5dataloader["labels"][index] ).float().to(self.device)

			if self.singlepix:
				# create mask
				a = np.zeros(9, dtype=bool)
				a[4] = 1
				kronmask = np.kron(np.ones(9,dtype=bool),a)
				self.kronmask = kronmask[:self.feature_trunc]
				
				#load masked data
				data_dict = {
					"data": data[::self.step,self.kronmask], 
					"time_steps": time_stamps[::self.step],
					"mask": mask[::self.step,self.kronmask],
					"labels": labels}
			else:
				data_dict = {
				"data": data[::self.step,:self.feature_trunc], 
				"time_steps": time_stamps[::self.step],
				"mask": mask[::self.step,:self.feature_trunc],
				"labels": labels}

			data_dict = utils.split_and_subsample_batch(data_dict, self.args, data_type = self.mode)

			return data_dict

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)



class Dataset(torch.utils.data.Dataset):
	def __init__(self, root, t=0.9, mode='all', eval_mode=False, fold=None, gt_path='data/SwissCrops/labels.csv',
		step=1, feature_trunc=10):
		
		self.root = root
		self.t = t
		self.augment_rate = 0
		self.step = step
		self.featrue_trunc = feature_trunc

		self.eval_mode = eval_mode
		self.fold = fold
		self.gt_path = gt_path

		self.shuffle = True
		self.normalization = True

		self.mode = mode

		self.data = h5py.File(self.raw_file, "r")
		self.samples = self.data["data"].shape[0]
		self.max_obs = self.data["data"].shape[1]
		self.spatial = self.data["data"].shape[2:-1]
		#self.n_classes = np.max( self.data["gt"] ) + 1
		
		
		#Get train/test split
		if self.fold != None:
			print('5fold: ', fold, '  Mode: ', mode)
			self.valid_list = self.split_5fold(mode, self.fold) 
		else:
			self.valid_list = self.split(mode)
		
		self.valid_samples = self.valid_list.shape[0]
		
		#self.max_obs = 71

		gt_path_ = gt_path		
		if not os.path.exists(gt_path):
			gt_path_ = './'  + gt_path
					
		file=open(gt_path_, "r")
		tier_1 = []
		tier_2 = []
		tier_3 = []
		tier_4 = []
		reader = csv.reader(file)
		for line in reader:
			tier_1.append(line[-5])
			tier_2.append(line[-4])
			tier_3.append(line[-3])
			tier_4.append(line[-2])
	
		tier_2[0] = '0_unknown'
		tier_3[0] = '0_unknown'
		tier_4[0] = '0_unknown'
	
		self.label_list = []
		for i in range(len(tier_2)):
			if tier_2[i] == '':
				tier_2[i] = '0_unknown'
			if tier_3[i] == '':
				tier_3[i] = '0_unknown'
			if tier_4[i] == '':
				tier_4[i] = '0_unknown'
			
			### Attention: Changed ozgur's code here ###
			#if tier_1[i] == 'Vegetation':
			if tier_1[i] == 'Vegetation' and tier_4[i] in ['Meadow','Potatoes', 'Pasture', 'Maize', 'Sugar_beets', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'WinterBarley', 'WinterRapeseed', 'WinterWheat', 'Spelt',
															'Hedge', 'Apples', 'Soy', 'Fallow', 'Peas', 'Berries', 'Oat', 'Field bean', 'Einkorn wheat', 'Rye', 'Tree Crop', 'Summer wheat']:
				self.label_list.append(i)

		tier_2_elements = list(set(tier_2))
		tier_3_elements = list(set(tier_3))
		tier_4_elements = list(set(tier_4))
		tier_2_elements.sort()
		tier_3_elements.sort()
		tier_4_elements.sort()
			
		tier_2_ = []
		tier_3_ = []
		tier_4_ = []
		for i in range(len(tier_2)):
			tier_2_.append(tier_2_elements.index(tier_2[i]))
			tier_3_.append(tier_3_elements.index(tier_3[i]))
			tier_4_.append(tier_4_elements.index(tier_4[i]))		

		self.label_list_local_1 = []
		self.label_list_local_2 = []
		self.label_list_glob = []
		self.label_list_local_1_name = []
		self.label_list_local_2_name = []
		self.label_list_glob_name = []
		for gt in self.label_list:
			self.label_list_local_1.append(tier_2_[int(gt)])
			self.label_list_local_2.append(tier_3_[int(gt)])
			self.label_list_glob.append(tier_4_[int(gt)])
			
			self.label_list_local_1_name.append(tier_2[int(gt)])
			self.label_list_local_2_name.append(tier_3[int(gt)])
			self.label_list_glob_name.append(tier_4[int(gt)])


		"""	
		for i in range(len(self.label_list_glob_name)):
			print(i, ' , ' ,self.label_list[i], ' , ' ,self.label_list_local_1[i],  ' , ' ,self.label_list_local_2[i], ' , ' ,self.label_list_glob[i])
			print(i, ' , ' ,self.label_list[i], ' , ' ,self.label_list_local_1_name[i],  ' , ' ,self.label_list_local_2_name[i], ' , ' ,self.label_list_glob_name[i])
			print('-'*20)
		"""

		self.n_classes = max(self.label_list_glob) + 1
		self.n_classes_local_1 = max(self.label_list_local_1) + 1
		self.n_classes_local_2 = max(self.label_list_local_2) + 1

		print('Dataset size: ', self.samples)
		print('Valid dataset size: ', self.valid_samples)
		print('Sequence length: ', self.max_obs)
		print('Spatial size: ', self.spatial)
		print('Number of classes: ', self.n_classes)
		print('Number of classes - local-1: ', self.n_classes_local_1)
		print('Number of classes - local-2: ', self.n_classes_local_2)


		#for consistency loss---------------------------------------------------------
		self.l1_2_g = np.zeros(self.n_classes)
		self.l2_2_g = np.zeros(self.n_classes)
		self.l1_2_l2 = np.zeros(self.n_classes_local_2)
		
		for i in range(1,self.n_classes):
			if i in self.label_list_glob:
				self.l1_2_g[i] = self.label_list_local_1[self.label_list_glob.index(i)]
				self.l2_2_g[i] = self.label_list_local_2[self.label_list_glob.index(i)]
		
		for i in range(1,self.n_classes_local_2):
			if i in self.label_list_local_2:
				self.l1_2_l2[i] = self.label_list_local_1[self.label_list_local_2.index(i)]
		#for consistency loss---------------------------------------------------------
		
		
	def __len__(self):
		return self.valid_samples

	@property
	def raw_folder(self):
		return os.path.join(self.root, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, 'processed')

	@property
	def raw_file(self):
		return os.path.join(self.raw_folder, "train_set_24x24_debug.hdf5")

	def __getitem__(self, idx):
					 
		idx = self.valid_list[idx]
		X = self.data["data"][idx]
		target_ = self.data["gt"][idx,...,0]
		cloud_cover = self.data["cloud_cover"][idx,...]
		if self.eval_mode:
			gt_instance = self.data["gt_instance"][idx,...,0]

		X = np.transpose(X, (0, 3, 1, 2))
		
		#Use half of the time series
		step = self.step
		feature_trunc = self.featrue_trunc
		if not (step==1 and feature_trunc>=9):
			X = X[0::step,:feature_trunc,...]
			cloud_cover = cloud_cover[0::step,...]
		
		#X = X[self.dates,...] 
		
		#Change labels 
		target = np.zeros_like(target_)
		target_local_1 = np.zeros_like(target_)
		target_local_2 = np.zeros_like(target_)
		for i in range(len(self.label_list)):
			#target[target_ == self.label_list[i]] = i
#			target[target_ == self.label_list[i]] = self.tier_4_elements_reduced.index(self.label_list_glob[i])
#			target_local_1[target_ == self.label_list[i]] = self.tier_2_elements_reduced.index(self.label_list_local_1[i])
#			target_local_2[target_ == self.label_list[i]] = self.tier_3_elements_reduced.index(self.label_list_local_2[i])
			target[target_ == self.label_list[i]] = self.label_list_glob[i]
			target_local_1[target_ == self.label_list[i]] = self.label_list_local_1[i]
			target_local_2[target_ == self.label_list[i]] = self.label_list_local_2[i]

		"""
		X = torch.from_numpy(X)
		cloud_cover = torch.from_numpy(cloud_cover).float()
		target = torch.from_numpy(target).float()
		target_local_1 = torch.from_numpy(target_local_1).float()
		target_local_2 = torch.from_numpy(target_local_2).float()
		if self.eval_mode:
			gt_instance = torch.from_numpy(gt_instance).float()

		"""
		#augmentation
		if self.eval_mode==False and np.random.rand() < self.augment_rate:
			flip_dir  = np.random.randint(3)
			if flip_dir == 0:
				X = X.flip(2)
				X = X.flip(2)
				target = target.flip(0)
				target_local_1 = target_local_1.flip(0)
				target_local_2 = target_local_2.flip(0)
				if self.eval_mode:					
					gt_instance = gt_instance.flip(0)
			elif flip_dir == 1:
				X = X.flip(3)
				cloud_cover = cloud_cover.flip(3)
				target = target.flip(1)
				target_local_1 = target_local_1.flip(1)
				target_local_2 = target_local_2.flip(1)
				if self.eval_mode:					
					gt_instance = gt_instance.flip(1)	
			elif flip_dir == 2:
				X = X.flip(2,3)
				cloud_cover = cloud_cover.flip(2,3)
				target = target.flip(0,1)  
				target_local_1 = target_local_1.flip(0,1)  
				target_local_2 = target_local_2.flip(0,1)  
				if self.eval_mode:					
					gt_instance = gt_instance.flip(0,1)

		#keep values between 0-1
		X = X * 1e-4
		
		if self.eval_mode:  
			return X.float(), target.long(), target_local_1.long(), target_local_2.long(), cloud_cover.long(), gt_instance.long()	 
		else:
			return X, target, target_local_1, target_local_2, cloud_cover
			#return X.float(), target.long(), target_local_1.long(), target_local_2.long(), cloud_cover.long()



	def get_rid_small_fg_tiles(self):
		valid = np.ones(self.samples)
		w,h = self.data["gt"][0,...,0].shape
		for i in range(self.samples):
			if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
				valid[i] = 0
		
		return np.nonzero(valid)[0]
		
	def split(self, mode):
		valid = np.zeros(self.samples)
		if mode=='test':
			valid[int(self.samples*0.75):] = 1.
		elif mode=='train':
			valid[:int(self.samples*0.75)] = 1.
		else:
			valid[:] = 1.

		w,h = self.data["gt"][0,...,0].shape
		for i in range(self.samples):
			if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
				valid[i] = 0
		
		return np.nonzero(valid)[0]

	def split_5fold(self, mode, fold):
		
		if fold == 1:
			test_s = int(0)
			test_f = int(self.samples*0.2)
		elif fold == 2:
			test_s = int(self.samples*0.2)
			test_f = int(self.samples*0.4)
		elif fold == 3:
			test_s = int(self.samples*0.4)
			test_f = int(self.samples*0.6)
		elif fold == 4:
			test_s = int(self.samples*0.6)
			test_f = int(self.samples*0.8)
		elif fold == 5:
			test_s = int(self.samples*0.8)
			test_f = int(self.samples)			
					 
		if mode=='test':
			valid = np.zeros(self.samples)
			valid[test_s:test_f] = 1.
		elif mode=='train':
			valid = np.ones(self.samples)
			valid[test_s:test_f] = 0.


		w,h = self.data["gt"][0,...,0].shape
		for i in range(self.samples):
			if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
				valid[i] = 0
		
		return np.nonzero(valid)[0]
	
	
	def chooose_dates(self):
		#chosen_dates = np.zeros(self.max_obs)
		#samples = np.random.randint(self.samples, size=1000)
		#samples = np.sort(samples)
		samples = self.data["cloud_cover"][0::10,...]
		samples = np.mean(samples, axis=(0,2,3))
		#print(np.nonzero(samples<0.1))
		return np.nonzero(samples<0.1)

	def chooose_dates_2(self):
		data_dir = '/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24/'
		DATA_YEAR = '2019'
		date_list = []
		batch_dirs = os.listdir(data_dir)
		for batch_count, batch in enumerate(batch_dirs):
			for filename in glob.iglob(data_dir + batch + '/**/patches_res_R10m.npz', recursive=True):
					date = filename.find(DATA_YEAR)
					date = filename[date:date+8]
					if date not in date_list:
						date_list.append(date)
		
		dates_text_file = open("./dates_1.txt", "r")
		specific_dates = dates_text_file.readlines()

		print('Number of dates: ', len(specific_dates))
		specific_date_indexes = np.zeros(len(specific_dates))
		for i in range(len(specific_dates)):
			specific_date_indexes[i] = date_list.index(specific_dates[i][:-1])
			
		return specific_date_indexes.astype(int)
	

	def data_stat(self):
		gt_set = self.data["gt"][self.valid_list,...,0]
		bins = np.arange(125+1)
		hist = np.histogram(gt_set, bins=bins)[0]

		#Normalize 
		#hist = hist/np.sum(hist)
		#sort histogram 
		#sorted_classes = np.argsort(hist)
		#hist_sorted = hist[sorted_classes.astype(int)]
		print('Histograms')
		print(hist)
		#plt.bar(bins[:-1],hist)
		#plt.plot(hist,'r')
#		plt.title("density") 
#		plt.savefig('train_gt_density_TG19.png')
#		plt.close()
		
		import csv
		
		with open('hist.csv', 'w', newline='') as myfile:
			 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			 wr.writerow(hist)


if __name__=="__main__":

	bs = 600

	train_dataset_obj = SwissCrops('data/SwissCrops', mode="train", datatype="2_toplabels")
	trainloader = FastTensorDataLoader(train_dataset_obj, batch_size=bs, shuffle=False)
	train_generator = utils.inf_generator(trainloader)


	#train_dataset = Dataset("data/SwissCrops/", 0.,'train')
	#raw_train_samples = len(train_dataset)

	#test_dataset = Dataset("data/SwissCrops/", 0.,'test')
	#raw_test_samples = len(test_dataset)

	#train_dataset_obj[0]
	print("Done")

	#pdb.set_trace()

	#Speed test
	#for t in tqdm(range(len(trainloader))):

	#	batch_dict = utils.get_next_batch(train_generator)
	