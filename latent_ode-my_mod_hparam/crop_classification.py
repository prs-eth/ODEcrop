#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:28:46 2020

@author: metzgern
"""

import os
import torch
#import cPickle as pickle
import pickle as pickle
from tqdm import tqdm
import numpy as np
import h5py
import lib.utils as utils
import pdb

class Crops(object):
	
	#make label Tags
	label = [ "other", "corn", "meadow", "asparagus", "rape", "hop", "summer oats", "winter spelt", "fallow", "winter wheat",
		   "winter barley", "winter rye", "beans", "winter triticale", "summer barley", "peas", "potatoe", "soybeans", "sugar beets" ]
	label_dict = {k: i for i, k in enumerate(label)}
	reverse_label_dict = {v: k for k, v in label_dict.items()}
	
	
	def __init__(self, root, args, download=False,
		reduce='average', mode='train', minseqlength=20,
		n_samples = None, device = torch.device("cpu"), list_form = True,
		step=1, trunc=6):
		
		self.list_form = list_form
		self.root = root
		self.reduce = reduce
		self.mode = mode
		self.device = device
		self.args = args
		self.second = False
		self.normalize = True
		self.shuffle = True
		self.nb = 3
				
		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if self.mode=="train":
			data_file = self.train_file
		elif self.mode=="eval":
			data_file = self.eval_file
		elif self.mode=="test":
			data_file = self.test_file
		
		self.hdf5dataloader = h5py.File(os.path.join(self.processed_folder, data_file), "r")
		self.nsamples = self.hdf5dataloader["data"].shape[0]
		self.features = self.hdf5dataloader["data"].shape[2]
		
		self.timestamps = h5py.File(os.path.join(self.processed_folder, self.time_file), "r")["tt"][:]
		
		# create mask
		self.step = step
		self.trunc = trunc# use all features
		self.feature_trunc = trunc*self.nb**2

		#for statistics
		#self.absolute_class_distribution = np.sum(self.hdf5dataloader["labels"], axis=0)
		#self.relative_class_distribution = self.absolute_class_distribution/np.sum(self.absolute_class_distribution)
		
		
	def download(self):
		
		if self._check_exists():
			return
		
		#create the directories
		os.makedirs(self.processed_folder, exist_ok=True)
		
		print('Downloading data...')
		
		# get the dataset from the web
		os.system('wget ftp://m1370728:m1370728@138.246.224.34/data.zip')
		os.system('unzip data.zip -d ' + self.raw_folder)
		os.system('rm data.zip')
		
		#Processing data
		print('Scanning data...')
		
		#collect all the possible time stamps
		train_localdir = os.path.join(self.raw_folder, 'data', 'train')
		test_localdir = os.path.join(self.raw_folder, 'data', 'test')
		eval_localdir = os.path.join(self.raw_folder, 'data', 'eval')
		first = True
		timeC = 0
		badweather_labels = np.array([0,1,2,3])
		
		unique_times = np.array([0])
		for filename in (os.listdir(train_localdir)):
			with open(os.path.join(train_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				X, Y, _ = u.load()
				if first:
					raw_batchsize, maxobs, nfeatures = X.shape
					_, _, nclasses = Y.shape
					first=False
				unique_times = np.unique(np.hstack([ X[:,:,0].ravel(), unique_times] ))
		
		unique_times = np.array([0])
		for filename in (os.listdir(test_localdir)):
			with open(os.path.join(test_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				X, _, _ = u.load()
				unique_times = np.unique(np.hstack([ X[:,:,0].ravel(), unique_times] ))
				
		unique_times = np.array([0])
		for filename in (os.listdir(eval_localdir)):
			with open(os.path.join(eval_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				X, _, _ = u.load()
				unique_times = np.unique(np.hstack([ X[:,:,timeC].ravel(), unique_times] ))
		
		#write timestamps file
		timestamps_hdf5 = h5py.File(os.path.join(self.processed_folder, self.time_file), 'w')
		timestamps_hdf5.create_dataset('tt', data=unique_times)
		
		trainbatchsizes = []
		
		#HDF5 style dataset
		#adjust the numbers! Or just resize in the end...
		
		print('Scanning Training data...')
		for filename in tqdm(os.listdir(train_localdir)):
			
			#starting a new batch
			X_mod = np.zeros((raw_batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
			
			with open(os.path.join(train_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind0, :] = X[sampleind, timeind, :]
						X_mod[sampleind, ind0, timeC] = 0 #set to zero
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind, :] = X[sampleind, timeind, :]
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True

						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				X_mod[~mask] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				# we also need more than one point in time, because we integrate
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mod = np.delete(X_mod, (samples_to_delete), axis=0)
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)

				#make assumptions about the label, harden
				Y_mod = np.sum(Y_mod, axis=1)/np.repeat(np.sum(Y_mod, axis=(1,2))[:,None], repeats=nclasses-badweather_labels.size, axis=1)
				

				trainbatchsizes.append(Y_mod.shape[0])
		ntrainsamples =sum(trainbatchsizes)
		testbatchsizes = []
		
		print('Scanning Testing data...')
		for filename in tqdm(os.listdir(test_localdir)):
			
			#starting a new batch
			X_mod = np.zeros((raw_batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
			
			with open(os.path.join(test_localdir, filename), "rb") as f:
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True
						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)
				
				testbatchsizes.append(Y_mod.shape[0])
		ntestsamples =sum(testbatchsizes)
		evalbatchsizes = []
				
		
		print('Scanning Evaluation data...')
		for filename in tqdm(os.listdir(eval_localdir)):
			
			#starting a new batch
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
			
			with open(os.path.join(eval_localdir, filename), "rb") as f:
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True
						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)
				
				evalbatchsizes.append(Y_mod.shape[0])
		nevalsamples =sum(evalbatchsizes)
		batchsizes = []
		
		ntargetclasses = nclasses-badweather_labels.size
		
		# Open a hdf5 files and create arrays
		hdf5_file_train = h5py.File(os.path.join(self.processed_folder, self.train_file) , mode='w')
		hdf5_file_train.create_dataset("data", (ntrainsamples, len(unique_times), nfeatures-1), np.float)
		hdf5_file_train.create_dataset("mask", (ntrainsamples, len(unique_times), nfeatures-1), np.bool)
		hdf5_file_train.create_dataset("labels", (ntrainsamples, ntargetclasses), np.float)
		
		hdf5_file_test = h5py.File(os.path.join(self.processed_folder, self.test_file) , mode='w')
		hdf5_file_test.create_dataset("data", (ntestsamples, len(unique_times), nfeatures-1), np.float)
		hdf5_file_test.create_dataset("mask", (ntestsamples, len(unique_times), nfeatures-1), np.bool)
		hdf5_file_test.create_dataset("labels", (ntestsamples, ntargetclasses), np.float)
		
		hdf5_file_eval = h5py.File(os.path.join(self.processed_folder, self.eval_file) , mode='w')
		hdf5_file_eval.create_dataset("data", (nevalsamples, len(unique_times), nfeatures-1), np.float)
		hdf5_file_eval.create_dataset("mask", (nevalsamples, len(unique_times), nfeatures-1), np.bool)
		hdf5_file_eval.create_dataset("labels", (nevalsamples, ntargetclasses), np.float)
		
		observed = 0
		missing = 0
		
		# prepare shuffeling of samples
		indices = np.arange(ntrainsamples)
		
		if self.shuffle:
			np.random.shuffle(indices)

		#Training data
		print("Building training dataset...")
		first_batch = True
		for fid, filename in enumerate(tqdm(os.listdir(train_localdir))): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((raw_batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(train_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind0, :] = X[sampleind, timeind, :]
						X_mod[sampleind, ind0, timeC] = 0 #set to zero
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind, :] = X[sampleind, timeind, :]
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True
						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				X_mod[~mask] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mod = np.delete(X_mod, (samples_to_delete), axis=0)
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)
				
				#make assumptions about the label, harden
				Y_mod = np.sum(Y_mod, axis=1)/np.repeat(np.sum(Y_mod, axis=(1,2))[:,None], repeats=nclasses-badweather_labels.size, axis=1)
				
				#for statistics
				missing += np.sum(mask == 0.)
				observed += np.sum(mask == 1.)
				
				valid_batchsize = X_mod.shape[0]
	
				#get the time stamps
				tt = unique_times
				
				if first_batch:
					start_ix = 0
					stop_ix = valid_batchsize
					first_batch = False
				else:
					start_ix = stop_ix
					stop_ix += valid_batchsize
					
				#fill in data to hdf5 file
				sorted_indices = np.sort(indices[start_ix:stop_ix])
				
				hdf5_file_train["data"][sorted_indices, ...] = X_mod
				hdf5_file_train["mask"][sorted_indices, ...] = X_mask_mod
				hdf5_file_train["labels"][sorted_indices, ...] = Y_mod
				
				#hdf5_file_train["data"][start_ix:stop_ix, ...] = X_mod
				#hdf5_file_train["mask"][start_ix:stop_ix, ...] = X_mask_mod
				#hdf5_file_train["labels"][start_ix:stop_ix, ...] = Y_mod
				
				
		
		#Testing data
		print("Building testing dataset...")
		first_batch = True
		for fid, filename in enumerate(tqdm(os.listdir(test_localdir))): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((raw_batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(test_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind0, :] = X[sampleind, timeind, :]
						X_mod[sampleind, ind0, timeC] = 0 #set to zero, last day of previous year
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind, :] = X[sampleind, timeind, :]
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True
						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				# "all masked out elements should be zeros"
				X_mod[~mask] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mod = np.delete(X_mod, (samples_to_delete), axis=0)
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)
				
				#make assumptions about the label
				Y_mod = np.sum(Y_mod, axis=1)/np.repeat(np.sum(Y_mod, axis=(1,2))[:,None], repeats=nclasses-badweather_labels.size, axis=1)
				
				#for statistics
				missing += np.sum(mask == 0.)
				observed += np.sum(mask == 1.)
				
				valid_batchsize = X_mod.shape[0]
	
				#get the time stamps
				tt = unique_times
				
				if first_batch:
					start_ix = 0
					stop_ix = valid_batchsize
					first_batch = False
				else:
					start_ix = stop_ix
					stop_ix += valid_batchsize
					
				#fill in data to hdf5 file
				hdf5_file_test["data"][start_ix:stop_ix, ...] = X_mod
				hdf5_file_test["mask"][start_ix:stop_ix, ...] = X_mask_mod
				hdf5_file_test["labels"][start_ix:stop_ix, ...] = Y_mod
				
		start_ix = 0
		stop_ix = 0
					
		
		#Evaluation data
		print("Building evaluation dataset...")
		first_batch = True
		for fid, filename in enumerate(tqdm(os.listdir(eval_localdir))): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((raw_batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((raw_batchsize, maxobs, nclasses))
			mask = np.zeros((raw_batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(eval_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f)
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				raw_batchsize, maxobs, nfeatures = X.shape
				_, _, nclasses = Y.shape
				times = X[:,:,timeC] #(500,26)
				
				#get the time ordering of time
				for ind, t in enumerate(unique_times):
					ind = ind
					if abs(t-1)<0.0001:
						#correct for the offset thing, where the first measurement is in the last year
						ind0=0
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind0, :] = X[sampleind, timeind, :]
						X_mod[sampleind, ind0, timeC] = 0 #set to zero
						Y_mod[sampleind, ind0, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, 0, :] = True
						
					elif abs(t)<0.0001: #no data => do nothing
						#print("was a 0")
						pass
					else:
						
						#Indices of corresponding times
						sampleind = np.nonzero(times==t)[0]
						timeind = np.nonzero(times==t)[1]
						
						#place at correct position
						X_mod[sampleind, ind, :] = X[sampleind, timeind, :]
						Y_mod[sampleind, ind, :] = Y[sampleind, timeind, :]
						
						#mark as observed in mask
						mask[sampleind, ind, :] = True
						
				# cloud/weather mask
				# 1 is observed, 0 is not observed, due to clouds/ice/snow and stuff
				# we mark the bad weather observations in the mask as unoberved
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_labels], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				# "all masked out elements should be zeros"
				X_mod[~mask] = 0
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_labels, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_labels.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#delete datapoints without any labels 
				#check that "mask" argument indeed contains a mask for data
				unobserved_datapt = np.where((np.sum(X_mask_mod==1., axis=(1,2)) == 0.)) #no data
				no_labels = np.where((np.sum(Y_mod, axis=(1,2)) == 0.)) #no labels
				too_few_obs_tp = np.where(np.sum(np.sum(X_mask_mod==1.,2)!=0, 1)<2)
				
				samples_to_delete = np.unique(np.hstack([ unobserved_datapt, no_labels, too_few_obs_tp] ))
				
				X_mod = np.delete(X_mod, (samples_to_delete), axis=0)
				X_mask_mod = np.delete(X_mask_mod, (samples_to_delete), axis=0)
				Y_mod = np.delete(Y_mod, (samples_to_delete), axis=0)
				
				#make assumptions about the label
				Y_mod = np.sum(Y_mod, axis=1)/np.repeat(np.sum(Y_mod, axis=(1,2))[:,None], repeats=nclasses-badweather_labels.size, axis=1)
				
				#for statistics
				missing += np.sum(mask == 0.)
				observed += np.sum(mask == 1.)
								
				valid_batchsize = X_mod.shape[0]
	
				#get the time stamps
				tt = unique_times
				
				if first_batch:
					start_ix = 0
					stop_ix = valid_batchsize
					first_batch = False
				else:
					start_ix = stop_ix
					stop_ix += valid_batchsize
					
				#fill in data to hdf5 file
				hdf5_file_eval["data"][start_ix:stop_ix, ...] = X_mod
				hdf5_file_eval["mask"][start_ix:stop_ix, ...] = X_mask_mod
				hdf5_file_eval["labels"][start_ix:stop_ix, ...] = Y_mod
				
		if self.normalize:
			
			print("Calculating mean and standard deviation of training dataset...")
			training_mean2 = np.ma.array(hdf5_file_train["data"][:], mask=~hdf5_file_train["mask"][:]).mean(axis=(0,1))
			training_std2 = np.ma.array(hdf5_file_train["data"][:], mask=~hdf5_file_train["mask"][:]).std(axis=(0,1),ddof=1)
			
			print("Normalizing data. This may take some time ...")
			#sorry for this large one-liner, but it's just normalization of the observed values
			hdf5_file_train["data"][:] = np.divide(  np.subtract(hdf5_file_train["data"], training_mean2, out=np.zeros_like(hdf5_file_train["data"][:]), where=hdf5_file_train["mask"][:])  ,  training_std2  , out=np.zeros_like(hdf5_file_train["data"][:]), where=hdf5_file_train["mask"][:])
			hdf5_file_test["data"][:] = np.divide(  np.subtract(hdf5_file_test["data"], training_mean2, out=np.zeros_like(hdf5_file_test["data"][:]), where=hdf5_file_test["mask"][:])  ,  training_std2  , out=np.zeros_like(hdf5_file_test["data"][:]), where=hdf5_file_test["mask"][:])
			hdf5_file_eval["data"][:] = np.divide(  np.subtract(hdf5_file_eval["data"], training_mean2, out=np.zeros_like(hdf5_file_eval["data"][:]), where=hdf5_file_eval["mask"][:])  ,  training_std2  , out=np.zeros_like(hdf5_file_eval["data"][:]), where=hdf5_file_eval["mask"][:])


		print("Preprocessing finished")
		
		hdf5_file_train.close()
		hdf5_file_test.close()
		hdf5_file_eval.close()
		
		missing_rate = missing/(observed+missing)
		print(missing_rate)
	
	def _check_exists(self):
		exist_train = os.path.exists(
				os.path.join(self.processed_folder, self.train_file)
				 )
		exist_test = os.path.exists(
				os.path.join(self.processed_folder, self.test_file)
				 )
		exist_eval = os.path.exists(
				os.path.join(self.processed_folder, self.eval_file)
				 )
		exist_time = os.path.exists(
				os.path.join(self.processed_folder, self.time_file)
				 )
		
		if not (exist_train and exist_test and exist_eval and exist_time):
			return False
		return True
	
	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')
	
	@property
	def time_file(self):
		return 'time.hdf5'
	
	@property
	def train_file(self):
		return 'train.hdf5'

	@property
	def test_file(self):
		return 'test.hdf5'
	
	@property
	def eval_file(self):
		return 'eval.hdf5'
	
	@property
	def get_label(self, record_id):
		return self.label_dict[record_id]
	
	@property
	def label_list(self):
		return self.label
	
	def __getitem__(self, index):
		
		#should accept indices and should output the datasamples, as read from disk
		if isinstance(index, slice):
			# do your handling for a slice object:
			output = []
			start = 0 if index.start is None else index.start
			step = 1 if index.start is None else index.step
			
			if self.list_form : #list format as the other datasets
				for i in range(start,index.stop,step):
					data = torch.from_numpy( self.hdf5dataloader["data"][i] )
					time_stamps = torch.from_numpy( self.timestamps )
					mask = torch.from_numpy(  self.hdf5dataloader["mask"][i] )
					labels = torch.from_numpy( self.hdf5dataloader["labels"][i] ) 
					output.append((data, time_stamps, mask, labels))
				return output

			else: #tensor_format (more efficient), 
				#raise Exception('Tensorformat not implemented yet!')
				
				data = torch.from_numpy( self.hdf5dataloader["data"][start:index.stop:step] ).float().to(self.device)
				time_stamps = torch.from_numpy( self.timestamps ).to(self.device)
				mask = torch.from_numpy(  self.hdf5dataloader["mask"][start:index.stop:step] ).float().to(self.device)
				labels = torch.from_numpy( self.hdf5dataloader["labels"][start:index.stop:step] ).float().to(self.device)
				
				#make it a dictionary to replace the collate function....
				data_dict = {
					"data": data[:,::self.step,:self.feature_trunc], 
					"time_steps": time_stamps[::self.step],
					"mask": mask[:,::self.step,:self.feature_trunc],
					"labels": labels}

				data_dict = utils.split_and_subsample_batch(data_dict, self.args, data_type = self.mode)
				
				return data_dict
				#return (data, time_stamps, mask, labels)
		else:
            # Do your handling for a plain index

			if self.second:
				raise Exception('Tensorformat not implemented yet!')
				self.second = True
			
			if self.list_form :
				data = torch.from_numpy( self.hdf5dataloader["data"][index] )
				time_stamps = torch.from_numpy( self.timestamps )
				mask = torch.from_numpy( self.hdf5dataloader["mask"][index] )
				labels = torch.from_numpy( self.hdf5dataloader["labels"][index] )
				return (data, time_stamps, mask, labels)
			else:
				data = torch.from_numpy( self.hdf5dataloader["data"][index] ).float().to(self.device)
				time_stamps = torch.from_numpy( self.timestamps ).to(self.device)
				mask = torch.from_numpy(self.hdf5dataloader["mask"][index] ).float().to(self.device)
				labels = torch.from_numpy( self.hdf5dataloader["labels"][index] ).float().to(self.device)

				data_dict = {
					"data": data, 
					"time_steps": time_stamps,
					"mask": mask,
					"labels": labels}

				data_dict = utils.split_and_subsample_batch(data_dict, self.args, data_type = self.mode)
				
				return data_dict
				#return (data, time_stamps, mask, labels)

	def __len__(self):
		if self.mode=="train":
			return min(self.args.n, self.hdf5dataloader["data"].shape[0])
		else:
			return min(self.args.validn, self.hdf5dataloader["data"].shape[0])
	
	def true_len__(self):
		if self.mode=="train":
			return self.hdf5dataloader["data"].shape[0]
		else:
			return self.hdf5dataloader["data"].shape[0]

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str
	
	
	
def variable_time_collate_fn_crop(batch, args, device = torch.device("cpu"), data_type="train", 
	data_min = None, data_max = None, list_form=True):
	"""
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	
	if list_form: #list format as the other datasets
		
		data, tt, mask, labels = batch[0]
		nfeatures = data.shape[1]
		N_labels = labels.shape[0]
		
		combined_vals = torch.zeros([len(batch), len(tt), nfeatures]).to(device)
		combined_mask = torch.zeros([len(batch), len(tt), nfeatures]).to(device)
		
		combined_labels = (torch.zeros([len(batch), N_labels])+ torch.tensor(float('nan'))).to(device)
		#combined_labels = (torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))).to(device = device)
		
		for b, (data, tt, mask, labels) in enumerate(batch):
			tt = tt.to(device)
			data = data.to(device)
			mask = mask.to(device)
			labels = labels.to(device)
			
			combined_vals[b] = data
			combined_mask[b] = mask
			
			combined_labels[b] = labels
		combined_tt = tt
		
	else: #tensor_format (more efficient), must agree with the __getitem__ function
		# Tensorformat	
		data, tt, mask, labels = batch
		
		combined_tt = tt
		combined_vals = data
		combined_mask = mask
		combined_labels = labels
	
	#combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
	#		att_min = data_min, att_max = data_max)
	
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
	
	return data_dict












