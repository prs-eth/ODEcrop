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

class Crops(object):
	
	# TODO: make label Tags
	labels = [ "other", "corn", "meadow", "asparagus", "rape", "hop", "summer oats", "winter spelt", "fallow", "winter wheat",
		   "winter barley", "winter rye", "beans", "winter triticale", "summer barley", "peas", "potatoe", "soybeans", "sugar beets" ]
	labels_dict = {k: i for i, k in enumerate(labels)}
	
	
	def __init__(self, root, download=False,
		reduce='average', mode='train', minseqlength=20,
		n_samples = None, device = torch.device("cpu")):
		
		self.root = root
		self.reduce = reduce
		self.mode = mode
		self.stored_batchsize = 100
				
		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if self.mode=="train":
			data_file = self.train_file
		elif self.mode=="eval":
			data_file = self.evaluation_file
		elif self.mode=="test":
			data_file = self.test_file
			
		"""
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, data_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, data_file))
		"""
		
		self.hdf5dataloader = h5py.File(os.path.join(self.processed_folder, data_file), "r")

		#Load one file and see how large the batchsize is
		self.raw_batchsize = self.data[0][2].shape[0]
		
		if n_samples is not None:
			self.data = self.data[:n_samples]
		
		
			
	def download(self):
		
		if self._check_exists():
			return
		
		#create the directories
		os.makedirs(self.processed_folder, exist_ok=True)
		
		print('Downloading data...')
		
		#get the dataset from the web
		#os.system('wget ftp://m1370728:m1370728@138.246.224.34/data.zip')
		#os.system('unzip data.zip -d ' + self.raw_folder)
		#os.system('rm data.zip')
		
		# test for completeness
		#os.system('wget ftp://m1370728:m1370728@138.246.224.34/data.sha512')
		#os.system('sha512sum -c data.sha512')
		
		#Processing data
		print('Processing data...')
		
		#collect all the possible time stamps
		train_localdir = os.path.join(self.raw_folder, 'data', 'train')
		test_localdir = os.path.join(self.raw_folder, 'data', 'test')
		evaluation_localdir = os.path.join(self.raw_folder, 'data', 'eval')
		first = True
		timeC = 0
		badweather_lables = np.array([0,1,2,3])
		
		unique_times = np.array([0])
		for filename in os.listdir(train_localdir):
			with open(os.path.join(train_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				X, Y, _ = u.load()
				if first:
					batchsize, maxobs, nfeatures = X.shape
					_, _, nclasses = Y.shape
					first=False
				unique_times = np.unique(np.hstack([ X[:,:,0].ravel(), unique_times] ))
		
		unique_times = np.array([0])
		for filename in os.listdir(test_localdir):
			with open(os.path.join(test_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				X, _, _ = u.load()
				unique_times = np.unique(np.hstack([ X[:,:,0].ravel(), unique_times] ))
				
		unique_times = np.array([0])
		for filename in os.listdir(evaluation_localdir):
			with open(os.path.join(evaluation_localdir, filename), "rb") as f:
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				X, _, _ = u.load()
				unique_times = np.unique(np.hstack([ X[:,:,timeC].ravel(), unique_times] ))
		
		self.timstamps = unique_times

		train_records = []
		test_records = []
		evaluation_records = []
		
		#Training data
		print("Building training dataset...")
		for filename in tqdm(os.listdir(train_localdir)): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((batchsize, maxobs, nclasses))
			mask = np.zeros((batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(train_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				batchsize, maxobs, nfeatures = X.shape
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
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_lables], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				X_mod[~mask] = None
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_lables, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_lables.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#get the time stamps
				tt = unique_times
				
				batchsize = X_mod.shape[0]
				for i in range(0,batchsize, self.stored_batchsize):
					# convert to torch variable and append it
					train_records.append((
								filename + "_" + str(i),
								torch.from_numpy(tt),
								torch.from_numpy(X_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(X_mask_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(Y_mod[i:i+self.stored_batchsize])
								))
		#save data
		print("Saving. This can take several minutes...")
		torch.save(train_records, os.path.join(self.processed_folder, self.train_file))
		print("Train data done!")
		
		#Testing data
		print("Building testing dataset...")
		for filename in tqdm(os.listdir(test_localdir)): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((batchsize, maxobs, nclasses))
			mask = np.zeros((batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(test_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				batchsize, maxobs, nfeatures = X.shape
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
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_lables], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				X_mod[~mask] = None
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_lables, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_lables.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#get the time stamps
				tt = unique_times
				
				batchsize = X_mod.shape[0]
				for i in range(0,batchsize, self.stored_batchsize):
					# convert to torch variable and append it
					test_records.append((
								filename + "_" + str(i),
								torch.from_numpy(tt),
								torch.from_numpy(X_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(X_mask_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(Y_mod[i:i+self.stored_batchsize])
								))
					
		#save data
		print("Saving...")
		torch.save(test_records, os.path.join(self.processed_folder, self.test_file))
		print("Test data done!")
		
		
		
		
		#Evaluation data
		print("Building evaluation dataset...")
		for filename in tqdm(os.listdir(evaluation_localdir)): #tqdm
	
			#starting a new batch
			X_mod = np.zeros((batchsize, maxobs, nfeatures))
			Y_mod = np.zeros((batchsize, maxobs, nclasses))
			mask = np.zeros((batchsize, maxobs, nfeatures),dtype=bool)
		
			with open(os.path.join(evaluation_localdir, filename), "rb") as f:
				
				#Unpacking procedure with pickels
				u = pickle._Unpickler(f);
				u.encoding = 'latin1'
				data = u.load()
				X, Y, obslen = data
				
				batchsize, maxobs, nfeatures = X.shape
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
				badweather_obs = np.nonzero(np.sum(Y_mod[:,:,badweather_lables], axis=2)!=0)
				mask[badweather_obs[0], badweather_obs[1], :] = 0
				
				#"destroy" data, that is corrputed by bad weather. We will never use it!
				X_mod[~mask] = None
				
				#Truncate the timestamp-column (timeC) from the features and mask
				X_mod = np.delete(X_mod, (timeC), axis=2)
				X_mask_mod = np.delete(mask, (timeC), axis=2)
				
				#truncate and renormalize the labels
				Y_mod = np.delete(Y_mod, badweather_lables, axis=2)
				tot_weight = np.repeat(np.sum(Y_mod, axis=2)[:,:,None], repeats=nclasses-badweather_lables.size, axis=2)
				Y_mod = np.divide(Y_mod, tot_weight, out=np.zeros_like(Y_mod), where=tot_weight!=0)
				
				#get the time stamps
				tt = unique_times
				
				batchsize = X_mod.shape[0]
				for i in range(0,batchsize, self.stored_batchsize):
					# convert to torch variable and append itappend it
					evaluation_records.append((
								filename + "_" + str(i),
								torch.from_numpy(tt),
								torch.from_numpy(X_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(X_mask_mod[i:i+self.stored_batchsize]),
								torch.from_numpy(Y_mod[i:i+self.stored_batchsize])
								))
					
		
		#save data
		print("Saving...")
		torch.save(evaluation_records, os.path.join(self.processed_folder, self.evaluation_file))
		print("Evaluation data done!")
		
		#self.nfeatures = nfeatures
		self.nclasses = nclasses
	
	def _check_exists(self):
		exist_train = os.path.exists(
				os.path.join(self.processed_folder, self.train_file)
				 )
		exist_test = os.path.exists(
				os.path.join(self.processed_folder, self.test_file)
				 )
		exist_eval = os.path.exists(
				os.path.join(self.processed_folder, self.train_file)
				 )
		
		if not (exist_train and exist_test and exist_eval):
			return False
		return True
	
	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')
	
	@property
	def data_file(self):
		return 'data.pt'
	
	@property
	def train_file(self):
		return 'train.hdf5'

	@property
	def test_file(self):
		return 'test.hdf5'
	
	@property
	def evaluation_file(self):
		return 'evaluation.hdf5'
	
	def __getitem__(self, index):
		#TODO: should accept indices and should output the datasamples, as read from disk
		return self.hdf5dataloader[index]

	def __len__(self):
		# TODO: should output the number of samples
		return len(self.data*self.stored_batchsize)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str
	
	
	
def variable_time_collate_fn_crop(batch, args, device = torch.device("cpu"), data_type="train"):
	#TODO
	return None












