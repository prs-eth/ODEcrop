#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:28:46 2020

@author: metzgern
"""

import os
import torch




class Crops(object):
	
	def __init__(self, root, download=False,
		reduce='average', train=True,
		n_samples = None, device = torch.device("cpu")):
		
		self.root = root
		self.reduce = reduce
		self.train = train

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if self.train:
			data_file = self.training_file
		else:
			data_file = self.test_file
			
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]
			
			
	def download(self):
		
		if self._check_exists():
			return
		
		#create the directories
		os.makedirs(self.processed_folder, exist_ok=True)
		
		
		print('Processing data...')
		
		
		#get the dataset from the web
		os.system('wget ftp://m1370728:m1370728@138.246.224.34/data.zip')
		os.system('unzip data.zip -d ' + self.processed_folder)
		os.system('rm data.zip')
		
		# test for completeness
		"""
		os.system('wget ftp://m1370728:m1370728@138.246.224.34/data.sha512')
		os.system('cd ' + processed_folder)
		os.system('sha512sum -c data.sha512')
		os.system('cd ..')
		os.system('cd ..')
		"""
		os.makedirs(os.path.join(self.processed_folder, 'Done'))
		print('Done!')
	
	
	def _check_exists(self):
		if not os.path.exists(
			os.path.join(self.processed_folder, 'Done') 
			):
			return False
		return True
		
	@property
	def processed_folder(self):
		return os.path.join(self.root)

	@property
	def training_files(self):
		return 'data/train'

	@property
	def test_files(self):
		return 'data/test'
	
	@property
	def eval_files(self):
		return 'data/eval'
	
	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		fmt_str += '    Max length: {}\n'.format(self.max_seq_length)
		fmt_str += '    Reduce: {}\n'.format(self.reduce)
		return fmt_str
	
	
	
def variable_time_collate_fn_crop(batch, args, device = torch.device("cpu"), data_type="train"):
	#TODO
	return None












