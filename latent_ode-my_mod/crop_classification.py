#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:28:46 2020

@author: metzgern
"""

import os




class Crops(object):
	
	
	def __init__(self, root, download=False,
		reduce='average', max_seq_length = 50,
		n_samples = None, device = torch.device("cpu")):
		
		self.root = root
		self.reduce = reduce
		self.max_seq_length = max_seq_length

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. You can use download=True to download it')
		
		if device == torch.device("cpu"):
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
		else:
			self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

		if n_samples is not None:
			self.data = self.data[:n_samples]
			
			
		
	def download():
		if self._check_exists():
			return
		
		#TODO: Download the dataset
		
	
	
	
	
	
	def _check_exists(self):
		
		#TODO: Adjust to our dataset....
		
		
		for url in self.urls:
			filename = url.rpartition('/')[2]
			if not os.path.exists(
				os.path.join(self.processed_folder, 'data.pt')
			):
				return False
		return True
		

	@property
	def raw_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.__class__.__name__, 'processed')



