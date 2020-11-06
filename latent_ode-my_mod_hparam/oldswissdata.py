


class Dataset_old(torch.utils.data.Dataset):
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
															'Hedge', 'Apples', 'Soy', 'Fallow', 'Peas', 'Berries', 'Oat', 'FieldBean', 'EinkornWheat', 'Rye', 'TreeCrop', 'SummerWheat']:
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
				cloud_cover = cloud_cover.flip(2)
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

	train_dataset_obj = SwissCrops('data/SwissCrops', mode="train", datatype="2_14toplabels")
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
	