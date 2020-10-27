import torch.utils.data
import os, glob
import rasterio
import torch
import numpy as np
#import torch.nn.functional as F
import h5py
import torch.nn.functional as F
from matplotlib import pyplot as plt 

class Dataset_eval(torch.utils.data.Dataset):
    def __init__(self, path, t=0.9, mode='all'):
        self.path = path
        self.data = h5py.File(path, "r")
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        self.t = t
        #self.augment_rate = 0.5
        
        #Get rid of tiles with too much background
        self.valid_list = self.split(mode)
        self.valid_samples = self.valid_list.shape[0]
        
        #self.dates = self.chooose_dates()[0].tolist()
        #self.dates = self.chooose_dates_2().tolist()
        #self.max_obs = len(self.dates)
        #self.max_obs = 55
        
        self.label_list = [0.,2,7,10,15,16,18,21,23,34,58,60,62,63,64]
        self.n_classes = len(self.label_list)
        
        
        print('Dataset size: ', self.samples)
        print('Valid dataset size: ', self.valid_samples)
        print('Sequence length: ', self.max_obs)
        print('Sequence length used: ', self.max_obs)
        print('Spatial size: ', self.spatial)
        print('Number of classes: ', self.n_classes)
        
        
        
    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
                     
        idx = self.valid_list[idx]
        X = self.data["data"][idx]
        target_ = self.data["gt"][idx,...,0]
        gt_instance = self.data["gt_instance"][idx,...,0]

        X = np.transpose(X, (0, 3, 1, 2))
 
        #Because 2018 data does not have these frames
        if self.max_obs != 110:
            X = X[34:,...]
            
        X = X[0::1,:,...]
        #X = X[self.dates,...] 
        
        #Change labels 
        target = np.zeros_like(target_)
        for i in range(len(self.label_list)):
            target[target_ == self.label_list[i]] = i


        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()
        gt_instance = torch.from_numpy(gt_instance).float()

        
        #keep values between 0-1
        X = X * 1e-4
            
        return X.float(), target.long(), gt_instance.long()



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
            valid[int(self.samples*0.85):] = 1.
        elif mode=='train':
            valid[:int(self.samples*0.85)] = 1.
        else:
            valid[:] = 1.

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
        bins = np.arange(self.n_classes+1)
        hist = np.histogram(gt_set, bins=bins)[0]

        #Normalize 
        hist = hist/np.sum(hist)
        #sort histogram 
        sorted_classes = np.argsort(hist)
        hist_sorted = hist[sorted_classes.astype(int)]
        print(sorted_classes)
        plt.bar(bins[:-1],np.flip(hist_sorted))
        #plt.plot(hist,'r')
        plt.title("density") 
        plt.savefig('train_gt_density_bar.png')
        plt.close()


if __name__=="__main__":

    traindataset = Dataset("/scratch/tmehmet/ZH_train_patches_24x24_instance.hdf5", 0.5,'all')
    #traindataset = Dataset("/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/ZH_train_patches_24x24_instance.hdf5", 0., 'all')
    
    traindataset.data_stat()
    
#    #Dates 
#    idxes = traindataset.chooose_dates()[0]
#    data_dir = '/home/pf/pfstaff/projects/ozgur_deep_filed/data_crop_CH/train_set_24x24/'
#    DATA_YEAR = '2019'
#    date_list = []
#    batch_dirs = os.listdir(data_dir)
#    for batch_count, batch in enumerate(batch_dirs):
#        for filename in glob.iglob(data_dir + batch + '/**/patches_res_R10m.npz', recursive=True):
#                date = filename.find(DATA_YEAR)
#                date = filename[date:date+8]
#                if date not in date_list:
#                    date_list.append(date) 
#    
#    print("Num dates:", idxes.shape[0])
#    date_list.sort()
#    for i in range(idxes.shape[0]):
#        print(date_list[idxes[i]])
    
    
    
    
    
    
