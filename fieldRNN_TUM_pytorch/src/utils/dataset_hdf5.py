import torch.utils.data
import os
import rasterio
import torch
import numpy as np
#import torch.nn.functional as F
import h5py
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, t=0.9):
        
        self.data = h5py.File(path, "r")
        self.samples = self.data["data"].shape[0]
        self.max_obs = self.data["data"].shape[1]
        self.spatial = self.data["data"].shape[2:-1]
        self.n_classes = np.max( self.data["gt"] ) + 1
        self.t = t
        self.augment_rate = 0.5
        
        #Get rid of tiles with too much background
        self.valid_list = self.get_rid_small_fg_tiles()
        self.valid_samples = self.valid_list.shape[0]
        
        print('Dataset size: ', self.samples)
        print('Valid dataset size: ', self.valid_samples)
        print('Sequence length: ', self.max_obs)
        print('Spatial size: ', self.spatial)
        print('Number of classes: ', self.n_classes)
        
    def __len__(self):
        return self.valid_samples

    def __getitem__(self, idx):
                     
        idx = self.valid_list[idx]
        X = self.data["data"][idx]
        target = self.data["gt"][idx,...,0]
        #gt_instance = self.data["gt_instance"][idx]

        X = np.transpose(X, (0, 3, 1, 2))
        
        X = X[0::4,...]
        
       # augmentation
#        if np.random.rand() < self.augment_rate:
#            X = np.fliplr(X)
#            target = np.fliplr(target)
#        if np.random.rand() < self.augment_rate:
#            X = np.flipud(X)
#            target = np.flipud(target)
#        if np.random.rand() < self.augment_rate:
#            angle = np.random.choice([1, 2, 3])
#            X = np.rot90(X, angle, axes=(2, 3))
#            target = np.rot90(target, angle, axes=(2, 3))

        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()

        #X = F.interpolate(X, size=[48,48])
        #target = F.interpolate(target.view(1,1,target.shape[0],target.shape[1]), size=[48,48])
        #target = target.squeeze()
        #print(X.shape, target.shape)
        
        #keep values between 0-1
        X = X * 1e-4
            
        return X.float(), target.long()



    def get_rid_small_fg_tiles(self):
        valid = np.ones(self.samples)
        w,h = self.data["gt"][0,...,0].shape
        for i in range(self.samples):
            if np.sum( self.data["gt"][i,...,0] != 0 )/(w*h) < self.t:
                valid[i] = 0
        
        return np.nonzero(valid)[0]
        
    
    
if __name__=="__main__":

    dataset = Dataset("/scratch/tmehmet/TG_train_2_patches_24x24_instance.hdf5")
    #X = dataset[:][1]
    ##print(np.unique(X))
    
    for i in range(8000):
        #print(i)
        X = dataset[i][1]
        target = dataset[i][1]
        if torch.max(target)>107:
            print(torch.max(target), torch.min(target))
            
        if  torch.sum(torch.isnan(X)) > 0 :
            print(X)
        