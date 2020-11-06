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
        self.samples = self.data["X_test"].shape[0]
        self.max_obs = self.data["X_test"].shape[1]
        self.spatial = self.data["X_test"].shape[2:-1]
        #self.n_classes = np.max( self.data["y_train"] ) + 1
        self.t = t
        self.n_classes = 106
        
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
        #X = self.data["data"][idx]
        #target = self.data["gt"][idx,...,0]
        #gt_instance = self.data["gt_instance"][idx]

        X = self.data["X_test"][idx]
        #target = self.data["y_train"][idx,...,0]
        #print(target.shape)
        target = self.data["y_test"][idx]
        target = np.argmax(target, axis=-1)

        X = np.transpose(X, (0, 3, 1, 2))
        
        #X = X[0::4,...]
        
        X = torch.from_numpy(X)
        target = torch.from_numpy(target).float()

        #X = F.interpolate(X, size=[48,48])
        #target = F.interpolate(target.view(1,1,target.shape[0],target.shape[1]), size=[48,48])
        #target = target.squeeze()
        #print(X.shape, target.shape)

        #print(target)
            
        return X.float(), target.long()


    def get_rid_small_fg_tiles(self):
        valid = np.ones(self.samples)
        #w,h = self.data["y_train"][0,...,0].shape
        w,h=10,10
        for i in range(self.samples):
            if np.sum( np.argmax(self.data["y_test"][i,...], axis=-1) != 105 )/(w*h) < self.t:
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
        