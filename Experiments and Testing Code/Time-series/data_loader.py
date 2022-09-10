import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import random
import pandas as pd
class WeatherDataset(Dataset):
    def __init__(self, file_dir, labels_col, seq_len,transform=None, target_transform=None):
        self.seq_len = seq_len

        self.data = pd.read_csv(file_dir)
        self.targets = self.data[labels_col]
        
        for idx,each in self.targets.iteritems():
            if each <=2.4:
                self.targets[idx] = 0
            elif each >=2.5:
                self.targets[idx] = 1
            else:
                self.targets[idx] = 1

        
        self.data = self.data.drop(labels_col, axis=1)
        self.data = self.data.drop(self.data.columns[[0,1,2,7,8]], axis=1)
        #Normalize
        self.data = (self.data-self.data.mean())/self.data.std()
        #self.targets = (self.targets-self.targets.mean())/self.targets.std()
        
        self.data.reset_index(drop=True, inplace=True)
        self.targets.reset_index(drop=True, inplace=True)
        self.targets = self.targets.astype('int64')

        self.transform = transform
        self.target_transform = transform
        self.num_features = len(self.data.columns)

    
    def __len__(self):
        return len(self.targets)-self.seq_len
    
    def get_data(self, idx):
        data = torch.Tensor(self.data.iloc[idx:idx+self.seq_len, :].to_numpy())
        label = self.targets[idx+self.seq_len-1]
        return data, label

    def __getitem__(self, idx):
        data, label = self.get_data(idx)
        
        if label == 0:
            rand = random.randrange(10)
            if rand<2:
                idx = random.randrange(self.__len__())
                data, label = self.get_data(idx)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data.float(), label