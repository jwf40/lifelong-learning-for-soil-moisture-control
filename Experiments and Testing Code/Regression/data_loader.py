import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
class WeatherDataset(Dataset):
    def __init__(self, file_dir, labels_col, transform=None, target_transform=None):
        self.data = pd.read_csv(file_dir)
        self.targets = self.data[labels_col]
        self.data = self.data.drop(labels_col, axis=1)
        self.data = self.data.drop(self.data.columns[[0,1,2]], axis=1)
        self.data = (self.data-self.data.mean())/self.data.std()
        #self.targets = (self.targets-self.targets.mean())/self.targets.std()
        
        # self.data = self.data.drop(self.data.index[2:])
        # self.targets = self.targets.drop(self.targets.index[2:])
        self.data.reset_index(drop=True, inplace=True)
        self.targets.reset_index(drop=True, inplace=True)
        print(self.data.head())

        self.transform = transform
        self.target_transform = transform
        self.num_features = len(self.data.columns)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.data.iloc[idx, :])
        sum_ = sum(data)
        #data = torch.Tensor((data-data.mean())/data.std())
        
        label = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data.float(), label