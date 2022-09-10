import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import pandas as pd
class WeatherDataset(Dataset):
    def __init__(self, file_dir, labels_col, transform=None, target_transform=None):
        self.data = pd.read_excel(file_dir,sheet_name=1, header=5)
        self.targets = self.data.iloc[self.data[labels_col]]     
        self.data = self.data.drop([0,1,5,12,14])
        self.data = (self.data-self.data.mean())/self.data.std())
        self.transform = transform
        self.target_transform = transform
        self.num_features = len(self.data.columns)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        data = torch.Tensor(self.data.iloc[idx, :])
        data = 
        
        label = self.targets[idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data.float(), label