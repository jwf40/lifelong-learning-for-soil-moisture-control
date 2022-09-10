import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import json
import random
import pandas as pd
class EWCData(Dataset):
    #https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231
    def __init__(self, data_config_path, labels_col,seq_len,transform=None, target_transform=None, pred=False, is_recording=False):
        self.is_recording = is_recording
        self.pred = pred
        self.seq_len = seq_len
        with open(data_config_path, 'r') as f:
            self.data_config = json.load(f)
        self.data = pd.read_csv(self.data_config['data_dir'])
        
        self.data = self.data.tail(self.seq_len) if self.pred else self.data.tail(self.seq_len+1)
        self.targets = self.data[labels_col]
        self.lstm_targets = self.data['lstm_sm']

        for idx,each in self.targets.iteritems():
            if each < self.data_config['min_moisture']:
                self.targets[idx] = 0
            elif each >= self.data_config['max_moisture']:
                self.targets[idx] = 1
            else:
                self.targets[idx] = -1
        
        for idx,each in self.lstm_targets.iteritems():
            if each < self.data_config['min_moisture']:
                self.lstm_targets[idx] = 0
            elif each >= self.data_config['max_moisture']:
                self.lstm_targets[idx] = 1
            else:
                self.lstm_targets[idx] = -1

            #self.data = self.data.drop(self.data.columns[[0,1,2,8]], axis=1)
        self.required_fields = self.data_config['required_fields'].split(',')
        self.data = self.data[self.required_fields]

        #Normalize
        self.data = (self.data-self.data.mean())/self.data.std()
        #self.targets = (self.targets-self.targets.mean())/self.targets.std()
        
        self.data.reset_index(drop=True, inplace=True)
        self.targets.reset_index(drop=True, inplace=True)
        self.targets = self.targets.astype('int64')
        self.lstm_targets.reset_index(drop=True, inplace=True)
        self.lstm_targets = self.targets.astype('int64')
        #print(self.targets.head())

        self.transform = transform
        self.target_transform = transform
        self.num_features = len(self.data.columns)       

    def __len__(self):
        return (1+len(self.targets))-self.seq_len if self.pred else len(self.targets)-self.seq_len
    
    def get_data(self, idx):
        data = torch.Tensor(self.data.iloc[idx:idx+self.seq_len, :].to_numpy())
        label = -200 if self.pred else self.targets[idx+self.seq_len]
        if self.is_recording:
            ewc_ = label 
            lstm_ = -200 if self.pred else self.lstm_targets[idx+self.seq_len]
            label = [ewc_, lstm_]
        return data, label
       

    def __getitem__(self, idx):
        data, label = self.get_data(idx)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data.float(), label    
