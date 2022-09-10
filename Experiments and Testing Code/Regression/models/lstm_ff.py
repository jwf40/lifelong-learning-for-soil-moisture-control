import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

class LSTM_FF(nn.Module):
    def __init__(self, in_features, hidden_features, batch_first=False, bidirectional=False):
        super(LSTM_FF, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_features, 2, batch_first=True, bidirectional=True)
        fc_in = hidden_features*2 if bidirectional else hidden_features
        self.fc1 = nn.Linear(fc_in, 1)
    
    def forward(self,x):
        x, [h,c]= self.lstm(x)
        x = self.fc1(x)
        return x