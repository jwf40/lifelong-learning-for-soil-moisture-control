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

class LSTM(nn.Module):
    def __init__(self, in_features, hidden_features, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_features, 2, batch_first=batch_first, bidirectional=bidirectional)
        
    
    def forward(self,x):
        x, [h,c]= self.lstm(x)
        return x