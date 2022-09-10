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
    def __init__(self, in_features, hidden_features, num_classes,batch_first=False, bidirectional=False):
        super(LSTM_FF, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_features, 2, batch_first=batch_first, bidirectional=bidirectional)
        fc_in = hidden_features*2 if bidirectional else hidden_features
        self.fc1 = nn.Linear(fc_in, num_classes)
        self.relu = nn.ReLU()
    def forward(self,x):
        x_batch = x[:,0].size()[0]
        x = x.view(x_batch,1,-1)
        x, [h,c]= self.lstm(x)
        x = self.fc1(self.relu(x))
        x = x[:, -1]
        return x.float()