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

class ConvBlock(nn.Module):
  def __init__(self, inchan, outchan, kernel=3, stride=1,pool=2, dropout=0.0):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv1d(inchan, outchan, kernel_size=kernel)
    self.pool = nn.MaxPool1d(pool)
    self.relu = nn.ReLU(inplace=True)
    self.dropout=nn.Dropout(dropout)
  
  def forward(self, x):
    out = self.dropout(x)
    out = self.conv1(out)
    #out = self.pool(out)
    out = self.relu(out)
    return out


class Conv_LSTM(nn.Module):
    def __init__(self, in_features, hidden_features, num_channels=1, batch_first=False, bidirectional=False):
        super(Conv_LSTM, self).__init__()
        self.block1 = ConvBlock(num_channels, num_channels*4, kernel=6)

        self.lstm = nn.LSTM(num_channels*4, hidden_features, 2, batch_first=True, bidirectional=False)
        
        fc_in = hidden_features*2 if bidirectional else hidden_features
        self.fc1 = nn.Linear(fc_in, 1)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        try:
          x_batch = x[:,0,0].size()[0]
        except:
          print(x.size())
        x = self.block1(x)
        x = x.view(x_batch,1,4)
        x, [h,c]= self.lstm(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x