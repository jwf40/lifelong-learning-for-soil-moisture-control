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
  def __init__(self, inchan, outchan, kernel=[3,2], stride=1,pool=2, dropout=0.0):
    super(ConvBlock, self).__init__()
    self.conv1 = nn.Conv2d(inchan, outchan, kernel_size=kernel)
    self.pool = nn.MaxPool2d(pool)
    self.relu = nn.ReLU(inplace=True)
    self.dropout=nn.Dropout(dropout)
  
  def forward(self, x):
    out = self.dropout(x)
    out = self.conv1(out)
    #out = self.pool(out)
    out = self.relu(out)
    return out


class CNN(nn.Module):
    def __init__(self, num_channels,num_classes, dropout_rate=0.0):
        super(CNN, self).__init__()
        self.num_channels = num_channels
        self.block1 = ConvBlock(num_channels, num_channels*2, kernel=[2,4],dropout=dropout_rate)
        self.block2 = ConvBlock(num_channels*2, num_channels*3, kernel=[2,1],dropout=dropout_rate)
        self.block3 = ConvBlock(num_channels*3,num_channels*4, kernel=[2,1], dropout=dropout_rate)
        self.block4 = ConvBlock(num_channels*4, num_channels*5, kernel=[2,1], dropout=dropout_rate)
        # self.block5 = ConvBlock(num_channels*5, num_channels*6, kernel=2, dropout=dropout_rate)
        # self.block6 = ConvBlock(num_channels*6, num_channels*8, kernel=2, dropout=dropout_rate)
        self.fc1 = nn.Linear(num_channels*15, num_classes)
        self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(num_channels*4, num_classes)

    def forward(self,x):
        # try:
        #   x_batch = x[:,0].size()[0]
        # except:
        #   print(x.size())
        # x = x.view(x_batch, 1, -1)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        #print(out.size())
        # out = self.block5(out)
        # out = self.block6(out)

        #out += x #Residual Connection

        #Need to flatten each input for linear layers, but maintain batch size.
        out = out.view(-1, out.numel())
        #print(out.shape)
        
        #out = self.relu(self.fc1(out))
        out = self.fc1(out)        
        #print(out.shape)
        return out