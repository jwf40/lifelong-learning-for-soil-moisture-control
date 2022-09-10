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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features1, hidden_features2, out_features, batch_size, dropout=0.0):
        super(MLP, self).__init__()
        self.batch_size = batch_size
        self.dropout=nn.Dropout(dropout)
        self.hidden1 = nn.Linear(in_features, hidden_features1)
        self.relu = nn.LeakyReLU()
        self.hidden2 = nn.Linear(hidden_features1, hidden_features2)
        self.output = nn.Linear(hidden_features2, out_features)

    def forward(self,x):
        
        x_batch = x[:,1].size()[0]
        #print(x_batch)
        x = x.view(x_batch, -1)        
        out = self.dropout(x)
        out = self.hidden1(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.hidden2(out)
        out = self.relu(out)

        out = self.output(out)
        print(out.size())
        return out.float()