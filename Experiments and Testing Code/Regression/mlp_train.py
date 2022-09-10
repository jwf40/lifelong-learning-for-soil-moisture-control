import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import random 

from data_loader import WeatherDataset
from models.mlp import MLP

if __name__ == '__main__':
    batch_size = 3 #12
    #test_batch_size = 1 #12
    #val_batch_size = 1 #12
    num_epochs = 40

    train_data_root = 'data/PARSED_NW_Ground_Stations_2016.csv'
    test_data_root = 'data/PARSED_NW_Ground_Stations_2017.csv'
    label_name = 'daily_precip'

    workers = 2

    train_dataset = WeatherDataset(train_data_root, label_name)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    test_dataset = WeatherDataset(test_data_root, label_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    #data, labels = next(iter(train_loader))

    num_gpus =1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus >0) else "cpu")

    
    num_train_samples = len(train_dataset.targets)
    batch_per_epoch = num_train_samples/batch_size
    num_classes = 1
    manual_seed = 555
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    lr = 0.0001
    momentum = 0.9
    in_features = train_dataset.num_features
    hidden_features1, hidden_features2 = int(in_features*4), int(in_features*8)#
    dropout_probability=0.0

    net = MLP(in_features, hidden_features1,hidden_features2,num_classes, batch_size, dropout=dropout_probability)
    net.to(device)
    optimiser = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    loss_func = nn.MSELoss()
    acc_over_time  = []
    loss_over_time = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs, labels = data[0].to(device),data[1].to(device)
            labels.unsqueeze_(1)
            #print(inputs)
            #print(labels)
            #print(inputs.size())
            optimiser.zero_grad()
            outputs = net(inputs)
            
            #outputs = outputs.reshape(outputs.shape[0])
            #print(outputs[:5], labels[:5])
            
            loss = torch.sqrt(loss_func(outputs, labels.float()))
            loss.backward()
            optimiser.step()
            #print(outputs, labels, loss.item())
            running_loss += loss.item()
        loss_over_time.append(running_loss/batch_per_epoch)
        print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per_epoch))
        if ((epoch+1) % 1) == 0:
            with torch.no_grad():
                running_loss = 0.0
                for i,data in enumerate(test_loader):
                    inputs, labels = data[0].float().to(device),data[1].float().to(device)
                    #print(inputs.size())
                    optimiser.zero_grad()
                    outputs = net(inputs.float())
                    outputs = outputs.reshape(outputs.shape[0])
                    
                    loss = torch.sqrt(loss_func(outputs.float(), labels.float()))
                    running_loss += loss.item()
            acc_over_time.append(running_loss/batch_per_epoch)
            print('[%d, %5d] TEST loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per_epoch))
    torch.save(acc_over_time, 'mlp_acc_over_time')
    torch.save(loss_over_time, 'mlp_loss_over_time')