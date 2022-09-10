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

from data_loader import WeatherDataset
from models.mlp import MLP
from models.cnn import CNN
from models.lstm import LSTM
from models.lstm_ff import LSTM_FF
from models.conv_lstm import Conv_LSTM

if __name__ == '__main__':
    batch_size = 43 #12
    #test_batch_size = 1 #12
    #val_batch_size = 1 #12
    num_epochs = 40

    train_data_root = 'data/PARSED_NW_Ground_Stations_2016.csv'
    test_data_root = 'data/PARSED_NW_Ground_Stations_2017.csv'
    label_name = 'daily_precip'

    workers = 2

    train_dataset = WeatherDataset(train_data_root, label_name)
    print(len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    test_dataset = WeatherDataset(test_data_root, label_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    #data, labels = next(iter(train_loader))

    #print(data,labels)

    num_gpus =1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus >0) else "cpu")

    
    num_train_samples = len(train_dataset.targets)
    batch_per_epoch = num_train_samples/batch_size
    num_classes = 1
    manual_seed = 555
    #random.seed(manual_seed)
    #torch.manual_seed(manual_seed)

    lr = 0.001
    net = LSTM(6,1)#LSTM_FF(6,12,batch_first=True, bidirectional=True)#Conv_LSTM(6,1)#
    
    net.to(device)
    optimiser = optim.Adam(net.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    acc_over_time = []
    loss_over_time = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs, labels = data[0].float().to(device),data[1].float().to(device)
            inputs = inputs.unsqueeze(1)
            
            #print(inputs)
            #print(labels)
            #print(inputs.size())
            optimiser.zero_grad()
            outputs = net(inputs.float())
            outputs=outputs.squeeze(1).squeeze(1)
            #outputs = outputs.reshape(outputs.shape[0])
            #print(outputs[:2], labels[:2])
            #print(outputs.shape)
            loss = torch.sqrt(loss_func(outputs, labels))
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            #if i % 1000 == 0:
            #    print(outputs[0:5], labels[0:5])
            

        print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per_epoch))
        loss_over_time.append(running_loss/batch_per_epoch)
        running_loss = 0.0

        if (epoch+1)%1==0:
            with torch.no_grad():
                running_loss = 0.0
                for i,data in enumerate(test_loader):
                    inputs, labels = data[0].float().to(device),data[1].float().to(device)
                    inputs = inputs.unsqueeze(1)
                    
                    #print(inputs.size())
                    optimiser.zero_grad()
                    outputs = net(inputs.float())
                    #outputs = outputs.reshape(outputs.shape[0])
                    outputs=outputs.squeeze(1).squeeze(1)

                    loss = torch.sqrt(loss_func(outputs.float(), labels.float()))
                    running_loss += loss.item()
            acc_over_time.append(running_loss/batch_per_epoch)
            print('[%d, %5d] TEST loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / batch_per_epoch))
        
    torch.save(acc_over_time, 'lstm_acc_over_time')
    torch.save(loss_over_time, 'lstm_loss_over_time')