import datetime 
import pickle
import matplotlib.pyplot as plt
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

class ModelTrainer():
    def __init__(self,net, lr, num_classes, loss_weights,batch_size, num_epochs, train_data_root, test_data_root, label_name, seq_len,name=None):
        self.net_folder = 'final_runs/'
        self.results_folder = 'final_runs/'
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        workers = 2
        self.train_dataset = WeatherDataset(train_data_root, label_name, seq_len)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
        self.test_dataset = WeatherDataset(test_data_root, label_name, seq_len)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.num_train_samples = len(self.train_dataset.targets)
        self.batch_per_epoch = self.num_train_samples/batch_size
        self.num_classes = num_classes

        self.lr = lr
        self.net = net.to(self.device)

        self.optimiser = optim.Adam(self.net.parameters(), lr=lr)
        self.weights = loss_weights.to(self.device)
        self.loss_func = nn.CrossEntropyLoss(weight=self.weights)

        time = datetime.datetime.now()
        strtime = time.strftime('%H-%M-%S_%Y-%m-%d')
        self.name = 'lstm_state_dict_' + strtime
        if name != None:
            self.name += name

    def save_data(self, name, data):
        name = self.results_folder + name
        with open(name, 'wb') as f:
            pickle.dump(data, f)

    def save_net(self, epoch):                
        name = self.net_folder + self.name + '_' + str(epoch)
        torch.save(self.net.state_dict(), name)

    def train_step(self):
        running_loss = 0.0
        for i,data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device),data[1].to(self.device)
            #labels.unsqueeze_(1)
            #inputs.unsqueeze_(0)
            #print(inputs.size())
            self.optimiser.zero_grad()
            outputs = self.net(inputs)
            #outputs.squeeze_(1)
            #outputs = outputs.reshape(outputs.shape[0])
            #print(outputs[:5], labels[:5])
            outputs.squeeze_(1)
            loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimiser.step()
            #print(outputs, labels, loss.item())
            running_loss += loss.item()
        return running_loss
    
    def train_loop(self):
        for run in range(5):
            loss_over_time = []
            test_loss_over_time = []
            class_correct_over_time = []
            class_incorrect_over_time = []
            class_total_over_time = []
            for epoch in range(self.num_epochs):
                loss_ = self.train_step()
                loss_over_time.append(loss_)
                print('[%d] loss: %.3f' %
                                (epoch + 1, loss_ / self.batch_per_epoch))
                if ((epoch+1) % 1) == 0:
                    self.save_net(epoch+1)

                    test_loss, class_correct, class_incorrect, class_total = self.test()
                    test_loss_over_time.append(test_loss)
                    class_correct_over_time.append(class_correct)
                    class_incorrect_over_time.append(class_incorrect)
                    class_total_over_time.append(class_total)

                    for k in range(self.num_classes):
                        try:
                            print('Validation Set Accuracy at Epoch %d for class %d is %.1f' % (epoch, k,100*class_correct[k]/class_total[k]))
                        except:
                            print("No Samples from this class")
                    val_acc = 100*sum(class_correct)/sum(class_total)
                    print("Total Accuracy : %.3f" % val_acc)            
                    
                    print('[%d - %.3f] TEST loss: %.3f' %
                                (epoch + 1, (test_loss / self.batch_per_epoch), self.weights[0]))
            
            data_dict = {'train_loss': loss_over_time, 'test_loss': test_loss_over_time, 'class_correct': class_correct_over_time, 'class_incorrect':class_incorrect_over_time, 'class_total':class_total_over_time}
            name = self.name + '_results_' + str(run)
            self.save_data(name, data_dict)

    def test(self):
        with torch.no_grad():
            class_correct = list(0 for _ in range(self.num_classes))
            class_incorrect = list(0 for _ in range(self.num_classes))
            class_total = list(0 for _ in range(self.num_classes))
            running_loss = 0.0
            for i,data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device),data[1].to(self.device)
                #print(inputs.size())
                self.optimiser.zero_grad()
                outputs = self.net(inputs)                    
                loss = self.loss_func(outputs, labels)
                running_loss += loss.item()
                _,predicted = torch.max(outputs.data,1)
                c = (predicted==labels).squeeze()
                try:
                    for j in range(len(c)):
                        label = labels[j]
                        class_correct[label] += c[j].item()
                        class_incorrect[label] += not c[j].item()
                        class_total[label] += 1
                except:
                    label = labels[0]
                    class_correct[label] += c.item()
                    class_incorrect[label] += not c.item()
                    class_total[label] += 1                    
            print(class_total)
            return running_loss, class_correct, class_incorrect, class_total


if __name__ == '__main__':
    batch_size = 30 #12
    num_epochs = 100

    train_data_root = '../data/train_data.csv'
    test_data_root = '../data/PARSED_NW_Ground_Stations_2018.csv'
    label_name = 'daily_precip'
    num_classes = 2
    # seq_lens = [3, 5, 7]
    # # lr = 0.0001
    # # hidden = 128
    # loss_weights = torch.Tensor([0.25,1])
    # loss_weight_list = [torch.Tensor([0.1,1]), torch.Tensor([0.25,1]), torch.Tensor([0.4,1])]
    # hidden = [64,128,256]
    # lrs = [0.00001,0.0001,0.001]
    # bidi = [False, True]
    # for loss_weights in loss_weight_list:
    #     for seq_len in seq_lens:
    #         for h in hidden:
    #             for lr in lrs:
    #                 for bi in bidi:
    #                     name = "_loss_weight_%.2f_seq_len_%d_hidden_%d_lr_%.6f_bidi_%s" % (loss_weights[0], seq_len,h, lr, str(bi))
    #                     net = LSTM_FF(5,h,num_classes,seq_len,batch_first=True, bidirectional=True)#LSTM(6,1)Conv_LSTM(6,128,num_classes,batch_first=True, bidirectional=True)
    #                     mt = ModelTrainer(net, lr, num_classes,loss_weights,batch_size, num_epochs, train_data_root, test_data_root, label_name, seq_len,name=name)
    #                     mt.train_loop()
    loss_weights = torch.Tensor([0.32,1])
    seq_len = 7
    bidi = False
    lr = 0.00001
    h = 128

    name = "NO_SPEED_FIXED_DATALOADER_FINAL_TRAINED_LSTM"
    net = LSTM_FF(4,h,num_classes,seq_len,batch_first=True, bidirectional=bidi)#LSTM(6,1)Conv_LSTM(6,128,num_classes,batch_first=True, bidirectional=True)
    mt = ModelTrainer(net, lr, num_classes,loss_weights,batch_size, num_epochs, train_data_root, test_data_root, label_name, seq_len,name=name)
    mt.train_loop()
    #loss_weight_list = [torch.Tensor([0.1,1]), torch.Tensor([0.25,1]), torch.Tensor([0.3,1]),torch.Tensor([0.4,1]),torch.Tensor([0.5,1])]    
    # for loss_weights in loss_weight_list:
    #     name = "_TESTING_GRID_SEARCH_RESULTS_LOSS_WEIGHTS_%.3f" % loss_weights[0]
    #     net = LSTM_FF(5,h,num_classes,seq_len,batch_first=True, bidirectional=bidi)#LSTM(6,1)Conv_LSTM(6,128,num_classes,batch_first=True, bidirectional=True)
    #     mt = ModelTrainer(net, lr, num_classes,loss_weights,batch_size, num_epochs, train_data_root, test_data_root, label_name, seq_len,name=name)
    #     mt.train_loop()
    # workers = 2

    # train_dataset = WeatherDataset(train_data_root, label_name)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # print(train_dataset.targets.count())#,train_dataset.targets.count(1),train_dataset.targets.count(2))

    # test_dataset = WeatherDataset(test_data_root, label_name)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)  


    # num_gpus =1
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and num_gpus >0) else "cpu")

    
    # num_train_samples = len(train_dataset.targets)
    # batch_per_epoch = num_train_samples/batch_size
    # num_classes = 2
    # manual_seed = 555
    #random.seed(manual_seed)
    #torch.manual_seed(manual_seed)

    # lr = 0.0001
    # net = LSTM_FF(5,128,num_classes,batch_first=True, bidirectional=True)#LSTM(6,1)Conv_LSTM(6,128,num_classes,batch_first=True, bidirectional=True)
    

    # net.to(device)
    # self.optimiser = optim.Adam(self.net.parameters(), lr=lr)
    # self.weights = torch.Tensor([0.25,1]).to(self.device)
    # self.loss_func = nn.CrossEntropyLoss(weight=self.weights)

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     for i,data in enumerate(train_loader):
    #         inputs, labels = data[0].to(device),data[1].to(device)
    #         #labels.unsqueeze_(1)
    #         #inputs.unsqueeze_(0)
    #         #print(inputs.size())
    #         optimiser.zero_grad()
    #         outputs = net(inputs)
    #         #outputs.squeeze_(1)
    #         #outputs = outputs.reshape(outputs.shape[0])
    #         #print(outputs[:5], labels[:5])
    #         outputs.squeeze_(1)
    #         loss = loss_func(outputs, labels)

    #         loss.backward()
    #         optimiser.step()
    #         #print(outputs, labels, loss.item())
    #         running_loss += loss.item()

    #     print('[%d, %5d] loss: %.3f' %
    #                     (epoch + 1, i + 1, running_loss / batch_per_epoch))
    #     if ((epoch+1) % 10) == 0:
    #         name = 'no_winddir_lstm_state_dict_62perc' + str(epoch+1)
    #         torch.save(net.state_dict(), name)
    #         with torch.no_grad():
    #             class_correct = list(0 for _ in range(num_classes))
    #             class_incorrect = list(0 for _ in range(num_classes))
    #             class_total = list(0 for _ in range(num_classes))
    #             running_loss = 0.0
    #             for i,data in enumerate(test_loader):
    #                 inputs, labels = data[0].to(device),data[1].to(device)
    #                 #print(inputs.size())
    #                 optimiser.zero_grad()
    #                 outputs = net(inputs)                    
    #                 loss = loss_func(outputs, labels)
    #                 running_loss += loss.item()
    #                 _,predicted = torch.max(outputs.data,1)
    #                 c = (predicted==labels).squeeze()
    #                 try:
    #                     for j in range(len(c)):
    #                         label = labels[j]
    #                         class_correct[label] += c[j].item()
    #                         class_incorrect[label] += not c[j].item()
    #                         class_total[label] += 1
    #                 except:
    #                     label = labels[0]
    #                     class_correct[label] += c.item()
    #                     class_incorrect[label] += not c.item()
    #                     class_total[label] += 1                    
    #             print(class_total)
    #             for k in range(num_classes):
    #                 try:
    #                     print('Validation Set Accuracy at Epoch %d for class %d is %.1f' % (epoch, k,100*class_correct[k]/class_total[k]))
    #                 except:
    #                     print("No Samples from this class")
    #             val_acc = 100*sum(class_correct)/sum(class_total)
    #             print("Total Accuracy : %.3f" % val_acc)            
                
    #             print('[%d, %5d] TEST loss: %.3f' %
    #                         (epoch + 1, i + 1, running_loss / batch_per_epoch))
        