from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

manual_seed = 555
torch.manual_seed(manual_seed)
np.random.seed(555)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list, dataset_labels:list, vae: nn.Module):
        #print("Need to swap back to F.CrossEntropy() loss in utils.py EWC class: normal_train() and ewc_train()")
        #print("also need to go back to softmax correct func in test()")
        self.model = model
        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.vae = vae
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.train()
        trainable_params = {'lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.weight_hi_l0'}
        for j,input in enumerate(self.dataset):
            self.model.zero_grad()
            input = variable(torch.Tensor(input))
            input.unsqueeze_(0)
            output = self.model(input.view(1, -1))
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            for n, p in self.model.named_parameters():
                if n in trainable_params:
                    open_idx = np.arange(p.numel())
                    idxs = np.random.choice(open_idx, size=(int(len(p)/16), 16), replace=False)
                    max_y, max_x = p.size()
                    vae_input = variable(torch.empty(16))
                    for i_list in idxs:
                        for i,each in enumerate(i_list):                 
                            x = int(each%max_y)
                            y = int(each/max_y)
                            vae_input[i] = p.grad.data[x,y]
                        likelihood,_ = self.vae.estimate_likelihood(vae_input)
                        for id,val in enumerate(open_idx):
                            if val in i_list:
                                open_idx[id]  = likelihood[0][self.dataset_labels[j]]
                    open_idx = open_idx.reshape(max_y, max_x)
                    params = variable(torch.Tensor(open_idx))
                    precision_matrices[n].data += params / len(self.dataset)
                else:
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)
        #TODO replace precision matrices with the 
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss

    def train_vae(self,vae, vae_optim,input_size):
        print("TRAINING VAE")
        trainable_params = {'lstm.weight_ih_l0','lstm.weight_hh_l0','lstm.weight_hi_l0'}
        param_len = 0
        for n,p in self.model.named_parameters():
            if n in trainable_params:
                param_len+=1 

        vae_total_loss = 0
        self.model.train()
        vae.train()

        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        
        for j,input in enumerate(self.dataset):
            self.model.zero_grad()
            input = variable(torch.Tensor(input))
            input.unsqueeze_(0)
            output = self.model(input.view(1, -1))
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()
            print("Start")
            vae_epoch_loss = 0
            for n, p in self.model.named_parameters():
                if n in trainable_params:
                    vae_optim.zero_grad()
                    open_idx = np.arange(p.numel())
                    idxs = np.random.choice(open_idx, size=(int(len(p)/input_size), input_size), replace=False)
                    max_y, max_x = p.size()
                    vae_input = variable(torch.empty(input_size))
                    for i_list in idxs:
                        for i,each in enumerate(i_list):                 
                            x = int(each%max_y)
                            y = int(each/max_y)
                            vae_input[i] = p.grad.data[x,y]
                        vae_x,vae_z,recon_x,mu,logvar = vae(vae_input)
                        (BCE_loss, KLD_loss, class_loss) = self.vae.loss_fnc(recon_x, vae_z, vae_x, mu, logvar, self.dataset_labels[j])
                        vae_loss = BCE_loss + KLD_loss + class_loss
                        vae_loss.backward()
                        vae_epoch_loss += (vae_loss/param_len)
                        vae_optim.step()
            print("End")
            vae_total_loss.append(vae_epoch_loss)
        return vae_total_loss





def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, loss_weights):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target, weight=loss_weights)
        epoch_loss += loss.data[0] if loss.data.numel() > 1 else loss.data.item()
        loss.backward()
        optimizer.step()
    return (epoch_loss / len(data_loader))


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float, loss_weights):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target, weight=loss_weights) + importance * ewc.penalty(model)
        epoch_loss += loss.data[0] if loss.data.numel() > 1 else loss.data.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
        #correct += F.mse_loss(output, target)
    return correct / len(data_loader.dataset)
