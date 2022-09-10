from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

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

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
            input = variable(input)
            output = self.model(input).view(1, -1)
            label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            pre_loss = loss
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            inf = torch.isinf(p)
            if inf.any():
                with torch.no_grad():
                    p = torch.nan_to_num(p) #p[(inf==True).nonzero()] = 0
            inf = torch.isnan(p)
            if inf.any():
                with torch.no_grad():
                    p = torch.nan_to_num(p) #p[(inf==True).nonzero()] = 0
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print("P IS BROKE")
                    
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            
            if torch.isinf(p).any() or torch.isnan(p).any():
                print("P IS NAN/INF")
            if torch.isinf(self._means[n]).any() or torch.isnan(self._means[n]).any():
                print("MEANS IS NAN/INF")
            if torch.isinf(self._precision_matrices[n]).any() or torch.isnan(self._precision_matrices[n]).any():
                print("MATREIX IS NAN/INF")
            loss += _loss.sum()
        if torch.isinf(loss):
            loss = torch.nan_to_num(loss)
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        try:
            epoch_loss += loss.data[0] if loss.data.numel() > 1 else loss.data.item()
        except:
            print(loss)
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()

    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        pen = ewc.penalty(model)
        if torch.isnan(pen).any():
            print("PEN IS NAN")
        if torch.isinf(pen).any():
            print("PEN IS INF")
        loss = F.cross_entropy(output, target) + importance * pen
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("LOSS")
            #loss = torch.nan_to_num(loss)
        ls = loss.data[0] if loss.data.numel() > 1 else loss.data.item()
        epoch_loss += ls
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    class_matrix = {}
    for i in range(10):
        class_matrix[i] = {}
        class_matrix[i]['total_pred'] =0
        class_matrix[i]['true_positive'] =0
        class_matrix[i]['false_negative'] =0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        prediction = F.softmax(output, dim=1).max(dim=1)[1]
        for idx,each in enumerate(prediction):
            class_matrix[each.item()]['total_pred'] += 1
            if each == target[idx]:
                class_matrix[each.item()]['true_positive'] += 1
            else:
                class_matrix[target[idx].item()]['false_negative'] += 1

        correct += (prediction == target).data.sum()
    return correct / len(data_loader.dataset), class_matrix
