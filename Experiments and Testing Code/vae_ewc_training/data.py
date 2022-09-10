import random
import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28        
        if self.train:
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.data])
        else:
            self.data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.data])
        if torch.isnan(self.data).any():
            print("DATA IS NONE! DATA IS NONE!")

    def __getitem__(self, index):

        if self.train:
            img, target = self.data[index], self.train_labels[index]
        else:
            img, target = self.data[index], self.targets[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.data[sample_idx]]
