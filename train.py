import torch
import tqdm
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import backbones

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BATCH_SIZE = 16
NUM_VAL_DATA = 6000

if __name__ == '__main__':
    train_set = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    val_set = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('data/', download=True, train=False, transform=transforms.ToTensor())
    
    indices = list(range(len(train_set)))
    train_indices, val_indices = indices[NUM_VAL_DATA:], indices[:NUM_VAL_DATA]
    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_set, batch_size=16, sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=16, sampler=val_sampler, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=1)

    net = backbones.DNN()
    for index, (x, target) in tqdm.tqdm(enumerate(train_loader), 1):
        print(target)

