import torch
import tqdm
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import const
import models
import heads

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 0.0001
NUM_VAL_DATA = 6000
LOG_LOSS_EVERY = 100
LOG = defaultdict(list)

def update_log(epoch, batch, loss=None, accuracy=None):
    LOG['epoch'].append(epoch)
    LOG['batch'].append(batch)
    LOG['loss'].append(loss)
    LOG['accuracy'].append(loss)

def train(net, optimizer, loss_function, train_loader, epoch):
    net.train()

    for batch, (x, target) in enumerate(train_loader, 1):
        optimizer.zero_grad()
    
        x = x.view(x.shape[0], -1)
        output = net(x, target)
        # loss = loss_function(output[const.ARCFACE], target)
        loss = loss_function(output[const.FC], target)
        loss.backward()
    
        optimizer.step()

        if batch % LOG_LOSS_EVERY == 0:
            print(f'Epoch {epoch}, Batch {batch}, loss={loss.item()}')
            update_log(epoch, batch, loss=loss.item(), accuracy=None)

def validate(net, optimizer, loss_function, val_loader, epoch):
    net.eval()

    num_correct = 0.
    for batch, (x, target) in enumerate(val_loader, 1):
        x = x.view(x.shape[0], -1)
        output = net(x, target)
        # num_correct += output[const.ARCFACE].argmax(1).eq(target).sum()
        num_correct += output[const.FC].argmax(1).eq(target).sum().item()

    print('Accuracy: %d/%d (%.2f)' % (num_correct, NUM_VAL_DATA, num_correct / NUM_VAL_DATA))

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

    # heads = {const.ARCFACE: heads.ArcFaceHead(dim_in=const.DIM_FEATURE, dim_out=const.NUM_CLASSES)}
    heads = {const.FC: heads.FullyConnectedHead(dim_in=const.DIM_FEATURE, dim_out=const.NUM_CLASSES)}
    net = models.SimpleDNN(heads=heads, dim_in=const.DIM_IN, dim_feature=const.DIM_FEATURE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_function = torch.nn.CrossEntropyLoss()

    net.describe()

    for epoch in range(NUM_EPOCHS):
        validate(net, optimizer, loss_function, val_loader, epoch)
        train(net, optimizer, loss_function, train_loader, epoch)
        
