import argparse
import torch
import tqdm
import yaml
import numpy as np
import pandas as pd

from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import const
import models
import heads
import trainer
import logger
import plotter

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Configuration.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    config = yaml.safe_load(open(args.config))
    logger = logger.Logger(config['log_file'], log_loss_every=const.LOG_LOSS_EVERY)

    train_set = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    val_set = datasets.MNIST('data/', download=True, train=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('data/', download=True, train=False, transform=transforms.ToTensor())
    
    indices = list(range(len(train_set)))
    train_indices, val_indices = indices[const.NUM_VAL_DATA:], indices[:const.NUM_VAL_DATA]
    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'], sampler=val_sampler, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=1)

    # Prepare heads according to the configuration.
    head_dict = dict()
    for head in config['heads']:
        if head == const.FC:
            head_dict[const.FC] = heads.FullyConnectedHead(dim_in=const.DIM_FEATURE, dim_out=const.NUM_CLASSES)
        elif head == const.ARCFACE:
            head_dict[const.ARCFACE] = heads.ArcFaceHead(dim_in=const.DIM_FEATURE, dim_out=const.NUM_CLASSES)
        else:  # TODO: Add cosface.
            pass

    # Attach heads to our backbone model.
    if config['backbone'] == const.SIMPLEDNN:
        net = models.SimpleDNN(heads=head_dict, dim_in=const.DIM_IN, dim_feature=const.DIM_FEATURE)
    else:  # TODO: Add convnet.
        pass

    net.to(const.DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'])

    # Define losses.
    criterion_dict = dict()
    for head in config['heads']:
        criterion_dict[head] = (1.0, torch.nn.CrossEntropyLoss())

    net.describe()

    for epoch in range(1, config['num_epochs']+1):
        trainer.train(net, optimizer, criterion_dict, train_loader, epoch, logger, flatten=True)
        trainer.validate(net, optimizer, criterion_dict, val_loader, epoch, logger, flatten=True)
        
    plotter.plot_acc(config['log_file'], config['img_file'])