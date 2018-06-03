#!/usr/bin/env python

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import time

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from dataloader import *
from models import *
from utils import *

best_prec1 = 0
configure("logs/log-1")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = NetworkConfig(args.config)

    args.distributed = config.distributed['world_size'] > 1
    if args.distributed:
        print('[+] Distributed backend')
        dist.init_process_group(backend=config.distributed['dist_backend'], init_method=config.distributed['dist_url'],\
                                world_size=config.distributed['world_size'])

    # creating model instance
    model = Model(config)
    # plotting interactively
    plt.ion()

    if args.distributed:
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu:
        model = nn.DataParallel(model).to(device)
    else: return

    # Data Loading
    train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'], sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'])

    # Training and Evaluation
    trainer = Trainer('cnn', config, train_loader, model)
    evaluator = Evaluator('cnn', config, val_loader, model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), config.hyperparameters['lr'],
                                momentum=config.hyperparameters['momentum'],
                                weight_decay=config.hyperparameters['weight_decay'])

    trainer.setCriterion(criterion)
    trainer.setOptimizer(optimizer)
    evaluator.setCriterion(criterion)

    # optionally resume from a checkpoint
    if args.resume:
        trainer.load_saved_checkpoint(checkpoint=None)

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True

    best_prec1 = 0
    for epoch in range(config.hyperparameters['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        trainer.adjust_learning_rate(epoch)
        trainer.train(epoch)

        prec1 = evaluator.evaluate(epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        trainer.save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Disentangling Variations', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--gpu', type=int, default=0, \
                        help="Turn ON for GPU support; default=0")
    parser.add_argument('--resume', type=int, default=0, \
                        help="Turn ON to resume training from latest checkpoint; default=0")
    parser.add_argument('--checkpoints', type=str, default="./checkpoints", \
                        help="Mention the dir that contains checkpoints")
    parser.add_argument('--config', type=str, required=True, \
                        help="Mention the file to load required configurations of the model")
    parser.add_argument('--seed', type=int, default=100, \
                        help="Seed for random function, default=100")
    parser.add_argument('--pretrained', type=int, default=0, \
                        help="Turn ON if checkpoints of model available in /checkpoints dir")
    parser.add_argument('--evaluate', type=int, default=0, \
                        help='evaluate model on validation set')
    args = parser.parse_args()

    main(args)
