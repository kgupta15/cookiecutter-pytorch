#!/usr/bin/env python

import os
import sys
import argparse
from argparse import RawTextHelpFormatter
import time

import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
import torchvision
from torch.autograd import Variable

from dataloader import *
from models import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params

def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = NetworkConfig(args.config)

    args.distributed = config.distributed['world_size'] > 1
    if args.distributed:
        print('[+] Distributed backend')
        dist.init_process_group(backend=config.distributed['dist_backend'], init_method=config.distributed['dist_url'],\
                                world_size=config.distributed['world_size'])

    if args.pretrained:
        print('[+] Using pretrained model')
    else:
        print('[!] Creating Model')

    for epoch in range(config.hyperparameters['num_epochs']):
        pass

def main():
    parser = argparse.ArgumentParser(description='Disentangling Variations', formatter_class=RawTextHelpFormatter)

    parser.add_argument('--gpu', type=int, default=0, \
                        help="Turn ON for GPU support; default=0")
    parser.add_argument('--resume', type=int, default=0, \
                        help="Turn ON to resume training from latest checkpoint; default=0")
    parser.add_argument('--chkpts', type=str, default="./checkpoints", \
                        help="Mention the dir that contains checkpoints")
    parser.add_argument('--config', type=str, required=True, \
                        help="Mention the file to load required configurations of the model")
    parser.add_argument('--seed', type=int, default=100, \
                        help="Seed for random function, default=100")
    parser.add_argument('--pretrained', type=int, default=1, \
                        help="Turn ON if checkpoints of model available in /checkpoints dir")
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
