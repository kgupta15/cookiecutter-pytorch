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

best_prec1 = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    """Counts trainable(active) parameters of a model"""
    total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
    return total_params

def save_checkpoint(state, is_best, loc, filename='checkpoint.pth.tar'):
    file_loc = os.path.join(loc, filename)
    torch.save(state, file_loc)
    if is_best:
        shutil.copyfile(file_loc, os.path.join(loc, 'model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = NetworkConfig(args.config)

    args.distributed = config.distributed['world_size'] > 1
    if args.distributed:
        print('[+] Distributed backend')
        dist.init_process_group(backend=config.distributed['dist_backend'], init_method=config.distributed['dist_url'],\
                                world_size=config.distributed['world_size'])

    if args.pretrained:
        model = Model(config, pretrained=True)
    else:
        model = Model(config)

    if not args.distributed:
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), config.hyperparameters['lr'],
                                momentum=config.hyperparameters['momentum'],
                                weight_decay=config.hyperparameters['weight_decay'])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.checkpoint):
            print("[!] Loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("[+] Loaded checkpoint '{}' (epoch {})"
                .format(args.checkpoint, checkpoint['epoch']))
        else:
            print("[-] No checkpoint found at '{}'".format(args.checkpoint))

    cudnn.benchmark = True

    # Data Loading
    traindir = os.path.join(config.data['root'], 'train')
    valdir = os.path.join(config.data['root'], 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'], sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=config.data['batch_size'], shuffle=config.data['shuffle'],
        num_workers=config.data['workers'], pin_memory=config.data['pin_memory'])

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(config.hyperparameters['num_epochs']):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

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
