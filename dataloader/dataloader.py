from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .mini_imagenet import OpenMini, GenMini
from .cifar import OpenCIFAR
from .tiered_imagenet import OpenTiered

import numpy as np

    
def get_dataloaders(opt,mode='open'):
    # dataloader
    opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)

    if mode == 'gopenmeta':
        assert opt.dataset == 'miniImageNet'
        n_cls = 64
        open_train_loader = DataLoader(GenMini(opt,'train','episode', True), batch_size=opt.n_train_para, shuffle=False, num_workers=opt.num_workers)
        meta_test_loader = DataLoader(GenMini(opt,'test','episode', False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers)
        return open_train_loader, meta_test_loader, n_cls

    assert mode == 'openmeta'

    if opt.dataset == 'miniImageNet':
        n_cls = 64
        open_train_loader = DataLoader(OpenMini(opt,'train','episode', True), batch_size=opt.n_train_para, shuffle=False, num_workers=opt.num_workers)
        meta_test_loader = DataLoader(OpenMini(opt,'test','episode', False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers)
        return open_train_loader, meta_test_loader, n_cls
    elif opt.dataset in ['FC100','CIFAR-FS']:
        n_cls = 60 if opt.dataset=='FC100' else 64
        open_train_loader = DataLoader(OpenCIFAR(opt,'train','episode', True), batch_size=opt.n_train_para, shuffle=False, num_workers=opt.num_workers)
        meta_test_loader = DataLoader(OpenCIFAR(opt,'test','episode', False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers)
        return open_train_loader, meta_test_loader, n_cls
    elif opt.dataset in ['tieredImageNet', 'tieredImageNetWord']:
        n_cls = 351
        open_train_loader = DataLoader(OpenTiered(opt,'train','episode', True), batch_size=opt.n_train_para, shuffle=False, num_workers=opt.num_workers)
        meta_test_loader = DataLoader(OpenTiered(opt,'test','episode', False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers)
        return open_train_loader, meta_test_loader, n_cls
    else:
        raise NotImplementedError(opt.dataset)
        