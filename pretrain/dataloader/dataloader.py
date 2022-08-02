from __future__ import print_function

import os,sys
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .cifar import PreCIFAR, MetaCIFAR
from .mini_imagenet import PreMini, MetaMini
from .tiered_imagenet import PreTiered, MetaTiered

def get_dataloaders(opt,mode='contrast'):
    # dataloader
    opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    
    if opt.dataset == 'miniImageNet':
        n_cls = 64
        meta_1shot_loader = DataLoader(MetaMini(opt,1,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_5shot_loader = DataLoader(MetaMini(opt,5,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_test_loader = (meta_1shot_loader,meta_5shot_loader)
        pre_train_loader = DataLoader(PreMini(opt,'train',True), batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        return pre_train_loader, meta_test_loader, n_cls
    
    elif opt.dataset in ['CIFAR-FS','FC100']:

        n_cls = 64 if opt.dataset == 'CIFAR-FS' else 60
        meta_1shot_loader = DataLoader(MetaCIFAR(opt,1,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_5shot_loader = DataLoader(MetaCIFAR(opt,5,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_test_loader = (meta_1shot_loader,meta_5shot_loader)
        pre_train_loader = DataLoader(PreCIFAR(opt,'train',True), batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        return pre_train_loader, meta_test_loader, n_cls
    
    elif opt.dataset == 'tieredImageNet':
        n_cls = 351
        meta_1shot_loader = DataLoader(MetaTiered(opt,1,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_5shot_loader = DataLoader(MetaTiered(opt,5,'test',False), batch_size=1, shuffle=False, drop_last=False,num_workers=opt.num_workers) 
        meta_test_loader = (meta_1shot_loader,meta_5shot_loader)
        pre_train_loader = DataLoader(PreTiered(opt,'train',True), batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers)
        return pre_train_loader, meta_test_loader, n_cls

    else:
        raise ValueError('Dataset Not in Record, Pls check the CONFIGS')
