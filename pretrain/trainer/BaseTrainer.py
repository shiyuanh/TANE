from __future__ import print_function

import os
import pdb
import numpy as np
import argparse
import socket
import time
import sys
from tqdm import tqdm
import mkl
import math
import h5py

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from architectures.Network import Backbone
from architectures.LossFeat import SupConLoss
from trainer.MetaEval import meta_evaluation
from util import adjust_learning_rate, accuracy, AverageMeter, rot_aug

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def model_name(args):
    model_name = '{}_{}_batch_{}_lr_{}'.format(args.dataset, args.restype, args.batch_size, args.learning_rate)
    if args.restype in ['ViT','Swin']:
        model_name = '{}_Trans{}'.format(model_name,args.vit_dim)
    if args.featype == 'Contrast':
        model_name = '{}_temp_{}_even_{}'.format(model_name,args.temp,args.even)
    if args.featype == 'Entropy':
        model_name = model_name + '_bce' if args.use_bce else model_name
    return model_name

class BaseTrainer(object):
    def __init__(self, args, dataset_trainer):
        args.logroot = os.path.join(args.logroot, args.featype)
        if not os.path.isdir(args.logroot):
            os.makedirs(args.logroot)
        
        # set the path according to the environment
        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = list([])
        for it in iterations:
            args.lr_decay_epochs.append(int(it))
        
        args.model_name = model_name(args)
        self.save_path = os.path.join(args.logroot, args.model_name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

        self.args = args
        self.train_loader, self.val_loader, self.n_cls = dataset_trainer

        # model & optimizer
        self.model = Backbone(args, args.restype, self.n_cls)
        if self.args.restype in ['ViT','Swin']:
            self.optimizer = optim.AdamW(self.model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05)
            # self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
        self.criterion = {'feat':SupConLoss(temperature=args.temp),'logit':nn.BCEWithLogitsLoss() if self.args.use_bce else nn.CrossEntropyLoss()} #
        # print(self.model)
        # print(self.optimizer)

        if torch.cuda.is_available():
            if args.n_gpu > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
            self.criterion = {name:loss.cuda() for name,loss in self.criterion.items()}
            cudnn.benchmark = True
    
    def train(self, eval_loader=None):

        trlog = {'args':vars(self.args), 'max_1shot_meta':0.0, 'max_5shot_meta':0.0, 'max_1shot_epoch':0, 'max_5shot_epoch':0}        
        writer = SummaryWriter(self.save_path)

        # routine: supervised pre-training
        for epoch in range(1, self.args.epochs + 1):

            adjust_learning_rate(epoch, self.args, self.optimizer)
            train_loss, train_msg = self.train_epoch(epoch, self.train_loader, self.model, self.criterion, self.optimizer, self.args)

            writer.add_scalar('train/loss', float(train_loss), epoch)
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            #evaluate
            if eval_loader is not None and (epoch % 10 == 0 or epoch > 55):
                start = time.time()
                eval_1shot_loader,eval_5shot_loader = eval_loader
                meta_1shot_acc, meta_1shot_std = meta_evaluation(self.model, eval_1shot_loader)
                meta_5shot_acc, meta_5shot_std = meta_evaluation(self.model, eval_5shot_loader)
                test_time = time.time() - start
                writer.add_scalar('MetaAcc/1shot', float(meta_1shot_acc), epoch)
                writer.add_scalar('MetaStd/1shot', float(meta_1shot_std), epoch)
                writer.add_scalar('MetaAcc/5shot', float(meta_5shot_acc), epoch)
                writer.add_scalar('MetaStd/5shot', float(meta_5shot_std), epoch)
                meta_msg = 'Meta Test Acc: 1-shot {:.4f} 5-shot {:.4f}, Meta Test std: {:.4f} {:.4f}, Time: {:.1f}'.format(meta_1shot_acc, meta_5shot_acc, meta_1shot_std, meta_5shot_std, test_time)
                train_msg = train_msg + ' | ' + meta_msg
                if trlog['max_1shot_meta'] < meta_1shot_acc:
                    trlog['max_1shot_meta'] = meta_1shot_acc
                    trlog['max_1shot_epoch'] = epoch
                    self.save_model(epoch,'max_meta')
                if trlog['max_5shot_meta'] < meta_5shot_acc:
                    trlog['max_5shot_meta'] = meta_5shot_acc
                    trlog['max_5shot_epoch'] = epoch
                    self.save_model(epoch,'max_meta_5shot') # will not use
            
            print(train_msg)
            if epoch % 10 == 0 or epoch==self.args.epochs:
                self.save_model(epoch,'last')
                print('The Best Meta 1(5)-shot Acc {:.4f}({:.4f}) in Epoch {}({})'.format(trlog['max_1shot_meta'],trlog['max_5shot_meta'],trlog['max_1shot_epoch'],trlog['max_5shot_epoch']))
            torch.save(trlog, os.path.join(self.save_path, 'trlog'))
        
    def train_epoch(self, epoch, train_loader, model, criterion, optimizer, args):
        """One epoch training"""
        return 0,'to be updated'
    
    def save_model(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'params': self.model.state_dict()
        }     
        file_name = '{}.pth'.format('epoch_'+str(epoch) if name is None else name)
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.save_path, file_name))
    
    def eval_report(self,eval_loader,path):
        print('Loading data from', path)
        params = torch.load(path)['params']
        if 'tiered' in self.args.dataset:
            params = {'.'.join(k.split('.')[1:]):v for k,v in params.items()}
        model_dict = self.model.state_dict()
        model_dict.update(params)
        self.model.load_state_dict(model_dict)
        self.model.eval()

        eval_1shot_loader,eval_5shot_loader = eval_loader
        meta_1shot_acc, meta_1shot_std = meta_evaluation(self.model, eval_1shot_loader)
        meta_5shot_acc, meta_5shot_std = meta_evaluation(self.model, eval_5shot_loader)
        print('Linear Regression: 1(5)-shot Accuracy {:.4f}({:.4f}) Std {:.4f}({:.4f})'.format(meta_1shot_acc,meta_5shot_acc,meta_1shot_std,meta_5shot_std))
        meta_1shot_acc, meta_1shot_std = meta_evaluation(self.model, eval_1shot_loader, type_classifier='proto')
        meta_5shot_acc, meta_5shot_std = meta_evaluation(self.model, eval_5shot_loader, type_classifier='proto')
        print('Proto Classification 1(5)-shot Accuracy {:.4f}({:.4f}) Std {:.4f}({:.4f})'.format(meta_1shot_acc,meta_5shot_acc,meta_1shot_std,meta_5shot_std))
