from __future__ import print_function

import os
import pdb
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from .BaseTrainer import BaseTrainer
from util import AverageMeter

def rot_aug(x):
    bs = x.size(0)
    x_90 = x.transpose(2,3).flip(2)
    x_180 = x.flip(2).flip(3)
    x_270 = x.flip(2).transpose(2,3)
    rot_data = torch.cat((x, x_90, x_180, x_270),0)
    rot_label = torch.cat((torch.zeros(bs),torch.ones(bs),2*torch.ones(bs),3*torch.ones(bs)))
    return rot_data, rot_label

class PreTrainer(BaseTrainer):
    def __init__(self, args, dataset_trainer):
        super(PreTrainer,self).__init__(args, dataset_trainer)

    def train_epoch(self, epoch, train_loader, model, criterion, optimizer, args):
        if args.featype == 'EntropyRot':
            return self.ce_rot_epoch(epoch, train_loader, model, criterion, optimizer)
        elif args.featype == 'Entropy':
            return self.ce_epoch(epoch, train_loader, model, criterion, optimizer)
    
    def ce_epoch(self, epoch, train_loader, model, criterion, optimizer):
        """One epoch training"""
        model.train()
        losses = AverageMeter()

        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, (image,target,_) in enumerate(pbar):

                batch_size = target.size()[0]
                # Forward
                _,cls_logits = model(image.cuda())
                if self.args.use_bce:
                    loss = criterion['logit'](cls_logits, F.one_hot(target,self.n_cls).float().cuda())
                else:
                    loss = criterion['logit'](cls_logits, target.cuda())
                losses.update(loss.item(), batch_size)

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Epoch {} Loss".format(epoch) :'{0:.2f}'.format(losses.avg)})

        message = 'Epoch {} Train_Loss {:.3f}'.format(epoch, losses.avg)
        return losses.avg, message
    
    def ce_rot_epoch(self, epoch, train_loader, model, criterion, optimizer):
        """One epoch training"""
        model.train()
        losses = AverageMeter()

        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, (image,target,_) in enumerate(pbar):

                batch_size = target.size()[0]
                image,_ = rot_aug(image)
                # Forward
                _,cls_logits = model(image.cuda())
                loss = criterion['logit'](cls_logits, target.repeat(4).cuda())
                losses.update(loss.item(), batch_size)

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_postfix({"Epoch {} Loss".format(epoch) :'{0:.2f}'.format(losses.avg)})

        message = 'Epoch {} Train_Loss {:.3f}'.format(epoch, losses.avg)
        return losses.avg, message