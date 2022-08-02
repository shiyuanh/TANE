import os,pdb
import torch
import argparse
import numpy as np

from trainer.PreTrainer import PreTrainer
from dataloader.dataloader import get_dataloaders

model_pool = ['ResNet12']
task_pool = ['Entropy','EntropyRot']
parser = argparse.ArgumentParser('argument for training')

# General
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=2, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=90, help='number of training epochs')
parser.add_argument('--eval', action='store_true', help='using cosine annealing')
parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')
parser.add_argument('--featype', type=str, default='EntropyRot', choices=task_pool, help='number of training epochs')

# dataset
parser.add_argument('--restype', type=str, default='ResNet12', choices=model_pool)
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet','CIFAR-FS', 'FC100'])
parser.add_argument('--model_path', type=str, default='./logs/', help='path to save model')
parser.add_argument('--data_root', type=str, default='/home/jiawei/DATA/', help='path to data root')

# few-shot setting
parser.add_argument('--n_episodes', type=int, default=1000, metavar='N', help='Number of test runs')
parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')

# optimization
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='60', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

#hyper parameters
parser.add_argument('--rotangle', type=int, default=10,help='rotation angle of the weak augmentation')
parser.add_argument('--temp', type=float, default=0.5, help='temperature of the contrastive loss')
parser.add_argument('--use_bce', action='store_true')


args = parser.parse_args()
args.logroot = os.path.join(os.path.abspath('.'),'logs')
if not os.path.isdir(args.logroot):
    os.makedirs(args.logroot)
args.n_gpu = torch.cuda.device_count()

if __name__ == "__main__":

    pre_train_loader, meta_test_loader, n_cls = get_dataloaders(args,'entropy')
    dataloader_trainer = (pre_train_loader, None, n_cls)
    trainer = PreTrainer(args,dataloader_trainer)
    trainer.train(meta_test_loader)
