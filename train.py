import os
import torch
import argparse
import numpy as np
# from models.base_model import Base_Model

from trainer.MetaTrainer import MetaTrainer
from dataloader.dataloader import get_dataloaders

from tensorboardX import SummaryWriter
import pdb


model_pool = ['ResNet18','ResNet12','WRN28']
parser = argparse.ArgumentParser('argument for training')

# General Setting
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=65, help='number of training epochs')
parser.add_argument('--featype', type=str, default='OpenMeta', help='number of training epochs')
parser.add_argument('--restype', type=str, default='ResNet12', choices=model_pool, help='Network Structure')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNetWord','CIFAR-FS', 'FC100'])
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

# Optimization
parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='30', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--tunefeat', type=float, default=0.0, help='update feature parameter')

# Specify folder
parser.add_argument('--logroot', type=str, default='./logs/', help='path to save model')
parser.add_argument('--data_root', type=str, default='data/', help='path to data root')
parser.add_argument('--pretrained_model_path', type=str, default='tieredImageNet_pre.pth', help='path to pretrained model')

# Meta Setting
parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
parser.add_argument('--n_train_para', type=int, default=2, metavar='test_batch_size', help='Size of test batch)')
parser.add_argument('--n_train_runs', type=int, default=300, help='Number of training episodes')
parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')

# Meta Control
parser.add_argument('--gamma', type=float, default=1.0, help='loss cofficient for mse loss')
parser.add_argument('--train_weight_base', type=int, default=0, help='enable training base class weights')
parser.add_argument('--neg_gen_type', type=str, default='semang', choices=['semang', 'attg', 'att', 'mlp'])
parser.add_argument('--base_seman_calib',type=int, default=0, help='base semantics calibration')
parser.add_argument('--agg', type=str, default='avg', choices=['avg', 'mlp'])

parser.add_argument('--tune_part', type=int, default=2, choices=[1,2, 3, 4])
parser.add_argument('--base_size', default=-1, type=int)
parser.add_argument('--n_open_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--funit', type=float, default=1.0)



if __name__ == "__main__":
    torch.manual_seed(0)

    args = parser.parse_args()

    args.n_train_runs = args.n_train_runs * args.n_train_para
    args.n_gpu = len(args.gpus.split(',')) 
    args.train_weight_base = args.train_weight_base==1
    args.base_seman_calib = args.base_seman_calib==1

    open_train_val_loader, meta_test_loader, n_cls = get_dataloaders(args,'openmeta')
    dataloader_trainer = (open_train_val_loader, meta_test_loader, n_cls)
    args.base_size = n_cls if args.base_size == -1 else args.base_size
    trainer = MetaTrainer(args,dataloader_trainer,meta_test_loader)
    trainer.train(meta_test_loader)
        
    