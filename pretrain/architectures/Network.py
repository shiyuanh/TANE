import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.nn.utils.weight_norm import WeightNorm
from scipy.special import gamma

import numpy as np
import math
import pdb
from architectures.ResNetFeat import create_feature_extractor
from architectures.LossFeat import *

DIM_CL = 128

class ClassifierCombo(nn.Module):
    def __init__(self, in_dim, n_classes, c_type, temp=10.0):
        super().__init__()
        if c_type == 'cosine':
            self.classifier = nn.Linear(in_dim, n_classes, bias = False)
            WeightNorm.apply(self.classifier, 'weight', dim=0) #split the weight update component to direction and norm
        elif c_type == 'linear':
            self.classifier = nn.Linear(in_dim, n_classes, bias = True)
        elif c_type == 'mlp':
            self.classifier = [nn.Linear(in_dim, 1024),nn.Tanh(),nn.Linear(1024, n_classes)]
            self.classifier = nn.Sequential(*self.classifier)
        # https://github.com/wyharveychen/CloserLookFewShot/blob/e03aca8a2d01c9b5861a5a816cd5d3fdfc47cd45/backbone.py#L22
        # https://github.com/arjish/PreTrainedFullLibrary_FewShot/blob/main/classifier_full_library.py#L44

        self.c_type = c_type
        self.temp = nn.Parameter(torch.tensor(temp),requires_grad=False)

    def forward(self, feat):
        if self.c_type in ['linear','mlp']:
            return self.classifier(feat)
        else:
            return self.temp * self.classifier(F.normalize(feat,dim=-1))

class Backbone(nn.Module):
    def __init__(self,args,restype,n_class):
        super(Backbone,self).__init__()
        self.args = args
        self.restype= restype
        self.n_class = n_class
        self.featype = args.featype
        
        self.feature = create_feature_extractor(restype=restype,dataset=args.dataset)
        self.cls_classifier = ClassifierCombo(self.feature.out_dim, self.n_class, 'linear')
        
    def forward(self, left, right=None, need_cont=False):
        the_img = left if right is None else torch.cat([left,right],dim=0)
        resfeat = self.feature(the_img)
        cls_logit = self.cls_classifier(resfeat)
        return resfeat, cls_logit