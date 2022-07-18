import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb
from architectures.ResNetFeat import create_feature_extractor
from architectures.AttnClassifier import Classifier


class FeatureNet(nn.Module):
    def __init__(self,args,restype,n_class,param_seman):
        super(FeatureNet,self).__init__()
        self.args = args
        self.restype = restype
        self.n_class = n_class
        self.featype = args.featype
        self.n_ways = args.n_ways
        self.tunefeat = args.tunefeat
        self.distance_label = torch.Tensor([i for i in range(self.n_ways)]).cuda().long()
        self.metric = Metric_Cosine()

        self.feature = create_feature_extractor(restype,args.dataset)
        self.feat_dim = self.feature.out_dim

        
        self.cls_classifier = Classifier(args, self.feat_dim, param_seman, args.train_weight_base) if 'OpenMeta' in self.featype else nn.Linear(self.feat_dim, n_class)

        assert 'OpenMeta' in self.featype
        if self.tunefeat == 0.0:
            for _,p in self.feature.named_parameters():
                p.requires_grad=False
        else:
            if args.tune_part <= 3:
                for _,p in self.feature.layer1.named_parameters():
                    p.requires_grad=False
            if args.tune_part <= 2:
                for _,p in self.feature.layer2.named_parameters():
                    p.requires_grad=False
            if args.tune_part <= 1:
                for _,p in self.feature.layer3.named_parameters():
                    p.requires_grad=False
                    

    def forward(self, the_img, labels=None, conj_ids=None, base_ids=None, test=False, SupCT=None):
        if labels is None:
            assert the_img.dim() == 4
            return (self.feature(the_img),None)
        else:
            return self.open_forward(the_img, labels, conj_ids, base_ids, test)
            
    
    def open_forward(self, the_input, labels, conj_ids, base_ids, test):
        # Hyper-parameter Preparation
        the_sizes = [_.size(1) for _ in the_input]
        (ne,_,nc,nh,nw) = the_input[0].size()

        # Data Preparation
        combined_data = torch.cat(the_input,dim=1).view(-1,nc,nh,nw)
        if not self.tunefeat:
            with torch.no_grad():
                combined_feat = self.feature(combined_data).detach()
        else:
            combined_feat = self.feature(combined_data)
        support_feat,query_feat,supopen_feat,openset_feat = torch.split(combined_feat.view(ne,-1,self.feat_dim),the_sizes,dim=1)
        (support_label,query_label,supopen_label,openset_label) = labels
        (supp_idx, open_idx) = conj_ids
        cls_label = torch.cat([query_label, openset_label], dim=1)
        test_feats = (support_feat, query_feat, openset_feat)


        ### First Task
        support_feat = support_feat.view(ne, self.n_ways, -1, self.feat_dim)
        test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, loss_funit = self.task_proto((support_feat,query_feat,openset_feat), (supp_idx,base_ids), cls_label, test)
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        test_cls_probs = self.task_pred(test_cosine_scores[0], test_cosine_scores[1])

        if test:
            test_feats = (support_feat, query_feat, openset_feat)
            return test_feats, cls_protos, test_cls_probs

        ## Second task
        supopen_feat = supopen_feat.view(ne, self.n_ways, -1, self.feat_dim)
        _, supp_protos_aug, fakeclass_protos_aug, loss_cls_aug, loss_funit_aug = self.task_proto((supopen_feat,openset_feat,query_feat), (open_idx,base_ids), cls_label, test)
        
        supp_protos = F.normalize(supp_protos, dim=-1)
        fakeclass_protos = F.normalize(fakeclass_protos, dim=-1)
        supp_protos_aug = F.normalize(supp_protos_aug, dim=-1)
        fakeclass_protos_aug = F.normalize(fakeclass_protos_aug, dim=-1)

        loss_open_hinge = 0.0
        # loss_open_hinge_1 = F.mse_loss(fakeclass_protos.repeat(1,self.n_ways, 1), supp_protos)
        # loss_open_hinge_2 = F.mse_loss(fakeclass_protos_aug.repeat(1,self.n_ways, 1), supp_protos_aug) 
        # loss_open_hinge = loss_open_hinge_1 + loss_open_hinge_2
        

        
        loss = (loss_cls+loss_cls_aug, loss_open_hinge, loss_funit+loss_funit_aug)
        return test_feats, cls_protos, test_cls_probs, loss
    
    
    def task_proto(self, features, cls_ids, cls_label,test=False):
        test_cosine_scores, supp_protos, fakeclass_protos, _, funit_distance = self.cls_classifier(features, cls_ids, test)
        (query_cls_scores,openset_cls_scores) = test_cosine_scores
        cls_scores = torch.cat([query_cls_scores,openset_cls_scores], dim=1)
        fakeunit_loss = fakeunit_compare(funit_distance,self.n_ways,cls_label)
        cls_scores,close_label,cls_label = cls_scores.view(-1, self.n_ways+1),cls_label[:,:query_cls_scores.size(1)].reshape(-1),cls_label.view(-1)
        loss_cls = F.cross_entropy(cls_scores, cls_label)
        return test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, fakeunit_loss
    
    
    def task_pred(self, query_cls_scores, openset_cls_scores, many_cls_scores=None):
        query_cls_probs = F.softmax(query_cls_scores.detach(), dim=-1)
        openset_cls_probs = F.softmax(openset_cls_scores.detach(), dim=-1)
        if many_cls_scores is None:
            return (query_cls_probs, openset_cls_probs)
        else:
            many_cls_probs = F.softmax(many_cls_scores.detach(), dim=-1)
            return (query_cls_probs, openset_cls_probs, many_cls_probs, query_cls_scores, openset_cls_scores)



class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1,2))
        return logits * self.temp   

    

def fakeunit_compare(funit_distance,n_ways,cls_label):
    cls_label_binary = F.one_hot(cls_label)[:,:,:-1].float()
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance, target=cls_label_binary))
    return loss