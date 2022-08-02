import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb
from architectures.AttnClassifier import *

class GClassifier(Classifier):
    def __init__(self, args, feat_dim, param_seman, train_weight_base=False):
        super(GClassifier, self).__init__(args, feat_dim, param_seman, train_weight_base)

        # Weight & Bias for Base
        self.train_weight_base = train_weight_base
        self.init_representation(param_seman)
        if train_weight_base:
            print('Enable training base class weights')
        
        self.calibrator = SupportCalibrator(nway=args.n_ways, feat_dim=feat_dim, n_head=1, base_seman_calib=args.base_seman_calib, neg_gen_type=args.neg_gen_type)
        self.open_generator = OpenSetGenerater(args.n_ways, feat_dim, n_head=1, neg_gen_type=args.neg_gen_type, agg=args.agg)
        self.metric  = Metric_Cosine()

    def forward(self, features, cls_ids, test=False):
        ## bs: features[0].size(0)
        ## support_feat: bs*nway*nshot*D
        ## query_feat: bs*(nway*nquery)*D
        ## base_ids: bs*54
        (support_feat, query_feat, openset_feat, baseset_feat) = features
        
        (nb,nc,ns,ndim),nq = support_feat.size(),query_feat.size(1)
        (supp_ids, base_ids) = cls_ids

        base_weights,base_wgtmem,base_seman,support_seman = self.get_representation(supp_ids,base_ids)
        support_feat = torch.mean(support_feat, dim=2)

        supp_protos,support_attn = self.calibrator(support_feat, base_weights, support_seman, base_seman)

        fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights, support_seman, base_seman)
        cls_protos = torch.cat([base_weights, supp_protos, fakeclass_protos], dim=1)


        query_funit_distance = 1.0- self.metric(recip_unit, query_feat)
        qopen_funit_distance = 1.0- self.metric(recip_unit, openset_feat)
        funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

        query_cls_scores = self.metric(cls_protos, query_feat)
        openset_cls_scores = self.metric(cls_protos, openset_feat)
        baseset_cls_scores = self.metric(cls_protos, baseset_feat)

        test_cosine_scores = (baseset_cls_scores,query_cls_scores,openset_cls_scores)
        return test_cosine_scores, supp_protos, fakeclass_protos, (base_weights,base_wgtmem), funit_distance



