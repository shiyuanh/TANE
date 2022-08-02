import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np
import math
import pdb

class Classifier(nn.Module):
    def __init__(self, args, feat_dim, param_seman, train_weight_base=False):
        super(Classifier, self).__init__()

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
        (support_feat, query_feat, openset_feat) = features

        (nb,nc,ns,ndim),nq = support_feat.size(),query_feat.size(1)
        (supp_ids, base_ids) = cls_ids
        base_weights,base_wgtmem,base_seman,support_seman = self.get_representation(supp_ids,base_ids)
        support_feat = torch.mean(support_feat, dim=2)

        supp_protos,support_attn = self.calibrator(support_feat, base_weights, support_seman, base_seman)

        fakeclass_protos, recip_unit = self.open_generator(supp_protos, base_weights, support_seman, base_seman)
        cls_protos = torch.cat([supp_protos, fakeclass_protos], dim=1)
        
        query_cls_scores = self.metric(cls_protos, query_feat)
        openset_cls_scores = self.metric(cls_protos, openset_feat)

        test_cosine_scores = (query_cls_scores,openset_cls_scores)

        query_funit_distance = 1.0- self.metric(recip_unit, query_feat)
        qopen_funit_distance = 1.0- self.metric(recip_unit, openset_feat)
        funit_distance = torch.cat([query_funit_distance,qopen_funit_distance],dim=1)

        return test_cosine_scores, supp_protos, fakeclass_protos, (base_weights,base_wgtmem), funit_distance

    def init_representation(self, param_seman):
        (params,seman_dict) = param_seman
        self.weight_base = nn.Parameter(params['cls_classifier.weight'], requires_grad=self.train_weight_base)
        self.bias_base = nn.Parameter(params['cls_classifier.bias'], requires_grad=self.train_weight_base)
        self.weight_mem = nn.Parameter(params['cls_classifier.weight'].clone(), requires_grad=False)

        self.seman = {k:nn.Parameter(torch.from_numpy(v),requires_grad=False).float().cuda() for k,v in seman_dict.items()}
    
    def get_representation(self, cls_ids, base_ids, randpick=False):
        if base_ids is not None:
            base_weights = self.weight_base[base_ids,:]   ## bs*54*D
            base_wgtmem = self.weight_mem[base_ids,:]
            base_seman = self.seman['base'][base_ids,:]
            supp_seman = self.seman['base'][cls_ids,:]
        else:
            bs = cls_ids.size(0)
            base_weights = self.weight_base.repeat(bs,1,1)
            base_wgtmem = self.weight_mem.repeat(bs,1,1)
            base_seman = self.seman['base'].repeat(bs,1,1)
            supp_seman = self.seman['novel_test'][cls_ids,:]
        if randpick:
            num_base = base_weights.shape[1]
            base_size = self.base_size
            idx = np.random.choice(list(range(num_base)), size=base_size, replace=False)
            base_weights = base_weights[:, idx, :]
            base_seman = base_seman[:, idx, :]
        return base_weights,base_wgtmem,base_seman,supp_seman


class SupportCalibrator(nn.Module):
    def __init__(self, nway, feat_dim, n_head=1,base_seman_calib=True, neg_gen_type='semang'):
        super(SupportCalibrator, self).__init__()
        self.nway = nway
        self.feat_dim = feat_dim
        self.base_seman_calib = base_seman_calib

        self.map_sem = nn.Sequential(nn.Linear(300,300),nn.LeakyReLU(0.1),nn.Dropout(0.1),nn.Linear(300,300))

        self.calibrator = MultiHeadAttention(feat_dim//n_head, feat_dim//n_head, (feat_dim,feat_dim))

        self.neg_gen_type = neg_gen_type
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(feat_dim+300,feat_dim)
            self.task_semfuse = nn.Linear(feat_dim+300,300)

    def _seman_calib(self, seman):
        seman = self.map_sem(seman)
        return seman


    def forward(self, support_feat, base_weights, support_seman, base_seman):
        ## support_feat: bs*nway*640, base_weights: bs*num_base*640, support_seman: bs*nway*300, base_seman:bs*num_base*300        
        n_bs, n_base_cls = base_weights.size()[:2]

        base_weights = base_weights.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, self.feat_dim)

        support_feat = support_feat.view(-1,1,self.feat_dim)

        
        if self.neg_gen_type == 'semang':
            support_seman = self._seman_calib(support_seman)
            if self.base_seman_calib:
                base_seman = self._seman_calib(base_seman)

            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)

            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0

            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        elif self.neg_gen_type == 'attg':
            base_mem_vis = base_weights
            base_seman = None
            support_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_feat
            base_mem_vis = support_feat
            support_seman = None
            base_seman = None

        else:
            return support_feat.view(n_bs,self.nway,-1), None

        support_center, _, support_attn, _ = self.calibrator(support_feat, base_weights, base_mem_vis, support_seman, base_seman)

        support_center = support_center.view(n_bs,self.nway,-1)
        support_attn = support_attn.view(n_bs,self.nway,-1)
        return support_center, support_attn


class OpenSetGenerater(nn.Module):
    def __init__(self, nway, featdim, n_head=1, neg_gen_type='semang', agg='avg'):
        super(OpenSetGenerater, self).__init__()
        self.nway = nway
        self.att = MultiHeadAttention(featdim//n_head, featdim//n_head, (featdim,featdim))
        self.featdim = featdim

        self.neg_gen_type = neg_gen_type
        if neg_gen_type == 'semang':
            self.task_visfuse = nn.Linear(featdim+300,featdim)
            self.task_semfuse = nn.Linear(featdim+300,300)


        self.agg = agg
        if agg == 'mlp':
            self.agg_func = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            
        self.map_sem = nn.Sequential(nn.Linear(300,300),nn.LeakyReLU(0.1),nn.Dropout(0.1),nn.Linear(300,300))

    def _seman_calib(self, seman):
        ### feat: bs*d*feat_dim, seman: bs*d*300
        seman = self.map_sem(seman)
        return seman

    def forward(self, support_center, base_weights, support_seman=None, base_seman=None):
        ## support_center: bs*nway*D
        ## weight_base: bs*nbase*D
        bs = support_center.shape[0]
        n_bs, n_base_cls = base_weights.size()[:2]
        
        base_weights = base_weights.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, self.featdim)
        support_center = support_center.view(-1, 1, self.featdim)
        
        if self.neg_gen_type=='semang':
            support_seman = self._seman_calib(support_seman)
            base_seman = base_seman.unsqueeze(dim=1).repeat(1,self.nway,1,1).view(-1, n_base_cls, 300)
            support_seman = support_seman.view(-1, 1, 300)
            
            base_mem_vis = base_weights
            task_mem_vis = base_weights
            
            base_mem_seman = base_seman
            task_mem_seman = base_seman
            avg_task_mem = torch.mean(torch.cat([task_mem_vis,task_mem_seman],-1), 1, keepdim=True)

            gate_vis = torch.sigmoid(self.task_visfuse(avg_task_mem)) + 1.0
            gate_sem = torch.sigmoid(self.task_semfuse(avg_task_mem)) + 1.0
            
            base_weights = base_mem_vis * gate_vis 
            base_seman = base_mem_seman * gate_sem

        
        elif self.neg_gen_type == 'attg':
            base_mem_vis = base_weights
            support_seman = None
            base_seman = None

        elif self.neg_gen_type == 'att':
            base_weights = support_center
            base_mem_vis = support_center
            support_seman = None
            base_seman = None

        else:
            fakeclass_center = support_center.mean(dim=0, keepdim=True)
            if self.agg == 'mlp':
                fakeclass_center = self.agg_func(fakeclass_center)
            return fakeclass_center, support_center.view(bs, -1, self.featdim)
            

        output, attcoef, attn_score, value = self.att(support_center, base_weights, base_mem_vis, support_seman, base_seman)  ## bs*nway*nbase

        output = output.view(bs, -1, self.featdim)
        fakeclass_center = output.mean(dim=1,keepdim=True)
        
        if self.agg == 'mlp':
            fakeclass_center = self.agg_func(fakeclass_center)
        
        return fakeclass_center, output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_k, d_v, d_model, n_head=1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #### Visual feature projection head
        self.w_qs = nn.Linear(d_model[0], n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model[1], n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model[-1], n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model[0] + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model[1] + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model[-1] + d_v)))

        #### Semantic projection head #######
        self.w_qs_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_ks_sem = nn.Linear(300, n_head * d_k, bias=False)
        self.w_vs_sem = nn.Linear(300, n_head * d_k, bias=False)
        
        nn.init.normal_(self.w_qs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_ks_sem.weight, mean=0, std=np.sqrt(2.0 / 600))
        nn.init.normal_(self.w_vs_sem.weight, mean=0, std=np.sqrt(2.0 / 600))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))            

        self.fc = nn.Linear(n_head * d_v, d_model[0], bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, q_sem=None, k_sem=None, mark_res=True):
        ### q: bs*nway*D
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        if q_sem is not None:
            sz_b, len_q, _ = q_sem.size()
            sz_b, len_k, _ = k_sem.size()
            q_sem = self.w_qs_sem(q_sem).view(sz_b, len_q, n_head, d_k)
            k_sem = self.w_ks_sem(k_sem).view(sz_b, len_k, n_head, d_k)
            q_sem = q_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) 
            k_sem = k_sem.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) 

        output, attn, attn_score = self.attention(q, k, v, q_sem, k_sem)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        if mark_res:
            output = output + residual

        return output, attn, attn_score, v


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, q_sem = None, k_sem = None):

        attn_score = torch.bmm(q, k.transpose(1, 2))

        if q_sem is not None:
            attn_sem = torch.bmm(q_sem, k_sem.transpose(1, 2))
            q = q + q_sem
            k = k + k_sem
            attn_score = torch.bmm(q, k.transpose(1, 2))
        
        attn_score /= self.temperature
        attn = self.softmax(attn_score)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn, attn_score
        

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
    
    
