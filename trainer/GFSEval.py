from __future__ import print_function

import sys, os, pdb 
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import f1_score


def run_test_gfsl(net, genopenloader): 
    net = net.eval()
    acc_base,acc_novel,acc_ball,acc_nall,acc_gen = [],[],[],[],[]
    acc_sepa,acc_mean = [],[]
    acc_delta = []
    auroc_gen_prob,auroc_gen_diff = [],[]
    auroc_f1score = []
    with torch.no_grad():
        with tqdm(genopenloader, total=len(genopenloader), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                # Data Preparation
                support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, baseset_data, baseset_label, supp_idx, open_idx = data

                num_query,num_open,num_base = query_label.size(1),openset_label.size(1),baseset_label.size(1)
                assert support_data.size(0) == 1
                num_base_cls, num_novel_cls = baseset_label.max().item()+1, net.n_ways

                support_data,support_label              = support_data.float().cuda(),support_label.cuda().long()
                query_data,query_label                  = query_data.float().cuda(),query_label.cuda().long()
                suppopen_data,suppopen_label            = suppopen_data.float().cuda(),suppopen_label.cuda().long()
                openset_data,openset_label              = openset_data.float().cuda(),openset_label.cuda().long()
                baseset_data,baseset_label              = baseset_data.float().cuda(),baseset_label.cuda().long()
                supp_idx,open_idx                       = supp_idx.long().cuda(),open_idx.long().cuda()
                openset_label = num_novel_cls * torch.ones_like(openset_label)

                the_img     = (support_data, query_data, suppopen_data, openset_data, baseset_data)
                the_label   = (support_label,query_label,suppopen_label,openset_label,baseset_label)
                num_baseclass = baseset_label.max()+1
                the_conj    = (supp_idx-num_baseclass, open_idx-num_baseclass)

                test_feats, cls_protos, test_cls_scores = net(the_img,the_label,the_conj,None,True)
                (baseset_cls_scores,query_cls_scores,openset_cls_scores) = test_cls_scores
                (support_feat, query_feat, openset_feat, baseset_feat) = test_feats

                # scores_gen = torch.mm(features_eval, centers_all.transpose(0,1))
                scores_gen = torch.cat([baseset_cls_scores[0],query_cls_scores[0],openset_cls_scores[0]],dim=0)
                probs_gen_plus = F.softmax(scores_gen,dim=-1).cpu().numpy()
                probs_gen_max = F.softmax(scores_gen[:,:num_base_cls+num_novel_cls],dim=-1).cpu().numpy()

                novel_label = query_label.view(-1).cpu().numpy()
                base_label = baseset_label.view(-1).cpu().numpy()
                open_label_binary = np.concatenate((np.ones(num_base+num_query),np.zeros(num_open)))
                general_label = np.concatenate([base_label,novel_label+base_label.max()+1],axis=0)

                acc_ball.append(metrics.accuracy_score(general_label[:num_base], np.argmax(probs_gen_max[:num_base],-1)))
                acc_nall.append(metrics.accuracy_score(general_label[num_base:], np.argmax(probs_gen_max[num_base:num_base+num_query],-1)))
                acc_gen.append(2*acc_ball[-1]*acc_nall[-1]/(acc_ball[-1]+acc_nall[-1])) ## harmonic mean
                acc_mean.append((acc_ball[-1]+acc_nall[-1])/2) ## arithmetic mean
                acc_base.append(metrics.accuracy_score(base_label, np.argmax(probs_gen_max[:num_base,:-net.n_ways],-1)))
                acc_novel.append(metrics.accuracy_score(novel_label, np.argmax(probs_gen_max[num_base:num_base+num_query,-net.n_ways:],-1)))
                acc_sepa.append((acc_base[-1]+acc_novel[-1])/2)
                acc_delta.append(0.5*(acc_base[-1]+acc_novel[-1]-acc_ball[-1]-acc_nall[-1]))

                auroc_gen_prob.append(metrics.roc_auc_score(1-open_label_binary,probs_gen_plus[:,-1]))
                auroc_gen_diff.append(metrics.roc_auc_score(1-open_label_binary,probs_gen_plus[:,-1]-probs_gen_plus[:,:-1].max(axis=-1)))
                all_labels = np.concatenate([general_label, (general_label.max()+1) * np.ones(num_open)], -1).astype(np.int)
                ypred = np.argmax(probs_gen_plus, axis=-1)
                auroc_f1score.append(f1_score(all_labels, ypred, average='macro'))

                pbar.set_postfix({
                    "OpenSet MetaEval Acc":'{0:.2f}'.format(acc_gen[-1]),
                    "ROC":'{0:.2f}'.format(auroc_gen_diff[-1]),
                    "Gen Acc":'{0:.2f}'.format(acc_gen[-1])
                })

            acc = {'base':mean_confidence_interval(acc_ball),'novel':mean_confidence_interval(acc_nall),'gen':mean_confidence_interval(acc_gen)}
            acc_aux = {'bb':mean_confidence_interval(acc_base),'nn':mean_confidence_interval(acc_novel),'sepa_mean':mean_confidence_interval(acc_sepa),'delta':mean_confidence_interval(acc_delta),'all_mean':mean_confidence_interval(acc_mean)}
            auroc_nplus = {'prob':mean_confidence_interval(auroc_gen_prob),'diff':mean_confidence_interval(auroc_gen_diff), 'f1':mean_confidence_interval(auroc_f1score)}
            
    return acc_aux['all_mean'], acc['gen'], acc_aux['delta'], auroc_nplus['prob'], auroc_nplus['f1']
    #return acc['novel']+acc['base']+acc['gen'], auroc_nplus['prob']+auroc_nplus['diff']+auroc_nplus['f1'], acc_aux['bb']+acc_aux['nn']+acc_aux['sepa_mean']+acc_aux['delta']+acc_aux['all_mean']

def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h
