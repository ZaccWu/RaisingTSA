import random
import pandas as pd
import numpy as np
import math
import argparse
import os
from sympy import KroneckerDelta
import torch
from torch._inductor.codegen.wrapper import KernelCallLine
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from RaiseModel import Raise, RaiseSep, sinkhorn, partial_sinkhorn
from load_data import LoadAliDt
from util.evaluate import transfer_pred, huber_loss, pairwise_ranking_loss, topk_recall, ndcg_at_k
from sklearn.metrics import roc_auc_score, average_precision_score

## TODO: 可能要更改ev_loss

def _configTrainArgs():
    parser = argparse.ArgumentParser('Raising star prediction: Commonality and individuality')

    parser.add_argument('--ot', type=str, help='sinkhorn type', default='partial')
    parser.add_argument('--ns', type=int, help='num of state', default=3)
    parser.add_argument('--rho', type=float, help='rho', default=0.99) # default 0.99
    parser.add_argument('--tai', type=float, default=0.1, help='entropic tai for partial OT') # default 0.1
    parser.add_argument('--h_dim', type=int, help='dimension of the seq emb', default=64) # 64

    parser.add_argument('--beta', type=float, help='beta', default=1) # default 1
    parser.add_argument('--lamb', type=float, help='rho', default=1) # default 1
    parser.add_argument('--lr', type=float, help='learning rate', default=0.01) # default 0.01


    parser.add_argument('--bs', type=int, help='batch size', default=8192) # cpu: 8192, gpu: 2048
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=30)
    parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
    return parser.parse_args()

LOG_THRESHOLD = math.log(1. + 1) # 1.21

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evalInBatches(model, data_loader, device, return_loss=True):
    model.eval()
    preds_list, y_cont_list, y_bin_list, prds_list = [], [], [], []
    loss_sum, n_sample = 0.0, 0
    for x_b, y_b in data_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        pred, _, prob = model(x_b)
        prd_select = prob.argmax(dim=-1)

        if return_loss:
            loss_b = huber_loss(pred, y_b)
            loss_sum += loss_b.item() * x_b.size(0)
            n_sample += x_b.size(0)

        preds_list.append(pred.detach().cpu())
        y_cont_list.append(y_b.detach().cpu())
        y_bin_list.append((y_b >= LOG_THRESHOLD).float().detach().cpu())
        prds_list.append(prd_select.detach().cpu())

    preds_all = torch.cat(preds_list, dim=0)
    y_cont_all = torch.cat(y_cont_list, dim=0)
    y_bin_all = torch.cat(y_bin_list, dim=0)
    prds_all = torch.cat(prds_list, dim=0)
    avg_loss = (loss_sum / max(n_sample, 1)) if return_loss else None
    return preds_all, y_cont_all, y_bin_all, prds_all, avg_loss

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_loader = LoadAliDt()
    trDt, vaDt, tsDt = data_loader.loadTrainTest()
    trDt_loader = DataLoader(trDt, batch_size=args.bs, shuffle=True)
    vaDt_loader = DataLoader(vaDt, batch_size=args.bs, shuffle=False)
    tsDt_loader = DataLoader(tsDt, batch_size=args.bs, shuffle=False)

    model = RaiseSep(in_dim=trDt.x.shape[-1], h_dim=args.h_dim, num_states=args.ns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = -1
    best_va_score, best_epoch_id = -np.inf, 0
    
    for i in range(args.n_epoch):
        model.training = True
        model.train()
        global_step += 1
        for batch_idx, (x_b, y_b) in enumerate(trDt_loader):
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred, all_preds, prob = model(x_b)

            # (1) 主损失：Huber 回归 + 成对排序
            reg_loss = huber_loss(pred, y_b)
            rank_loss = pairwise_ranking_loss(pred, y_b)
            
            pred_loss = reg_loss + args.beta * rank_loss
            if prob is not None:
                L = huber_loss(all_preds, y_b[:, None].expand_as(all_preds), reduction='none')          # (B, num_states)
                L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

                if args.ot == 'full':
                    P = sinkhorn(-L, epsilon=0.1) 
                else:
                    P = partial_sinkhorn(-L, epsilon=0.1, tai=args.tai)
                lamb = args.lamb * (args.rho ** global_step) # lamb=0 still have multi-expert
                ot_loss = prob.log().mul(P).sum(dim=-1).mean()
                loss = pred_loss - lamb * ot_loss
                #print(reg_loss.detach(),  rank_loss.detach(), (-ot_loss).detach())
            loss.backward()
            optimizer.step()
        
        model.training = False
        va_pred, _, va_y_bin, va_prds, va_loss = evalInBatches(model, vaDt_loader, device)
        group_size = data_loader.num_stock  # 每个时间步的股票数
        NK = int(group_size*0.1)
        va_recall_k = topk_recall(va_y_bin, va_pred, NK, group_size)
        va_ndcg_k  = ndcg_at_k(va_y_bin, va_pred, NK, group_size)
        va_score = va_recall_k

        #print(' Epoch {}, va_loss {:.4f},  va_score {:.4f}'.format(i, va_loss, va_score))
        #print('Predictors: ', pd.Series(va_prds.numpy()).value_counts())
        if va_score > best_va_score:
            best_va_score = va_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_id = i
    final_model_state = copy.deepcopy(model.state_dict())
    
    model.training = False
    model.load_state_dict(best_model_state)
    ts_pred, _, ts_y_bin, ts_prds, _ = evalInBatches(model, tsDt_loader, device, return_loss=False)
    print('Best epoch: ', best_epoch_id, 'Predictors: ', pd.Series(ts_prds.numpy()).value_counts())

    ts_recall_k = topk_recall(ts_y_bin, ts_pred, NK, group_size)
    ts_ndcg_k   = ndcg_at_k(ts_y_bin, ts_pred, NK, group_size)
    ts_auc = roc_auc_score(ts_y_bin.numpy(), ts_pred.numpy())
    ts_auprc = average_precision_score(ts_y_bin.numpy(), ts_pred.numpy())
    return {'auc':ts_auc, 'auprc':ts_auprc, 'recK':ts_recall_k, 'ndcgK':ts_ndcg_k}

if __name__ == "__main__":
    args = _configTrainArgs()
    repeat_res = {
        'auc': [], 'auprc': [], 'recK': [], 'ndcgK': [],
    }
    for seed in range(101,111):
        set_seed(seed)
        #ts_res, ts_res_f = train(args)
        ts_res = train(args)
        print(' seed {}, '.format(seed),
            ' | BestEval: AUROC {:.4f}, '.format(ts_res['auc']),
            ' PRAUC {:.4f}, '.format(ts_res['auprc']),
            ' REC-10% {:.4f}, '.format(ts_res['recK']),
            ' NDCG-10% {:.4f}, '.format(ts_res['ndcgK']),)
        repeat_res['auc'].append(ts_res['auc'])
        repeat_res['auprc'].append(ts_res['auprc'])
        repeat_res['recK'].append(ts_res['recK'])
        repeat_res['ndcgK'].append(ts_res['ndcgK'])

    print('All results | ns: ', args.ns, 'lamb', args.lamb)
    print('AUROC {:.4f} ({:.3f}), '.format(np.mean(repeat_res['auc']), np.std(repeat_res['auc'])),
    ' PRAUC {:.4f} ({:.3f}), '.format(np.mean(repeat_res['auprc']), np.std(repeat_res['auprc'])),
    ' REC-10% {:.4f} ({:.3f}), '.format(np.mean(repeat_res['recK']), np.std(repeat_res['recK'])),
    ' NDCG-10% {:.4f} ({:.3f}), '.format(np.mean(repeat_res['ndcgK']), np.std(repeat_res['ndcgK'])),)
    

