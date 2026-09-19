import random
import pandas as pd
import numpy as np
import math
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
from RaiseModel import Raise, sinkhorn, partial_sinkhorn
from load_data import LoadAliDt
from util.evaluate import focal_loss, transfer_pred, huber_loss, pairwise_ranking_loss
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

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
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=100)
    parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
    parser.add_argument('--seed', type=int, help='random', default=101)
    return parser.parse_args()

LOG_THRESHOLD = math.log(2.21 + 1)

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

    model = Raise(in_dim=trDt.x.shape[-1], h_dim=args.h_dim, num_states=args.ns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = -1
    best_va_score, best_epoch_id = -np.inf, 0
    ts_res = {}
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
                    P = partial_sinkhorn(L, epsilon=0.1, tai=args.tai)
                lamb = args.lamb * (args.rho ** global_step)
                ot_loss = prob.log().mul(P).sum(dim=-1).mean()
                loss = pred_loss - lamb * ot_loss
                #print(reg_loss.detach(),  rank_loss.detach(), (-ot_loss).detach())
            loss.backward()
            optimizer.step()
        
        model.training = False
        va_pred_all, _, va_y_all, va_prds_all, va_loss = evalInBatches(model, vaDt_loader, device)
        #va_score = average_precision_score(va_y_all.numpy(), va_pred_all.numpy())
        r1 = transfer_pred(va_pred_all, torch.quantile(va_pred_all, 0.9))
        class_rep = classification_report(
            va_y_all.numpy().astype(int), r1.numpy().astype(int),
            output_dict=True, zero_division=0)
        va_score = class_rep.get('1', {}).get('recall', 0.0)
        print(' Epoch {}, va_loss {:.4f},  va_score {:.4f}'.format(i, va_loss, va_score))
        print('Predictors: ', pd.Series(va_prds_all.numpy()).value_counts())
        if va_score > best_va_score:
            best_va_score = va_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_id = i
        final_model = model
    
    model.training = False
    #ts_pred, _, ts_y, ts_prds, _ = evalInBatches(final_model, tsDt_loader, device, return_loss=False)

    model.load_state_dict(best_model_state)
    ts_pred, _, ts_y, ts_prds, _ = evalInBatches(model, tsDt_loader, device, return_loss=False)
    print('Best epoch: ', best_epoch_id, 'Predictors: ', pd.Series(ts_prds.numpy()).value_counts())


    ts_auc = roc_auc_score(ts_y.numpy(), ts_pred.numpy())
    ts_auprc = average_precision_score(ts_y.numpy(), ts_pred.numpy())

    r1 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.9))
    class_rep = classification_report(
        ts_y.numpy().astype(int), r1.numpy().astype(int),
        output_dict=True, zero_division=0)
    ts_prec1 = class_rep.get('1', {}).get('precision', 0.0)
    ts_rec1  = class_rep.get('1', {}).get('recall', 0.0)
    ts_f1    = class_rep.get('1', {}).get('f1-score', 0.0)

    ts_res['auc'], ts_res['auprc'] = ts_auc, ts_auprc
    ts_res['rec'], ts_res['prec'], ts_res['f1'] = ts_rec1, ts_prec1, ts_f1
    return ts_res

if __name__ == "__main__":
    args = _configTrainArgs()
    set_seed(args.seed)
    ts_res = train(args)

    print(' AUC {:.4f}, '.format(ts_res['auc']),
          ' PRAUC {:.4f}, '.format(ts_res['auprc']),
          ' PREC-1 {:.4f}, '.format(ts_res['prec']),
          ' REC-1 {:.4f}, '.format(ts_res['rec']),
          ' F1-1 {:.4f}, '.format(ts_res['f1']),)
    print('ns: ', args.ns, 'lamb', args.lamb)

