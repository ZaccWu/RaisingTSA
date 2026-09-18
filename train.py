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

from RaiseModel import Raise, sinkhorn
from load_data import LoadAliDt
from util.evaluate import focal_loss, transfer_pred
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def _configTrainArgs():
    parser = argparse.ArgumentParser('Raising star prediction: Commonality and individuality')
    parser.add_argument('--ns', type=int, help='num of state', default=3)
    parser.add_argument('--rho', type=float, help='rho', default=0.99) # default 0.99
    parser.add_argument('--lamb', type=float, help='rho', default=1) # default 1
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3) # default 1e-3

    parser.add_argument('--h_dim', type=int, help='dimension of the seq emb', default=32) # 64

    parser.add_argument('--bs', type=int, help='batch size', default=8192) # cpu: 8192, gpu: 2048
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=50)
    parser.add_argument('--gpu', type=int, help='idx for the gpu to use', default=0)
    parser.add_argument('--seed', type=int, help='random', default=101)
    return parser.parse_args()


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
    preds_list, labels_list = [], []
    loss_sum, n_sample = 0.0, 0
    for x_b, y_b in data_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        pred, _, _ = model(x_b)
        if return_loss:
            loss_b = focal_loss(pred, y_b)
            loss_sum += loss_b.item() * x_b.size(0)
            n_sample += x_b.size(0)
        preds_list.append(pred.detach().cpu())
        labels_list.append(y_b.detach().cpu())

    preds_all = torch.cat(preds_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)
    avg_loss = (loss_sum / max(n_sample, 1)) if return_loss else None
    return preds_all, labels_all, avg_loss

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
    best_va_score = -np.inf
    ts_res = {}
    for i in range(args.n_epoch):
        model.training = True
        model.train()
        global_step += 1
        for batch_idx, (x_b, y_b) in enumerate(trDt_loader):
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred, all_preds, prob = model(x_b)
            loss = focal_loss(pred, y_b)
            L = focal_loss(all_preds, y_b[:, None], reduction=None)
            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = args.lamb * (args.rho ** global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg
            loss.backward()
            optimizer.step()
        
        model.training = False
        va_pred_all, va_y_all, va_loss = evalInBatches(model, vaDt_loader, device)
        va_score = average_precision_score(va_y_all.numpy(), va_pred_all.numpy())
        print(' Epoch {}, va_loss {:.4f},  va_score {:.4f}'.format(i, va_loss, va_score))
        if va_score > best_va_score:
            best_va_score = va_score
            best_model = model
    
    model.training = False
    ts_pred, ts_y, _ = evalInBatches(best_model, tsDt_loader, device, return_loss=False)
    ts_auc = roc_auc_score(ts_y.numpy(), ts_pred.numpy())
    ts_auprc = average_precision_score(ts_y.numpy(), ts_pred.numpy())
    r1 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.9))
    class_rep = classification_report(ts_y.numpy(), r1.numpy(), output_dict=True)
    ts_rec1 = class_rep['1.0']['recall']
    ts_prec1 = class_rep['1.0']['precision']      # 异常类精确率（若需查看）
    ts_f1 = class_rep['1.0']['f1-score']          # 异常类 F1（新增）

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

