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
from util.evaluate import focal_loss
from sklearn.metrics import roc_auc_score, average_precision_score

def _configTrainArgs():
    parser = argparse.ArgumentParser('Raising star prediction: Commonality and individuality')
    parser.add_argument('--ns', type=int, help='num of state', default=3)
    parser.add_argument('--rho', type=float, help='rho', default=0.99) # default 0.99
    parser.add_argument('--lamb', type=float, help='rho', default=1) # default 1
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3) # default 1e-3

    parser.add_argument('--h_dim', type=int, help='dimension of the seq emb', default=16) # 64

    parser.add_argument('--bs', type=int, help='batch size', default=8192)
    parser.add_argument('--n_epoch', type=int, help='number of epochs', default=10)
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

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_loader = LoadAliDt()
    trDt, vaDt, tsDt = data_loader.loadTrainTest()
    trDt_loader = DataLoader(trDt, batch_size=args.bs, shuffle=True)

    model = Raise(in_dim=trDt.x.shape[-1], h_dim=args.h_dim, num_states=args.ns).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = -1
    best_va_score = -np.inf
    ts_res = {}
    for i in range(args.n_epoch):
        model_training = True
        global_step += 1
        for batch_idx, (x_b, y_b) in enumerate(trDt_loader):
            #print(trDt_b)
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
        
        model_training = False
        va_pred, _, _ = model(vaDt.x)
        va_loss = focal_loss(va_pred, vaDt.y)
        print(' Epoch {}, va_loss {:.4f}, '.format(i, va_loss))
        va_score = average_precision_score(vaDt.y.detach().numpy(), va_pred.detach().numpy())
        if va_score > best_va_score:
            best_va_score = va_score
            best_model = model
        
    ts_pred, _, _ = best_model(tsDt.x)
    ts_auc = roc_auc_score(tsDt.y.detach().numpy(), ts_pred.detach().numpy())
    ts_auprc = average_precision_score(tsDt.y.detach().numpy(), ts_pred.detach().numpy())
    ts_res['auc'], ts_res['auprc'] = ts_auc, ts_auprc
    return ts_res

if __name__ == "__main__":
    args = _configTrainArgs()
    set_seed(args.seed)
    ts_res = train(args)

    print(' AUC {:.4f}, '.format(np.mean(ts_res['auc'])),
          ' PRAUC {:.4f}, '.format(np.mean(ts_res['pr-auc']))
    )
