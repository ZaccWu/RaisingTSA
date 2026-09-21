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
from util.evaluate import transfer_pred, ev_loss, pairwise_ranking_loss, cal_ndcgK
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
## TODO: 可能要更改ev_loss

def _configTrainArgs():
    parser = argparse.ArgumentParser('Raising star prediction: Commonality and individuality')
    parser.add_argument('--model', type=str, help='sinkhorn type', default='rai') # 'rai', 'raisp'
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
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def transfer_pred(out, threshold):
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    return pred

@torch.no_grad()
def evalInBatches(model, data_loader, device, return_loss=True):
    model.eval()
    preds_list, y_list, y_bin_list, prds_list = [], [], [], []
    loss_sum, n_sample = 0.0, 0
    for x_b, y_b in data_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)
        pred, _, prob = model(x_b)
        prd_select = prob.argmax(dim=-1)

        if return_loss:
            loss_b = ev_loss(pred, y_b)
            loss_sum += loss_b.item() * x_b.size(0)
            n_sample += x_b.size(0)

        preds_list.append(pred.detach().cpu())
        y_list.append(y_b.detach().cpu())
        prds_list.append(prd_select.detach().cpu())

    preds_all = torch.cat(preds_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    prds_all = torch.cat(prds_list, dim=0)
    avg_loss = (loss_sum / max(n_sample, 1)) if return_loss else None
    return preds_all, y_all, prds_all, avg_loss

def train(args):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    data_loader = LoadAliDt()
    trDt, vaDt, tsDt = data_loader.loadTrainTest()
    trDt_loader = DataLoader(trDt, batch_size=args.bs, shuffle=True)
    vaDt_loader = DataLoader(vaDt, batch_size=args.bs, shuffle=False)
    tsDt_loader = DataLoader(tsDt, batch_size=args.bs, shuffle=False)

    if args.model == 'rai':
        model = Raise(in_dim=trDt.x.shape[-1], h_dim=args.h_dim, num_states=args.ns).to(device)
    elif args.model == 'raisp':
        model = RaiseSep(in_dim=trDt.x.shape[-1], h_dim=args.h_dim, num_states=args.ns).to(device)
    else:
        assert ValueError('Model not specified')
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
            pred, all_preds, prob = model(x_b) # all_preds, prob: (B, num_states)
            cla_loss = ev_loss(pred, y_b)
            #rank_loss = pairwise_ranking_loss(pred, y_b)
            pred_loss = cla_loss #+ args.beta * rank_loss

            if prob is not None:
                L = ev_loss(all_preds, y_b)          # (B, num_states)
                L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input

                if args.ot == 'full':
                    P = sinkhorn(-L, epsilon=0.1) 
                else:
                    P = partial_sinkhorn(-L, epsilon=0.1, tai=args.tai)
                lamb = args.lamb * (args.rho ** global_step) # lamb=0 still have multi-expert
                ot_loss = prob.log().mul(P).sum(dim=-1).mean()
                loss = pred_loss - lamb * ot_loss
            loss.backward()
            optimizer.step()
        
        model.training = False
        va_pred, va_y, va_prds, va_loss = evalInBatches(model, vaDt_loader, device)
        # group_size = data_loader.num_stock  # 每个时间步的股票数
        # NK = int(group_size*0.1)
        va_rec_r2 = transfer_pred(va_pred, torch.quantile(va_pred, 0.9, dim=None, keepdim=False))



        rec_val = classification_report(va_y.numpy(), va_rec_r2.numpy(), target_names=['class0', 'class1'],
                                output_dict=True)['class1']['recall']
        va_score = rec_val

        print(' Epoch {}, va_loss {:.4f},  va_score {:.4f}'.format(i, va_loss, va_score))
        #print('Predictors: ', pd.Series(va_prds.numpy()).value_counts())
        if va_score > best_va_score:
            best_va_score = va_score
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_id = i
    final_model_state = copy.deepcopy(model.state_dict())
    
    model.training = False
    model.load_state_dict(best_model_state)
    ts_pred, ts_y, ts_prds, _ = evalInBatches(model, tsDt_loader, device, return_loss=False)

    ts_rec_r1 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.95, dim=None, keepdim=False))
    ts_rec_r2 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.9, dim=None, keepdim=False))
    ts_rec_r3 = transfer_pred(ts_pred, torch.quantile(ts_pred, 0.8, dim=None, keepdim=False))
    r1_rec = classification_report(ts_y.numpy(), ts_rec_r1.numpy(), target_names=['class0', 'class1'],
                            output_dict=True)['class1']['recall']
    r2_rec = classification_report(ts_y.numpy(), ts_rec_r2.numpy(), target_names=['class0', 'class1'],
                            output_dict=True)['class1']['recall']
    r3_rec = classification_report(ts_y.numpy(), ts_rec_r3.numpy(), target_names=['class0', 'class1'],
                            output_dict=True)['class1']['recall']
    auc = roc_auc_score(ts_y.numpy(),ts_pred.numpy())

    _, pred_r1 = torch.topk(ts_pred.detach(), k=len(np.nonzero(ts_rec_r1.numpy())[0]))  # pred_r1: (k_rec_content)
    _, pred_r2 = torch.topk(ts_pred.detach(), k=len(np.nonzero(ts_rec_r2.numpy())[0]))
    _, pred_r3 = torch.topk(ts_pred.detach(), k=len(np.nonzero(ts_rec_r3.numpy())[0]))

    r1_ndcg = cal_ndcgK(np.nonzero(ts_y.numpy())[0], pred_r1.numpy())
    r2_ndcg = cal_ndcgK(np.nonzero(ts_y.numpy())[0], pred_r2.numpy())
    r3_ndcg = cal_ndcgK(np.nonzero(ts_y.numpy())[0], pred_r3.numpy())

    print('Best epoch: ', best_epoch_id, 'Predictors: ', pd.Series(ts_prds.numpy()).value_counts())
    print('r1_rec {:3f},'.format(r1_rec),
        'r2_rec {:3f},'.format(r2_rec),
        'r3_rec {:3f},'.format(r3_rec),
        'r1_ndcg {:3f},'.format(r1_ndcg),
        'r2_ndcg {:3f},'.format(r2_ndcg),
        'r3_ndcg {:3f},'.format(r3_ndcg),
        'AUC {:3f},'.format(auc))

    return {
        'r1_rec':  r1_rec,
        'r2_rec':  r2_rec,
        'r3_rec':  r3_rec,
        'r1_ndcg': r1_ndcg,
        'r2_ndcg': r2_ndcg,
        'r3_ndcg': r3_ndcg,
        'auc':     auc,
    }

if __name__ == "__main__":
    args = _configTrainArgs()
    repeat_res = {
        'r1_rec': [], 'r2_rec': [], 'r3_rec': [],
        'r1_ndcg': [], 'r2_ndcg': [], 'r3_ndcg': [],
        'auc': [],
    }
    for seed in range(101,111):
        set_seed(seed)
        #ts_res, ts_res_f = train(args)
        ts_res = train(args)
        repeat_res['r1_rec'].append(ts_res['r1_rec'])
        repeat_res['r2_rec'].append(ts_res['r2_rec'])
        repeat_res['r3_rec'].append(ts_res['r3_rec'])
        repeat_res['r1_ndcg'].append(ts_res['r1_ndcg'])
        repeat_res['r2_ndcg'].append(ts_res['r2_ndcg'])
        repeat_res['r3_ndcg'].append(ts_res['r3_ndcg'])
        repeat_res['auc'].append(ts_res['auc'])

    print('All results | model: ', args.model,  'ns: ', args.ns, 'lamb', args.lamb)
    print('AUC {:.4f} ({:.3f}), '.format(np.mean(repeat_res['auc']), np.std(repeat_res['auc'])),
          ' R@1 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r1_rec']), np.std(repeat_res['r1_rec'])),
          ' R@2 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r2_rec']), np.std(repeat_res['r2_rec'])),
          ' R@3 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r3_rec']), np.std(repeat_res['r3_rec'])),
          ' N@1 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r1_ndcg']), np.std(repeat_res['r1_ndcg'])),
          ' N@2 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r2_ndcg']), np.std(repeat_res['r2_ndcg'])),
          ' N@3 {:.4f} ({:.3f}), '.format(np.mean(repeat_res['r3_ndcg']), np.std(repeat_res['r3_ndcg'])),)
    

