import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as DataLoader

from model import MyModel
from util.causalData import CausalData
from util.visual import AllPlot
from util.criteria import AllCriterion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    dt = pd.read_csv(PARAM['dt_path'])
    Tf = np.array(dt['T']).reshape(-1,1) # (N, 1)
    col_pre = ['pre_'+str(i) for i in range(1,PARAM['pretreat_len']+1)]
    col_post = ['post_'+str(i) for i in range(1,PARAM['posttreat_len']+1)]
    col_counterpost = ['counterpost_' + str(i) for i in range(1, PARAM['posttreat_len']+1)]
    preYf, postYf, postYcf = np.array(dt[col_pre]), np.array(dt[col_post]), np.array(dt[col_counterpost])   # (N, pretime_len), (N, posttime_len), (N, posttime_len)

    # create counterfactual
    Tcf = Tf.copy()
    Tcf[np.where(Tf == 0)] = 1
    Tcf[np.where(Tf == 1)] = 0

    dt_train = CausalData(preYf, postYf, Tf)
    trainset = DataLoader.DataLoader(dt_train, batch_size=PARAM['batch_size'], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myModel = MyModel(PARAM).to(device)
    optimizer = torch.optim.Adam(myModel.parameters(), lr=PARAM['lr'])

    criterion = AllCriterion(PARAM)
    allplot = AllPlot(figsize=PARAM['figsize'])

    # training
    def train():
        for epoch in range(PARAM['epoch']):
            myModel.train()
            total_loss = []
            for i, (preYf, postYf, Tf) in enumerate(trainset):
                # (batch, pretime_len), (batch, posttime_len), (batch, 1)
                preYf, postYf, Tf = preYf.to(device), postYf.to(device), Tf.to(device)
                optimizer.zero_grad()

                Tf0_idx = np.array(np.argwhere(Tf == 0)[0])
                Tf1_idx = np.array(np.argwhere(Tf == 1)[0])

                postYf0 = postYf[Tf0_idx]  # (batch[t==1], posttime_len)
                postYf1 = postYf[Tf1_idx]  # (batch[t==0], posttime_len)

                pred0, pred1, pred_t0, pred_t1 = myModel(preYf, Tf0_idx, Tf1_idx)
                loss = criterion.propensity_loss(pred0, pred1, postYf0, postYf1, pred_t0, pred_t1)
                loss.backward()
                optimizer.step()
                total_loss.append(loss.detach().numpy())

            if epoch % PARAM['print_every_epoch'] == 0:
                print("epoch:", epoch, "loss:", np.mean(total_loss))

    # testing
    def test(preYf, postYcf, postYf, Tf, Tcf):
        print("testing the model")
        myModel.eval()
        # with torch.no_grad():
        #     for preYf, postYcf, Tcf in testset:
        preYf, postYcf, postYf, Tf, Tcf = torch.Tensor(preYf), torch.Tensor(postYcf), torch.Tensor(postYf), torch.Tensor(Tf),torch.Tensor(Tcf)
        Tcf0_idx = np.array(np.argwhere(Tcf == 0)[0])
        Tcf1_idx = np.array(np.argwhere(Tcf == 1)[0])
        Tf0_idx = np.array(np.argwhere(Tf == 0)[0])
        Tf1_idx = np.array(np.argwhere(Tf == 1)[0])

        postYcf0 = postYcf[Tcf0_idx]  # (N[t==0], posttime_len)
        postYcf1 = postYcf[Tcf1_idx]  # (N[t==1], posttime_len)
        postYf0 = postYf[Tf0_idx]  # (N[t==1], posttime_len)
        postYf1 = postYf[Tf1_idx]  # (N[t==0], posttime_len)

        # pred0: actually receive T=1, predict counter factual T=0
        pred0, pred1, _, _ = myModel(preYf, Tcf0_idx, Tcf1_idx) # pred: # (N[t==j], posttime_len)

        # all after rearangement
        preds = torch.cat([pred0, pred1], dim=0)  # (batch, posttime_len)
        postYcf_arr = torch.cat([postYcf0, postYcf1], dim=0)  #
        cf_mse = criterion.cf_mse(preds, postYcf_arr).detach().numpy()
        ate_ae = criterion.ate_mse(pred0, pred1, postYcf0, postYcf1, postYf1, postYf0).detach().numpy()

        print("Evaluating metrics \n"
              "cf_mse: {:.3f}\n"
              "ate_ae: {:.3f}".format(cf_mse, ate_ae))

        if PARAM['is_plot']==True:
            allplot.timeplot(pred0[0].detach().numpy(), postYcf0[0])


    train()
    test(preYf, postYcf, postYf, Tf, Tcf)


PARAM = {
    # data loading param
    'dt_path': 'time_dt.csv',
    'pretreat_len': 120,
    'posttreat_len': 20,

    # model hyperparam
    'Hidden': 4,

    # regularization
    'pro_reg': 0.05,

    # optimizing param
    'batch_size': 64,
    'lr': 0.01,
    'epoch': 40,
    'print_every_epoch': 2,

    # plot
    'is_plot': True,
    'figsize': (8,6),
}

SEED = 100

if __name__ == "__main__":
    set_seed(SEED)
    main()