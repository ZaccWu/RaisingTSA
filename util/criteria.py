import random
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F


class AllCriterion():
    def __init__(self, PARAM):
        self.pro_reg = PARAM['pro_reg']

    def propensity_loss(self, pred0, pred1, postYf0, postYf1, pred_t0, pred_t1):
        assert len(pred0)==len(postYf0), "lenght not match!"
        assert len(pred1) == len(postYf1), "lenght not match!"
        preds = torch.cat([pred0, pred1],dim=0)  # (batch, posttime_len)
        posttime = torch.cat([postYf0, postYf1], dim=0)  # (batch, posttime_len)

        t0 = torch.zeros(pred_t0.shape)
        t1 = torch.ones(pred_t1.shape)

        return F.mse_loss(preds, posttime) + self.pro_reg * (F.mse_loss(pred_t0, t0) + F.mse_loss(pred_t1, t1))

    def cf_mse(self, preds, postYcf_arr):
        return F.mse_loss(preds, postYcf_arr)

    def ate_mse(self, pred0, pred1, postYcf0, postYcf1, postYf1, postYf0):
        true_ite_vec = torch.cat([postYcf1-postYf0, postYf1-postYcf0], dim=0)   # (N, posttime_len)
        estim_ite_vec = torch.cat([pred1-postYf0, postYf1-pred0], dim=0)        # (N, posttime_len)
        return F.mse_loss(estim_ite_vec, true_ite_vec)
