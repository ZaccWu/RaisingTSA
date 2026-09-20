import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TSAData(Dataset):
    def __init__(self, x_l, y_l):
        super(TSAData, self).__init__()
        self.x = torch.FloatTensor(x_l)
        self.y = torch.FloatTensor(y_l)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return (len(self.y))

def normFeatureWise(all_stock_feature, tr_len):
    x = all_stock_feature
    mu  = x[:, :tr_len, :].mean(axis=1, keepdims=True)   # (S, 1, F)
    std = x[:, :tr_len, :].std(axis=1,  keepdims=True) + 1e-6
    x = (x - mu) / std
    return x

def preprocessAliDt(df, tr_len):
    num_content = df.shape[0]
    all_stock_feature, all_stock_sales = [], []
    for j in range(num_content):
        df_j = df[j,:,:]
        fea_j = df_j[np.argsort(df_j[:,1])][:,2:].astype(np.float64)  # (time_step, fea_dim), remove 'content_id', 'visite_time'
        all_stock_feature.append(fea_j)
        sales_j = fea_j[:,-1].astype(np.float64)
        all_stock_sales.append(sales_j)
    all_stock_feature = np.array(all_stock_feature)  # (stock_num, time_step, feature_dim)
    all_stock_sales = np.array(all_stock_sales).transpose((1, 0))  # -> (time_step, stock_num)
    all_stock_feature_norm = normFeatureWise(all_stock_feature, tr_len)
    return all_stock_feature_norm, all_stock_sales

def getDv(all_stock_sales, time_len, num_stock, tau):
    all_stock_return = np.zeros((time_len- tau, num_stock))  # (time_step-ts, stock_num)
    for i in range(time_len - tau):
        all_stock_return[i] = np.sum(all_stock_sales[i: i + tau], axis=0) / (np.sum(all_stock_sales[i-tau: i], axis=0) + np.ones(all_stock_sales[i].shape))  ## avoid zero
        all_stock_dv, all_stock_dvclass = np.array(all_stock_return.copy()), np.array(all_stock_return.copy())
        all_stock_dvclass[all_stock_return < 1] = 0
        all_stock_dvclass[all_stock_return >= 1] = 1
        all_stock_dv = np.log(all_stock_dv+1)
    return all_stock_dv, all_stock_dvclass


class LoadAliDt():
    def __init__(self):
        self.df = np.load('data/dv_count2.npy', allow_pickle=True) # (9916, 97, 12)
        self.K = 30  # lookback window size (larger than tau)
        self.tau = 3 # predict timestep ahead (define of raising)
        self.num_stock, self.time_length = self.df.shape[0], self.df.shape[1]
        self.tr_len, self.trva_len = int(self.time_length*0.7), int(self.time_length*0.8)

        # process feature (stock_num, time_step, feature_dim) and target (time_step, stock_num)
        self.all_stock_feature, self.all_stock_sales = preprocessAliDt(self.df, self.tr_len)
        # dvckass -> (time_step-tau, stock_num)
        self.all_stock_dv, self.all_stock_dvclass = getDv(self.all_stock_sales, self.time_length, self.num_stock, self.tau)

    def loadSamples(self, date, type='clas'):
        features = self.all_stock_feature[:, date:date + self.K, :] # process feature (N, time_step, feature_dim)
        labels = self.all_stock_dv[date+self.K, :].T # -> (N, time_step)
        return features, labels
    
    def loadTrainTest(self):
        trX, trY = [], []
        for date in range(0, self.tr_len-self.tau-self.K):
            features, labels = self.loadSamples(date)
            trX.append(features)
            trY.append(labels)
        trX, trY = np.concatenate(trX, axis=0), np.concatenate(trY, axis=0)
        trDt = TSAData(trX, trY)

        vaX, vaY = [], []
        for date in range(self.tr_len-self.tau-self.K, self.trva_len-self.tau-self.K):
            features, labels = self.loadSamples(date)
            vaX.append(features)
            vaY.append(labels)
        vaX, vaY = np.concatenate(vaX, axis=0), np.concatenate(vaY, axis=0)
        vaDt = TSAData(vaX, vaY)

        tsX, tsY = [], []
        for date in range(self.trva_len-self.tau-self.K, self.time_length-self.tau-self.K):
            features, labels = self.loadSamples(date)
            tsX.append(features)
            tsY.append(labels)
        tsX, tsY = np.concatenate(tsX, axis=0), np.concatenate(tsY, axis=0)
        tsDt = TSAData(tsX, tsY)

        return trDt, vaDt, tsDt



if __name__ == '__main__':
    #pass
    dataLoader = LoadAliDt()
    # print(np.mean(dataLoader.all_stock_dv),np.std(dataLoader.all_stock_dv),np.max(dataLoader.all_stock_dv),np.min(dataLoader.all_stock_dv))
    trDt, vaDt, tsDt = dataLoader.loadTrainTest()
    print(trDt.x.shape, vaDt.x.shape, tsDt.x.shape) # [337144, 30, 10], [99160, 30, 10], [198320, 30, 10]
    print(trDt.y.shape, vaDt.y.shape, tsDt.y.shape) # [337144], [99160], [198320]

    print(pd.Series(dataLoader.all_stock_dvclass.flatten()).value_counts())



