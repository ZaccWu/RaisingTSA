import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class TSAData(Dataset):
    def __init__(self, x_l, y_l):
        super(TSAData, self).__init__()
        self.x = torch.FloatTensor(x_l)
        self.y = torch.LongTensor(y_l)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return (len(self.y))

def normFeatureWise(all_stock_feature):
    # normalize for each feature
    normalized_data = np.zeros_like(all_stock_feature)
    for feature_idx in range(all_stock_feature.shape[2]):
        feature_data = all_stock_feature[:, :, feature_idx]
        min_val = feature_data.min()
        max_val = feature_data.max()
        normalized_data[:, :, feature_idx] = (feature_data - min_val) / (max_val - min_val)
    normalized_feature = normalized_data
    return normalized_feature

def preprocessAliDt(df):
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
    all_stock_feature_norm = normFeatureWise(all_stock_feature)
    return all_stock_feature_norm, all_stock_sales

def getDv(all_stock_sales, time_len, num_stock, tau):
    all_stock_return = np.zeros((time_len- tau, num_stock))  # (time_step-ts, stock_num)
    for i in range(time_len - tau):
        all_stock_return[i] = np.sum(all_stock_sales[i: i + tau], axis=0) / (np.sum(all_stock_sales[i-tau: i], axis=0) + np.ones(all_stock_sales[i].shape))  ## avoid zero
        all_stock_dvclass = np.array(all_stock_return.copy())
        all_stock_dvclass[all_stock_return < 2.21] = 0
        all_stock_dvclass[all_stock_return >= 2.21] = 1
    return all_stock_dvclass


class LoadAliDt():
    def __init__(self):
        self.df = np.load('data/dv_count2.npy', allow_pickle=True) # (9916, 97, 12)
        self.K = 30  # lookback window size (larger than tau)
        self.tau = 3 # predict timestep ahead (define of raising)
        self.num_stock, self.time_length = self.df.shape[0], self.df.shape[1]
        self.tr_len, self.trva_len = int(self.time_length*0.7), int(self.time_length*0.8)

        # process feature (stock_num, time_step, feature_dim) and target (time_step, stock_num)
        self.all_stock_feature, self.all_stock_sales = preprocessAliDt(self.df)
        # dvckass -> (time_step-tau, stock_num)
        self.all_stock_dvclass = getDv(self.all_stock_sales, self.time_length, self.num_stock, self.tau)

    def loadSamples(self, date, type='clas'):
        features = self.all_stock_feature[:, date:date + self.K, :] # process feature (N, time_step, feature_dim)
        labels = self.all_stock_dvclass[date+self.K, :].T # -> (N, time_step)
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
    pass
    # dataLoader = LoadAliDt()
    # trDt, vaDt, tsDt = dataLoader.loadTrainTest()
    # print(trDt.x.shape, vaDt.x.shape, tsDt.x.shape)
    # print(trDt.y.shape, vaDt.y.shape, tsDt.y.shape)



