import numpy as np
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

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

def preprocessFmcgDt(product):
    df = getFmcgRawDt(product)
    thscode_select = list(df['filename'].unique())  # the product we choose
    time_stump_df = df['日期'].unique()  # total timesteps
    tks = {
        'nc': {'TrVaSpl': 20240501, 'VaTsSpl': 20240511},
        'fs': {'TrVaSpl': 20240501, 'VaTsSpl': 20240511},
    }
    df_trans = encodingFeatures(df)
    all_stock_feature, all_stock_sales = getNumpyTsaXYfromDf(df_trans, thscode_select)
    
    train_len = len(df[df['日期'] < tks[product]['TrVaSpl']]['日期'].unique())  # time period for train task
    trainval_len = len(df[df['日期'] < tks[product]['VaTsSpl']]['日期'].unique())  # time period for train + validation
    all_stock_feature_norm = normFeatureWise(all_stock_feature, train_len)
    return all_stock_feature_norm, all_stock_sales, train_len, trainval_len, time_stump_df, thscode_select


def getDv(all_stock_sales, total_time_step, num_stock, tau):
    all_stock_return = np.zeros((total_time_step - tau, num_stock))  # (time_step-ts, stock_num)
    for i in range(total_time_step - tau):
        all_stock_return[i] = np.sum(all_stock_sales[i: i + tau], axis=0) / (np.sum(all_stock_sales[i-tau: i], axis=0) + np.ones(all_stock_sales[i].shape))  ## avoid zero
    all_stock_dv, all_stock_dvclass = np.array(all_stock_return.copy()), np.array(all_stock_return.copy())
    all_stock_dvclass[all_stock_return <= 1] = 0
    all_stock_dvclass[all_stock_return > 1] = 1
    all_stock_dv = np.log(all_stock_dv+1)
    return all_stock_dv, all_stock_dvclass

def encodingFeatures(df):
    ## TODO: fixed feature encoding
    # feature encoding
    replace_dict_trans = {
        '30%+': 32.5, '30%~35%': 32.5, '25%~30%': 27.5, '20%~25%': 22.5, '15%~20%': 17.5,
        '10%~15%': 12.5, '5%~10%': 7.5, '0~5%': 2.5, '0%': 0,
    }
    replace_dict_sales = {
        '0': 0, '0~25': 12.5, '25~50': 37.5, '50~75': 62.5, '75~100': 87.5, '100~250': 175,
        '250~500': 375, '500~750': 625, '750~1000': 875, '1000~2500': 1750, '2500~5000': 3750,
        '5000~7500': 6250, '7500~1w': 8750, '1w~2.5w': 17500, '2.5w~5w': 37500,
        '5w~7.5w': 62500, '7.5w~10w': 87500, '10w~25w': 175000,
    }
    df['转化率'] = df['转化率'].map(replace_dict_trans)
    df['销量'] = df['销量'].map(replace_dict_sales)
    df['转化率'] = df['转化率'].fillna(0)
    df['销量'] = df['销量'].fillna(0)
    df = df.replace('-', 0)
    df['浏览量'] = pd.to_numeric(df['浏览量'], errors='coerce').fillna(0)
    return df

def getNumpyTsaXYfromDf(df_trans, thscode_select):
    all_stock_sales = []  # (stock_num, time_step)
    all_stock_feature = []  # (stock_num, time_step, feature_dim)
    #feature_col_norm = ['销量', '浏览量', '转化率', '视频个数', '直播个数', '热推达人数'] # using clik
    feature_col_norm = ['浏览量', '转化率', '视频个数', '直播个数', '热推达人数'] ## TODO: to be processed and fixed
    for thscode in thscode_select:
        dt = df_trans[df_trans['filename'] == thscode]  # stock data of this code
        stock_i_feature = np.array(dt[feature_col_norm])
        all_stock_feature.append(stock_i_feature)
        stock_i_sales = np.array(dt['浏览量'])
        all_stock_sales.append(stock_i_sales)
    all_stock_feature = np.array(all_stock_feature)  # (stock_num, time_step, feature_dim)
    all_stock_sales = np.array(all_stock_sales).transpose((1, 0))  # -> (time_step, stock_num)
    return all_stock_feature, all_stock_sales


def getFmcgRawDt(product):
    if product == 'nc':
        df = pd.read_csv('data/product/日用百货汇总.csv')
    else:
        df = pd.read_csv('data/product/食品饮料汇总.csv')
    return df



class LoadFmcgDt():
    def __init__(self, product='fs'):
        self.K = 30  # lookback window size (larger than tau)
        self.tau = 3 # predict timestep ahead (define of raising)
        self.all_stock_feature, self.all_stock_sales, self.train_len, self.trainval_len, self.time_stump_df, self.thscode_select = preprocessFmcgDt(product=product)
        self.num_stock = len(self.thscode_select)
        self.time_length = len(self.time_stump_df)
        # dvckass -> (time_step-tau, stock_num)
        self.all_stock_dv, self.all_stock_dvclass = getDv(self.all_stock_sales, total_time_step=len(self.time_stump_df), num_stock=self.num_stock, tau=self.tau)
        
    def loadSamples(self, date, type='clas'):
        features = self.all_stock_feature[:, date:date + self.K, :] # process feature (N, time_step, feature_dim)
        #labels = torch.LongTensor(self.all_stock_dvclass[date + self.K])  # (stock_num)
        labels = torch.FloatTensor(self.all_stock_dv[date + self.K])  # (stock_num)
        return features, labels
    
    def loadTrainTest(self):
        trX, trY = [], []
        for date in range(0, self.train_len-self.tau-self.K):
            features, labels = self.loadSamples(date)
            trX.append(features)
            trY.append(labels)
        trX, trY = np.concatenate(trX, axis=0), np.concatenate(trY, axis=0)
        trDt = TSAData(trX, trY)

        vaX, vaY = [], []
        for date in range(self.train_len-self.tau-self.K, self.trainval_len-self.tau-self.K):
            features, labels = self.loadSamples(date)
            vaX.append(features)
            vaY.append(labels)
        vaX, vaY = np.concatenate(vaX, axis=0), np.concatenate(vaY, axis=0)
        vaDt = TSAData(vaX, vaY)

        tsX, tsY = [], []
        for date in range(self.trainval_len-self.tau-self.K, self.time_length-self.tau-self.K):
            features, labels = self.loadSamples(date)
            tsX.append(features)
            tsY.append(labels)
        tsX, tsY = np.concatenate(tsX, axis=0), np.concatenate(tsY, axis=0)
        tsDt = TSAData(tsX, tsY)

        return trDt, vaDt, tsDt



if __name__ == '__main__':
    #pass
    dataLoader = LoadFmcgDt(product='fs')
    # print(np.mean(dataLoader.all_stock_dv),np.std(dataLoader.all_stock_dv),np.max(dataLoader.all_stock_dv),np.min(dataLoader.all_stock_dv))
    trDt, vaDt, tsDt = dataLoader.loadTrainTest()
    print(trDt.x.shape, vaDt.x.shape, tsDt.x.shape) # [337144, 30, 10], [99160, 30, 10], [198320, 30, 10]
    print(trDt.y.shape, vaDt.y.shape, tsDt.y.shape) # [337144], [99160], [198320]

    print(pd.Series(dataLoader.all_stock_dvclass.flatten()).value_counts())
    #print(dataLoader.all_stock_dv)




