import random
import pandas as pd
import numpy as np
import math

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf, as_tuple=False)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def sinkhorn(Q, n_iters=3, epsilon=0.01):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        # Q = shoot_infs(Q)
        #
        # Q = torch.exp(Q / epsilon)
        # print(Q)
        Q=-Q
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
        #print(Q)
    return Q


class MyDataset(Dataset):
    def __init__(self, data):
        self._index, self._feature, self._label = _set_data(data)

        self.seq_len = 48
        self.horizon = 0
        self.num_states = 3
        #self.batch_size = 20
        self.shuffle = False

        # add memory to feature -> self._data: (N, channel + num_pattern)
        #self._data = np.c_[self._feature, np.zeros((len(self._feature), self.num_states), dtype=np.float32)]
        # padding tensor -> self.zeros: (seq_len, channel + num_pattern)
        #self.zeros = np.zeros((self.seq_len, self._data.shape[1]), dtype=np.float32)
        # create batch slices
        self.batch_slices = _create_ts_slices(self._index, self.seq_len)
        self.slices = self.batch_slices.copy()

        self.data, self.label, self.index = [], [], []
        for slc in self.slices:
            self.data.append(self._feature[slc].copy())
            self.label.append(self._label[slc.stop - 1])
            self.index.append(slc.stop - 1)

        self.index = torch.tensor(self.index, device=device)
        self.data = torch.tensor(self.data, device=device)
        self.label = torch.tensor(self.label, device=device)


    def __getitem__(self, id):
        ## TODO: update memory
        return self.index[id], self.data[id], self.label[id]

    def __len__(self):
        return len(self.data)

    def assign_data(self, index, vals):
        vals = vals.detach().cpu().numpy()
        index = index.detach().cpu().numpy()
        self._data[index, -self.num_states :] = vals

    def clear_memory(self):
        self._data[:, -self.num_states :] = 0


class LSTMHA(torch.nn.Module):
    def __init__(self, lstm_input_size=4, lstm_h_dim=16,
                 lstm_num_layers=1, dropout=0.2):
        """
        Receive: batch_size, seq_len, input_size
        """
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_h_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)

    def forward(self, x):
        outputs, _ = self.lstm(x)   # outputs: (batch, seq_len, hidden_size)
        #outputs = outputs.transpose(1,2)  # (batch*stock_num, hidden_size, window_size_K)
        return outputs[:,-1,:]

class Tra(torch.nn.Module):
    def __init__(self, input_size=16, num_states=3):
        super().__init__()
        self.num_state = num_states
        self.tau = 1
        self.training = True

        self.router = torch.nn.LSTM(
            input_size = input_size,
            hidden_size = self.num_state,
            num_layers = 1,
            batch_first = True,
        )
        # self.fc = torch.nn.Linear(hidden_size + input_size, num_states)
        self.predictors = torch.nn.Linear(16,3)

    def forward(self, hidden):
        # input: (batch, hidden)
        preds = self.predictors(hidden) # preds: (batch, 3)

        if self.num_state == 1:
            return preds.squeeze(-1), preds, None

        # prob: (batch, num_state)
        prob = F.gumbel_softmax(preds, dim=-1, tau=self.tau, hard=False)

        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]

        # final_pred: (batch)
        return final_pred, preds, prob


class TRAModel(object):
    def __init__(self):
        self.tra = Tra().to(device)
        self.model = LSTMHA().to(device)
        self.optimizer = torch.optim.Adam(self.tra.parameters(), lr=0.01)
        self.global_step = -1
        self.lamb = PARAM['lamb']

    def train_epoch(self, train_data):
        '''
        :param train_data: pytorch dataloader
        :return:
        '''
        self.tra.train()
        self.model.train()
        self.tra.training = True

        epoch_loss = 0
        total_count = 0

        for batch_idx, data in enumerate(train_data):
            # count += 1
            # if count > max_step:
            #     break
            self.global_step += 1

            # feature: (batch, seq_len, channel)
            index, feature, label = data[0], data[1], data[2]    # click, conver, <uid, time>
            hidden = self.model(feature)    # outputs: (batch, hidden_size)

            pred, all_preds, prob = self.tra(hidden)
            loss = (pred - label).pow(2).mean()

            L = (all_preds.detach() - label[:, None]).pow(2)
            # print("L before norm:", L)
            #L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            # print("L after norm:", L)
            if prob is not None:
                P = sinkhorn(-L, epsilon=0.01)  # sample assignment matrix
                lamb = self.lamb * (PARAM['rho'] ** self.global_step)
                reg = prob.log().mul(P).sum(dim=-1).mean()
                loss = loss - lamb * reg
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            total_count += len(pred)


        epoch_loss /= total_count

        return epoch_loss


    def test_epoch(self, test_data, return_pred=False):
        self.tra.eval()
        self.tra.training = False
        preds = []
        metrics = []
        for idx, data in enumerate(test_data):
            #print(idx)
            index, feature, label = data[0], data[1], data[2]
            with torch.no_grad():
                hidden = self.model(feature)
                pred, all_preds, prob = self.tra(hidden)

            L = (all_preds - label[:, None]).pow(2)
            L -= L.min(dim=-1, keepdim=True).values  # normalize & ensure positive input
            X = np.c_[
                pred.cpu().numpy(),
                label.cpu().numpy(),
            ]
            columns = ["score", "label"]
            if prob is not None:
                X = np.c_[X, all_preds.cpu().numpy(), prob.cpu().numpy()]
                columns += ["score_%d" % d for d in range(all_preds.shape[1])] + [
                    "prob_%d" % d for d in range(all_preds.shape[1])
                ]

            pred = pd.DataFrame(X, index=index.cpu().numpy(), columns=columns)

            metrics.append(evaluate(pred))

        metrics = pd.DataFrame(metrics)
        metrics = {
            "MSE": metrics.MSE.mean(),
            "MAE": metrics.MAE.mean(),
        }

        return metrics, preds

def main():

    static = pd.read_csv('ufeature.csv').set_index('uid')
    dynamic = pd.read_csv('trajectory.csv').set_index('uid')

    train_dt = dynamic[dynamic['time']<400]
    test_dt = dynamic[dynamic['time']>=400]

    HData_train, HData_test = MyDataset(train_dt), MyDataset(test_dt)
    train_loader = DataLoader(HData_train, batch_size=256)
    test_loader = DataLoader(HData_test)

    traModel = TRAModel()

    # train
    traModel.global_step = -1
    for epoch in range(PARAM['n_epoch']):
        epoch_loss = traModel.train_epoch(train_loader)
        print("Finish Training Epoch: ", epoch, " Loss: ", epoch_loss)
    metrics, preds = traModel.test_epoch(test_loader)
    print(metrics)

SEED = 100
set_seed(SEED)

PARAM = {
    'n_epoch': 5,  # default: 50
    'pattern': 3,   # default: 3
    'rho': 0.99,    # default: 0.99
    'lamb': 0,      # default: 1
}

if __name__ == "__main__":
    main()