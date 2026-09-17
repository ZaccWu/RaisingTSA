import torch
import torch.nn as nn


class lstmExtractor(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=1):
        super(lstmExtractor, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, # time series feature dim
            hidden_size = hidden_dim,   # latent dim
            num_layers = 1,     # lstm layers
            batch_first=True,   # input: (batch, seq_len, input_size), output: (batch, seq_len, hidden_size), (hn, cn)
            dropout = 0.2,
        )

    def forward(self, pretime):
        pretime = pretime.unsqueeze(-1) # (batch, pretime_len, 1)
        emb = self.lstm(pretime)[0]
        emb = emb.transpose(1,2)   # (batch, seq_len, hidden_size) -> (batch, hidden_size, seq_len)
        return emb[:,:,-1]  # the last timestep's emb: (batch, hidden_size)

class nnExtractor(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim=120):
        super(nnExtractor, self).__init__()
        self.NN = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, pretime):
        emb = self.NN1(pretime)
        return emb


class Predictor(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim=20):
        super(Predictor, self).__init__()
        #self.NN = torch.nn.Linear(PARAM['Hidden'], output_dim)
        self.NN1 = torch.nn.Linear(hidden_dim, 8)
        self.NN2 = torch.nn.Linear(8, output_dim)

    def forward(self, emb):
        mid = self.NN1(emb)
        pred = self.NN2(mid)
        return pred

class PropensityNet(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(PropensityNet, self).__init__()
        self.NN = torch.nn.Linear(hidden_dim,1)

    def forward(self, emb):
        pred_pro = self.NN(emb)
        return pred_pro