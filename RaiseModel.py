import torch
import torch.nn.functional as F

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

class LSTMHA(torch.nn.Module):
    def __init__(self, in_dim, h_dim,
                 lstm_num_layers=2, dropout=0.2):
        """
        Receive: batch_size, seq_len, input_size
        """
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=in_dim,
                            hidden_size=h_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=dropout)

    def forward(self, x):
        outputs, _ = self.lstm(x)   # outputs: (batch, seq_len, hidden_size)
        #outputs = outputs.transpose(1,2)  # (batch*stock_num, hidden_size, window_size_K)
        return outputs[:,-1,:]

class Raise(torch.nn.Module):
    def __init__(self, in_dim, h_dim, num_states=3):
        super().__init__()
        self.num_states = num_states
        self.gstai = 1
        self.feature_extractor = LSTMHA(in_dim, h_dim)
        self.training = True
        self.router = torch.nn.LSTM(
            input_size = in_dim,
            hidden_size = self.num_states,
            num_layers = 1,
            batch_first = True,
        )
        # self.fc = torch.nn.Linear(hidden_size + input_size, num_states)
        self.predictors = torch.nn.Linear(h_dim, self.num_states)

    def forward(self, x):

        emb = self.feature_extractor(x)
        # input: (batch, hidden)
        preds = self.predictors(emb) # preds: (batch, 3)

        if self.num_states == 1:
            return preds.squeeze(-1), preds, None
        # prob: (batch, num_state)
        prob = F.gumbel_softmax(preds, dim=-1, tau=self.gstai, hard=False)
        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]
        # final_pred: (batch)
        return final_pred, preds, prob
