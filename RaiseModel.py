import torch
import torch.nn.functional as F

def sinkhorn(Q, epsilon=0.1, n_iters=20):
    # epsilon should be adjusted according to logits value's scale
    with torch.no_grad():
        Q = shoot_infs(Q)
        Q = torch.exp(Q / epsilon)
        for i in range(n_iters):
            Q /= Q.sum(dim=0, keepdim=True)
            Q /= Q.sum(dim=1, keepdim=True)
    return Q

def partial_sinkhorn(Q, epsilon=0.1, tai=0.1, n_iters=20):
    rho = tai / (tai + 0.1)  
    with torch.no_grad():
        Q = shoot_infs(Q)
        K = torch.exp(Q / epsilon)                          # (num_sample, n_state)
        B, Kd = K.shape
        a = torch.ones(B,  1, device=K.device, dtype=K.dtype)     # (num_sample, 1)
        b = torch.ones(Kd, 1, device=K.device, dtype=K.dtype)     # (n_state, 1) 
        u = torch.ones(B,  1, device=K.device, dtype=K.dtype)     # (num_sample, 1)
        v = torch.ones(Kd, 1, device=K.device, dtype=K.dtype)     # (n_state, 1)  
        for _ in range(n_iters):
            u = (a / (K @ v + 1e-12)).pow(rho)
            v = (b / (K.t() @ u + 1e-12)).pow(rho)
        Q = u * K * v.t() 
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
        self.attn_W = torch.nn.Linear(h_dim, h_dim, bias=True)
        self.attn_v = torch.nn.Linear(h_dim, 1, bias=False)

    def forward(self, x):
        outputs, _ = self.lstm(x)   # outputs: (batch, seq_len, hidden_size)

        # (B, K, H) -> (B, K, 1)
        score = self.attn_v(torch.tanh(self.attn_W(outputs)))   # (B, K, 1)
        alpha = torch.softmax(score, dim=1)                     # (B, K, 1)
        out   = (outputs * alpha).sum(dim=1)                    # (B, H)
        return out

        #outputs = outputs.transpose(1,2)  # (batch*stock_num, hidden_size, window_size_K)
        #return outputs[:,-1,:]

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
            #prob = F.softmax(preds / self.gstai, dim=-1)
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]
        # final_pred: (batch)
        return final_pred, preds, prob
