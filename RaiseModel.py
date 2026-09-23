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
        nu = torch.ones(Kd, 1, device=K.device, dtype=K.dtype) / Kd
        b = nu * B               # (n_state, 1)
        #b = torch.ones(Kd, 1, device=K.device, dtype=K.dtype)     # (n_state, 1) 
        u = torch.ones(B,  1, device=K.device, dtype=K.dtype)     # (num_sample, 1)
        v = torch.ones(Kd, 1, device=K.device, dtype=K.dtype)     # (n_state, 1)  
        for _ in range(n_iters):
            u = (a / (K @ v + 1e-12)).pow(rho)
            v = (b / (K.t() @ u + 1e-12)).pow(rho)
        Q = u * K * v.t() 
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
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



class LSTM(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.dropout = 0.2
        self.LSTM = torch.nn.LSTM(in_dim, h_dim, 2, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.linear = torch.nn.Linear(h_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.training = True
    def forward(self, x):
        ht, _ = self.LSTM(x)   # ht: (batch*num_stock, K, h_dim)
        z_out = self.act(self.linear(ht[:,-1,:]))   # z_out: (batch*num_stock, out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out

class GRU(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.dropout = 0.2
        self.GRU = torch.nn.GRU(in_dim, h_dim, 2, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.linear = torch.nn.Linear(h_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.training = True
    def forward(self, x):
        ht, hn = self.GRU(x)  # ht: (batch*num_stock, K, h_dim)
        z_out = self.act(self.linear(ht[:, -1, :]))  # z_out: (batch*num_stock, out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out

class Transformer(torch.nn.Module):   # encoder only transformer
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super(Transformer, self).__init__()
        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=2, dim_feedforward=h_dim, dropout=0.2, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_encoder.apply(self.init_weights)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, lag, h_dim))
        torch.nn.init.xavier_normal_(self.positional_encoding)
        self.fc_out = torch.nn.Linear(h_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.training = True
    def forward(self, x):
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)  # ht: (batch*num_stock, K, h_dim)
        z_out = self.act(self.fc_out(ht[:, -1, :])) 
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out
    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

class LSTMTATT(torch.nn.Module):
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        self.dropout = 0.2
        self.LSTM = torch.nn.LSTM(in_dim, h_dim, 2, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.W = torch.nn.Parameter(torch.zeros(lag, h_dim))
        torch.nn.init.xavier_normal_(self.W.data)
        self.fc = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU(True))

    def forward(self, x):
        ht, (hn, cn) = self.LSTM(x)
        ht_W = ht.mul(self.W)
        ht_W = torch.sum(ht_W, dim=2)
        att = F.softmax(ht_W, dim=1)
        t_att = att.unsqueeze(dim=1)
        att_ht = torch.bmm(t_att, ht).squeeze(1)
        z_out = self.fc(att_ht)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out



class LSTMHA(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.dropout = 0.2
        self.lstm = torch.nn.LSTM(in_dim, h_dim, 2, batch_first=True, bidirectional=False, dropout=self.dropout)
        self.attn_W = torch.nn.Linear(h_dim, h_dim, bias=True)
        self.attn_v = torch.nn.Linear(h_dim, 1, bias=False)
        self.fc = torch.nn.Sequential(torch.nn.Linear(h_dim, out_dim), torch.nn.LeakyReLU(True))

    def forward(self, x):
        outputs, _ = self.lstm(x)   # outputs: (batch, seq_len, hidden_size)
        # (B, K, H) -> (B, K, 1)
        score = self.attn_v(torch.tanh(self.attn_W(outputs)))   # (B, K, 1)
        alpha = torch.softmax(score, dim=1)                     # (B, K, 1)
        z_out   = (outputs * alpha).sum(dim=1)                    # (B, H)
        z_out = self.fc(z_out)       # (B, H)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out



class LSTMAE(torch.nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.encoder = torch.nn.LSTM(in_dim, h_dim, num_layers=1,
                                     batch_first=True, bidirectional=False)
        self.decoder = torch.nn.LSTM(h_dim, h_dim, num_layers=1,
                                     batch_first=True, bidirectional=False)
        self.out = torch.nn.Linear(h_dim, in_dim)

    def forward(self, x):
        # x: (B, K, F)
        enc_seq, _ = self.encoder(x)        # (B, K, H)
        dec_seq, _ = self.decoder(enc_seq)  # (B, K, H)
        x_hat = self.out(dec_seq)           # (B, K, F)
        return enc_seq, x_hat



class Raise(torch.nn.Module):
    def __init__(self, in_dim, h_dim, num_states=3, extractor='lstmtatt'):
        super().__init__()
        self.num_states = num_states
        self.gstai = 1
        if extractor == 'lstm':
            self.feature_extractor = LSTM(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'gru':
            self.feature_extractor = GRU(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'trans':
            self.feature_extractor = Transformer(30, in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'lstmha':
            self.feature_extractor = LSTMHA(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'lstmtatt':
            self.feature_extractor = LSTMTATT(30, in_dim, h_dim, out_dim=h_dim)
        else:
            raise ValueError('Extractor not specify')

        self.training = True
        self.router = LSTMAE(in_dim, h_dim)
        self.fc = torch.nn.Linear(h_dim + in_dim, num_states)
        self.predictors = torch.nn.Linear(h_dim, self.num_states)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        emb = self.feature_extractor(x) # (n, K, fea_dim)->(n, h_dim)
        # input: (batch, hidden)
        _, x_hat = self.router(x)  # (n, K, fea_dim)->(n, K, h_dim)
        recon_err = (x_hat - x).pow(2).mean(dim=1)   # ->(n, fea_dim)
        preds = self.act(self.predictors(emb)) # preds: (batch, 3)

        if self.num_states == 1:
            return preds.squeeze(-1), preds, None, None
        # prob: (batch, num_state)
        #prob = F.gumbel_softmax(preds, dim=-1, tau=self.gstai, hard=False)
        rot_out = self.act(self.fc(torch.cat([emb, recon_err], dim=-1)))

        if self.training:
            prob = F.gumbel_softmax(rot_out, tau=self.gstai, hard=False)
            final_pred = (preds * prob).sum(dim=-1)
        else:
            prob = F.softmax(rot_out, dim=-1)
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]
        # final_pred: (batch)
        return final_pred, preds, prob, None


class RaiseSep(torch.nn.Module):
    def __init__(self, in_dim, h_dim, num_states=3, extractor='lstmtatt'):
        super().__init__()
        self.num_states = num_states
        self.gstai = 1
        if extractor == 'lstm':
            self.feature_extractor = LSTM(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'gru':
            self.feature_extractor = GRU(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'trans':
            self.feature_extractor = Transformer(30, in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'lstmha':
            self.feature_extractor = LSTMHA(in_dim, h_dim, out_dim=h_dim)
        elif extractor == 'lstmtatt':
            self.feature_extractor = LSTMTATT(30, in_dim, h_dim, out_dim=h_dim)
        else:
            raise ValueError('Extractor not specify')

        self.training = True
        self.router = LSTMAE(in_dim, h_dim)
        self.fc = torch.nn.Linear(h_dim + in_dim, num_states)
        self.predictors = torch.nn.Linear(h_dim, self.num_states)
        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        emb = self.feature_extractor(x) # (n, K, fea_dim)->(n, h_dim)
        # input: (batch, hidden)
        _, x_hat = self.router(x)  # (n, K, fea_dim)->(n, K, h_dim)
        recon_err = (x_hat - x).pow(2).mean(dim=1)   # ->(n, fea_dim)
        preds = self.act(self.predictors(emb)) # preds: (batch, 3)


        # prob: (batch, num_state)
        rot_out = self.act(self.fc(torch.cat([emb, recon_err], dim=-1)))

        per_sample_err = recon_err.mean(dim=-1)    # (n,)
        batch_mean_err = per_sample_err.mean()     # 标量

        if self.num_states == 1:
            return preds.squeeze(-1), preds, None, batch_mean_err

        high_re_mask = (per_sample_err > batch_mean_err).to(preds.dtype)  # (n,) 0/1

        if self.training:
            prob = F.gumbel_softmax(rot_out, tau=self.gstai, hard=False)
        else:
            prob = F.softmax(rot_out, dim=-1)

        soft_pred = (preds * prob).sum(dim=-1)     # (n,)
        hard_idx = prob.argmax(dim=-1, keepdim=True)              # (n, 1)
        hard_onehot = torch.zeros_like(prob).scatter_(1, hard_idx, 1.0)
        hard_ste = hard_onehot - prob.detach() + prob             # 直通
        hard_pred = (preds * hard_ste).sum(dim=-1)                # (n,)
        final_pred = hard_pred * (1.0 - high_re_mask) + soft_pred * high_re_mask
        return final_pred, preds, prob, batch_mean_err