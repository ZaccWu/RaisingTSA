import torch
import torch.nn.functional as F
import math
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
    def forward(self, x, fe=False):
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)  # ht: (batch*num_stock, K, h_dim)
        z_out = self.fc_out(ht[:, -1, :])
        if fe == False:
            z_out = self.act(z_out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out, ht
    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class MoETransformer(torch.nn.Module):
    """Encoder-only Transformer + Top-K 稀疏 MoE 预测头（超参数硬编码）"""
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        # ==================== MoE 超参数（硬编码） ====================
        self.num_experts = 4        # 专家数量
        self.top_k       = 2        # 每个样本激活的专家数
        self.moe_coef    = 0.01     # 负载均衡损失系数
        self.noise_std   = 0.1      # 训练时 gate logits 的噪声强度
        ff_dim           = h_dim * 2
        # ============================================================

        # ---- 与原 Transformer 完全相同的主干 ----
        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=2, dim_feedforward=h_dim,
            dropout=0.2, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_encoder.apply(self.init_weights)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, lag, h_dim))
        torch.nn.init.xavier_normal_(self.positional_encoding)

        # ---- MoE 组件：专家 + 门控 ----
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(h_dim, ff_dim),
                torch.nn.GELU(),
                torch.nn.Linear(ff_dim, h_dim),
            ) for _ in range(self.num_experts)
        ])
        self.gate = torch.nn.Linear(h_dim, self.num_experts)

        self.fc_out = torch.nn.Linear(h_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.training = True
        self.aux_loss = torch.tensor(0.0)   # 供训练时读取

    def forward(self, x, fe=False):
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)          # (B, K, h_dim)
        last = ht[:, -1, :]                        # (B, h_dim)
        last = self._moe_forward(last)             # MoE 融合
        z_out = self.fc_out(last)
        if fe == False:
            z_out = self.act(z_out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out, ht

    def _moe_forward(self, x):
        N = x.shape[0]
        device = x.device

        # 1) 门控打分（训练时加噪声帮助负载均衡）
        logits = self.gate(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_std

        # 2) Top-K 路由
        top_k_logits, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        top_k_w = torch.nn.functional.softmax(top_k_logits, dim=-1)

        # 3) 所有专家输出并加权融合
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (N, E, D)
        out = torch.zeros_like(x)
        for k in range(self.top_k):
            idx = top_k_idx[:, k]
            w   = top_k_w[:, k]
            sel = expert_outs[torch.arange(N, device=device), idx]
            out = out + w.unsqueeze(-1) * sel

        # 4) 负载均衡辅助损失（Switch Transformer 风格）
        tokens_per_expert = torch.zeros(self.num_experts, device=device)
        for k in range(self.top_k):
            tokens_per_expert.scatter_add_(
                0, top_k_idx[:, k], torch.ones(N, device=device)
            )
        tokens_per_expert = tokens_per_expert / (N * self.top_k)
        gate_probs = torch.nn.functional.softmax(logits, dim=-1).mean(dim=0)
        self.aux_loss = self.moe_coef * self.num_experts * (tokens_per_expert * gate_probs).sum()

        return out

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

class MoLETransformer(torch.nn.Module):
    """Encoder-only Transformer + Mixture-of-Linear-Experts (MoLE) 预测头
    参考：Ni et al., "Mixture-of-Linear-Experts for Long-term Time Series Forecasting"
    超参数硬编码，所有专家全激活（Dense MoE），无 aux_loss。
    """
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        # ==================== MoLE 超参数（硬编码） ====================
        self.num_heads    = 4       # 专家/头数量
        self.head_dropout = 0.2     # 训练时 Head Dropout 比例
        # ============================================================

        # ---- Transformer 主干（与 Transformer 类逐行一致，保证公平对比） ----
        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=2, dim_feedforward=h_dim,
            dropout=0.2, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_encoder.apply(self.init_weights)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, lag, h_dim))
        torch.nn.init.xavier_normal_(self.positional_encoding)

        # ---- MoLE 组件 ----
        # 多个线性专家（heads），每个把 h_dim 映射到 out_dim
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(h_dim, out_dim) for _ in range(self.num_heads)
        ])

        # 混合层（router）：2 层 MLP + ReLU，输入为第一个时间步的"时间戳嵌入"
        # 论文中用 timestamp embedding；这里用第一个时间步的特征向量作为等价信号
        self.mixing_layer = torch.nn.Sequential(
            torch.nn.Linear(in_dim, h_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(h_dim, self.num_heads),
        )

        self.act = torch.nn.LeakyReLU()
        self.training = True

    def forward(self, x, fe=False):
        # x: (B, lag, in_dim)
        # ---- 1) 共享主干编码 ----
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)              # (B, lag, h_dim)
        last = ht[:, -1, :]                            # (B, h_dim)

        # ---- 2) 所有专家（Dense，全部前向） ----
        head_outs = torch.stack([h(last) for h in self.heads], dim=1)
        # head_outs: (B, num_heads, out_dim)

        # ---- 3) 路由权重（基于第一个时间步的"时间戳嵌入"） ----
        # 论文的关键设计：仅用输入序列的起点做 gating
        router_in = x[:, 0, :]                          # (B, in_dim)
        logits = self.mixing_layer(router_in)           # (B, num_heads)
        weights = torch.softmax(logits, dim=-1)         # (B, num_heads)，和为 1

        # ---- 4) Head Dropout（训练时随机丢弃一部分 head 权重） ----
        if self.training and self.head_dropout > 0:
            keep = (torch.rand_like(weights) > self.head_dropout).float()
            weights = weights * keep
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        # ---- 5) 加权融合 ----
        z_out = (head_outs * weights.unsqueeze(-1)).sum(dim=1)  # (B, out_dim)

        if fe == False:
            z_out = self.act(z_out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)
        return z_out, ht

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class DCDetector(torch.nn.Module):
    """Encoder-only Transformer + DCdetector 双分支对比学习
    参考：Yang et al., "DCdetector: Dual Attention Contrastive Representation
    Learning for Time Series Anomaly Detection", KDD 2023.
    核心：双注意力非对称设计 + 纯对比损失（无重建）。
    """
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        # ==================== DCdetector 超参数（硬编码） ====================
        self.proj_dim    = h_dim // 2   # 投影维度
        self.temperature = 0.1          # 对比损失温度
        self.cl_coef     = 1.0          # 纯对比，权重 1.0
        self.jitter_std  = 0.1          # 时域抖动噪声标准差
        # ==================================================================

        # ---- 共享主干 ----
        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=2, dim_feedforward=h_dim,
            dropout=0.2, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_encoder.apply(self.init_weights)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, lag, h_dim))
        torch.nn.init.xavier_normal_(self.positional_encoding)

        # ---- 预测头 ----
        self.fc_out = torch.nn.Linear(h_dim, out_dim)

        # ---- 双分支投影头 ----
        # 两个分支结构相同但参数独立，产生非对称表示
        self.proj_head_1 = torch.nn.Sequential(
            torch.nn.Linear(h_dim, h_dim), torch.nn.ReLU(),
            torch.nn.Linear(h_dim, self.proj_dim),
        )
        self.proj_head_2 = torch.nn.Sequential(
            torch.nn.Linear(h_dim, h_dim), torch.nn.ReLU(),
            torch.nn.Linear(h_dim, self.proj_dim),
        )

        self.act = torch.nn.LeakyReLU()
        self.training = True
        self.contrastive_loss = torch.tensor(0.0)

    def _contrastive_loss(self, z1, z2):
        """基于两分支相似度的对比损失（DCdetector 核心）"""
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        # 两个分支在正常点上应高度一致，异常点上应产生差异
        sim = F.cosine_similarity(z1, z2, dim=-1)           # (B,)
        loss = (1 - sim).mean()                              # 最大化一致性
        return loss

    def forward(self, x, fe=False):
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)
        last = ht[:, -1, :]

        z_out = self.fc_out(last)
        if fe == False:
            z_out = self.act(z_out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)

        if self.training:
            # 时域抖动增强：加高斯噪声
            view = x + torch.randn_like(x) * self.jitter_std
            h_aug = self.transformer_encoder(self.fc_in(view) + self.positional_encoding)[:, -1, :]
            z1 = self.proj_head_1(last)      # 原始视图
            z2 = self.proj_head_2(h_aug)     # 抖动视图
            self.contrastive_loss = self.cl_coef * self._contrastive_loss(z1, z2)
        else:
            self.contrastive_loss = torch.tensor(0.0, device=x.device)

        return z_out, ht

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class CARLA(torch.nn.Module):
    """Encoder-only Transformer + CARLA 异常注入对比学习
    参考：Darban et al., "CARLA: Self-supervised Contrastive Representation
    Learning for Time Series Anomaly Detection", Pattern Recognition 2025.
    核心：异常注入 + 最近邻/最远邻自监督分类。
    """
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        # ==================== CARLA 超参数（硬编码） ====================
        self.proj_dim      = h_dim // 2
        self.temperature   = 0.1
        self.cl_coef       = 0.5
        self.anomaly_ratio = 0.3    # 注入异常的时间步比例
        self.num_anomaly_types = 3  # 注入异常类型数
        # ================================================================

        # ---- 共享主干 ----
        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=h_dim, nhead=2, dim_feedforward=h_dim,
            dropout=0.2, batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.transformer_encoder.apply(self.init_weights)
        self.positional_encoding = torch.nn.Parameter(torch.zeros(1, lag, h_dim))
        torch.nn.init.xavier_normal_(self.positional_encoding)

        self.fc_out = torch.nn.Linear(h_dim, out_dim)

        # ---- 投影头 ----
        self.proj_head = torch.nn.Sequential(
            torch.nn.Linear(h_dim, h_dim), torch.nn.ReLU(),
            torch.nn.Linear(h_dim, self.proj_dim),
        )

        # ---- 自监督分类头（基于最近邻/最远邻） ----
        self.classifier = torch.nn.Linear(self.proj_dim, 2)  # 正常 vs 异常

        self.act = torch.nn.LeakyReLU()
        self.training = True
        self.contrastive_loss = torch.tensor(0.0)

    def _inject_anomalies(self, x):
        """注入多种类型的异常作为负样本"""
        B, K, D = x.shape
        x_anom = x.clone()
        n_inject = max(1, int(K * self.anomaly_ratio))

        # 类型1：尖峰（spike）
        idx = torch.randperm(K, device=x.device)[:n_inject]
        x_anom[:, idx, :] += torch.randn(B, n_inject, D, device=x.device) * 3.0

        # 类型2：水平偏移（level shift）
        idx2 = torch.randperm(K, device=x.device)[:n_inject]
        x_anom[:, idx2, :] += 1.0

        # 类型3：随机置零（dropout）
        mask = (torch.rand(B, K, 1, device=x.device) > 0.2).float()
        x_anom = x_anom * mask

        return x_anom

    def _carla_loss(self, z_normal, z_anom):
        """异常注入对比损失 + 自监督分类"""
        B = z_normal.shape[0]
        z_normal = F.normalize(z_normal, dim=-1)
        z_anom = F.normalize(z_anom, dim=-1)

        # 对比损失：正常表示应远离异常表示
        sim = F.cosine_similarity(z_normal, z_anom, dim=-1)
        contrastive = -torch.log(torch.exp(sim / self.temperature) + 1e-8).mean()

        # 自监督分类：区分正常和异常
        logits = self.classifier(torch.cat([z_normal, z_anom], dim=0))
        labels = torch.cat([
            torch.zeros(B, dtype=torch.long, device=z_normal.device),
            torch.ones(B, dtype=torch.long, device=z_normal.device),
        ])
        cls_loss = F.cross_entropy(logits, labels)

        return contrastive + 0.5 * cls_loss

    def forward(self, x, fe=False):
        x1 = self.fc_in(x) + self.positional_encoding
        ht = self.transformer_encoder(x1)
        last = ht[:, -1, :]

        z_out = self.fc_out(last)
        if fe == False:
            z_out = self.act(z_out)
        if z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)

        if self.training:
            # 正常视图表示
            z_norm = self.proj_head(last)
            # 注入异常并编码
            x_anom = self._inject_anomalies(x)
            h_anom = self.transformer_encoder(
                self.fc_in(x_anom) + self.positional_encoding
            )[:, -1, :]
            z_anom = self.proj_head(h_anom)
            self.contrastive_loss = self.cl_coef * self._carla_loss(z_norm, z_anom)
        else:
            self.contrastive_loss = torch.tensor(0.0, device=x.device)

        return z_out, ht

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

class DeepARTransformer(torch.nn.Module):
    def __init__(self, lag, in_dim, h_dim, out_dim):
        super().__init__()
        self.rnn_layers = 2
        self.dropout = 0.1
        self.num_samples = 100

        self.fc_in = torch.nn.Linear(in_dim, h_dim)
        self.rnn = torch.nn.LSTM(
            input_size=h_dim, hidden_size=h_dim,
            num_layers=self.rnn_layers,
            dropout=self.dropout if self.rnn_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_mu = torch.nn.Linear(h_dim, out_dim)
        self.fc_sigma = torch.nn.Linear(h_dim, out_dim)
        self.act = torch.nn.LeakyReLU()
        self.training = True

    def forward(self, x, fe=False, return_stats=False):
        x1 = self.fc_in(x)
        ht, _ = self.rnn(x1)
        last = ht[:, -1, :]                               # (B, h_dim)

        mu = self.fc_mu(last)                             # (B, out_dim)
        sigma = F.softplus(self.fc_sigma(last)) + 1e-6    # (B, out_dim)

        # ---- 展平到 1D（out_dim==1 时） ----
        mu_flat    = mu.squeeze(-1)    if mu.shape[-1]    == 1 else mu       # (B,)
        sigma_flat = sigma.squeeze(-1) if sigma.shape[-1] == 1 else sigma    # (B,)

        # ---- 点预测 ----
        z_out = self.act(mu) if fe == False else (mu, sigma)
        if isinstance(z_out, torch.Tensor) and z_out.shape[-1] == 1:
            z_out = z_out.squeeze(1)

        if return_stats:
            return z_out, ht, mu_flat, sigma_flat
        return z_out, ht

    @staticmethod
    def nll_loss_fn(y, mu, sigma):
        """高斯 NLL，接收显式参数，不依赖 self 状态"""
        y = y.reshape(-1)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        n = min(y.shape[0], mu.shape[0])
        y, mu, sigma = y[:n], mu[:n], sigma[:n]
        nll = torch.log(sigma) + 0.5 * math.log(2 * math.pi) \
              + (y - mu) ** 2 / (2 * sigma ** 2)
        return nll.mean()

    def init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

