import torch
from modeling.net import lstmExtractor, nnExtractor, Predictor, PropensityNet


class MyModel(torch.nn.Module):
    def __init__(self, PARAM):
        super().__init__()
        # self.extractor = nnExtractor(input_dim=120, hidden_dim=PARAM['Hidden'])
        self.extractor = lstmExtractor(hidden_dim=PARAM['Hidden'])
        self.predictor0 = Predictor(hidden_dim=PARAM['Hidden'])
        self.predictor1 = Predictor(hidden_dim=PARAM['Hidden'])
        self.propensitynet = PropensityNet(hidden_dim=PARAM['Hidden'])

    def forward(self, pretime, t0_idx, t1_idx):
        emb = self.extractor(pretime)   # (batch, hidden_dim)
        emb0 = emb[t0_idx]    # (batch[t==1], hidden_dim)
        emb1 = emb[t1_idx]    # (batch[t==0], hidden_dim)
        preds0, preds1 = self.predictor0(emb0), self.predictor1(emb1)   # preds: (batch[t==j], posttime_len)
        pred_pro0, pred_pro1 = self.propensitynet(emb0), self.propensitynet(emb1)
        return preds0, preds1, pred_pro0, pred_pro1
