import torch
import torch.nn.functional as F

def focal_loss(pred, y, reduction = 'mean', alpha=0.75, gamma=2.0):
    # pred 为 logits
    bce = F.binary_cross_entropy_with_logits(pred, y, reduction='none')
    p = torch.sigmoid(pred) # (n) or (n,k)
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)   # 正样本 alpha 大
    loss = alpha_t * (1 - p_t).pow(gamma) * bce # (n) or (n,k)
    if reduction == 'mean':
        return loss.mean() # final prediction
    else:
        return loss # loss, matrix
