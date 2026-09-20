import torch
import torch.nn.functional as F

def focal_loss(pred, y, reduction = 'mean', alpha=0.75, gamma=2.0):
    # pred 为 logits
    if y.shape != pred.shape:
        y = y.expand_as(pred).contiguous()
    bce = F.binary_cross_entropy_with_logits(pred, y, reduction='none')
    p = torch.sigmoid(pred) # (n) or (n,k)
    p_t = p * y + (1 - p) * (1 - y)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)   # 正样本 alpha 大
    loss = alpha_t * (1 - p_t).pow(gamma) * bce # (n) or (n,k)
    if reduction == 'mean':
        return loss.mean() # final prediction
    else:
        return loss # loss, matrix

def transfer_pred(out, threshold):
    pred = out.clone()
    pred[torch.where(out < threshold)] = 0
    pred[torch.where(out >= threshold)] = 1
    return pred

def huber_loss(pred, target, delta=1.0, reduction='mean'):
    """连续回归损失，对离群点稳健"""
    return F.huber_loss(pred, target, delta=delta, reduction=reduction)


def pairwise_ranking_loss(pred, target, min_target_diff=0.1):
    """
    Batch 内相邻对排序损失：
    按 target 降序排列后，相邻样本应保持 pred 的降序。
    只用相对顺序，不依赖绝对边界，适合稀有正样本。
    """
    B = pred.size(0)
    if B < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    _, idx = torch.sort(target, descending=True)
    pred_sorted = pred[idx]
    target_sorted = target[idx]

    pred_diff = pred_sorted[:-1] - pred_sorted[1:]         # 应 > 0
    target_diff = target_sorted[:-1] - target_sorted[1:]   # >= 0

    valid = target_diff > min_target_diff
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # softplus(-x) = log(1 + exp(-x))：x 越大损失越小
    return F.softplus(-pred_diff[valid]).mean()

def topk_recall(y_bin, y_pred, k, group_size):
    assert y_bin.shape[0] % group_size == 0, \
        f"N_total={y_bin.shape[0]} 不能被 group_size={group_size} 整除"
    n_groups = y_bin.shape[0] // group_size
    y_bin_g  = y_bin.view(n_groups, group_size)
    y_pred_g = y_pred.view(n_groups, group_size)
    k = min(k, group_size)

    _, topk_idx = torch.topk(y_pred_g, k, dim=1)
    topk_labels = y_bin_g.gather(1, topk_idx)

    n_pos = y_bin_g.sum(dim=1)
    hits  = topk_labels.sum(dim=1)
    valid = n_pos > 0
    if not valid.any():
        return 0.0
    recalls = hits[valid] / n_pos[valid]
    return float(recalls.mean().item())


def ndcg_at_k(y_bin, y_pred, k, group_size, use_exp_gain=True):
    assert y_bin.shape[0] % group_size == 0, \
        f"N_total={y_bin.shape[0]} 不能被 group_size={group_size} 整除"
    n_groups = y_bin.shape[0] // group_size
    y_bin_g  = y_bin.view(n_groups, group_size).float()
    y_pred_g = y_pred.view(n_groups, group_size)
    k = min(k, group_size)

    _, topk_idx = torch.topk(y_pred_g, k, dim=1)
    gains = y_bin_g.gather(1, topk_idx)
    if use_exp_gain:
        gains = 2.0 ** gains - 1.0
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))
    dcg = (gains * discounts).sum(dim=1)

    sorted_true, _ = torch.sort(y_bin_g, dim=1, descending=True)
    ideal_gains = sorted_true[:, :k]
    if use_exp_gain:
        ideal_gains = 2.0 ** ideal_gains - 1.0
    idcg = (ideal_gains * discounts).sum(dim=1)

    ndcg = torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))
    return float(ndcg.mean().item())