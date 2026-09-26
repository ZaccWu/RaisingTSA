import torch
import torch.nn.functional as F
import numpy as np

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

def ev_loss(pred, y):
    
    # pred 为 logits
    if y.shape != pred.shape:
        y = y[:, None].expand_as(pred).contiguous()
    
    loss = F.mse_loss(pred, y, reduction='none')
    return loss # loss, matrix

# def ev_loss(pred, y):  
#     EPS = 1e-15
#     # gamma=1.0 version

#     prop_0 = len((1-y).nonzero())  # label = 0
#     prop_1 = len(y.nonzero())      # label = 1
#     pred_score_sigmoid = torch.sigmoid(pred)

#     if y.shape != pred.shape:
#         y = y[:, None].expand_as(pred).contiguous()
#         # 逐元素计算正负样本损失，并按照类别比例加权
#         pos_loss = -torch.log(pred_score_sigmoid + EPS) * (prop_0 / (prop_0 + prop_1)) * y
#         neg_loss = -torch.log(1 - pred_score_sigmoid + EPS) * (prop_1 / (prop_0 + prop_1)) * (1 - y)
#     else:
#         pos_loss = -torch.log(pred_score_sigmoid[y.nonzero()] + EPS).mean() * (prop_0/(prop_0+prop_1))
#         neg_loss = -torch.log(1 - pred_score_sigmoid[(1-y).nonzero()] + EPS).mean() * (prop_1/(prop_0+prop_1))

#     loss = pos_loss + neg_loss  # shape: (num_sample, num_predictor)
#     return loss

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

def find_positions_in_vru(list_a, list_b):
    positions_dict = {}  # 创建一个字典用于存储列B表中每个元素的位置
    for i, element in enumerate(list_b):
        positions_dict[element] = i
    result = []  # 用于存储列表A中每个元素在列表B中的位置
    for element in list_a:
        if element in positions_dict:
            result.append(positions_dict[element])
    return result

def cal_ndcgK(vcu, vru):
    # vru是正序的排名(按照推荐指数)
    position = find_positions_in_vru(vcu, vru)
    dcg = np.sum([1/np.log2(2+i) for i in position])
    inter_len = len(set(vru) & set(vcu))
    idcg = np.sum([1/np.log2(2+i) for i in range(inter_len)])
    if idcg == 0:
        return 0
    return dcg/idcg