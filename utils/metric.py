import numpy as np
import torch
from tqdm import tqdm
# from utils.util import convert_all_data_to_gpu
import datetime


def recall_score(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=40)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = (torch.logical_and(torch.logical_and (predict,truth), truth == 1)).sum(-1), truth.sum(-1)
    return (tp.float() / t.float()).sum().item()


def dcg(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:

    """
    value, predict_indices = y_pred.topk(k=40)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true, y_pred, top_k):
    """
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:

    """
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    return (dcg_score / idcg_score).sum().item()


def PHR(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    value, predict_indices = y_pred.topk(k=40)
    value = value[:, :top_k]
    predict_indices = predict_indices[:, :top_k]
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    hit_num = torch.logical_and(predict, truth).sum(dim=1).nonzero(as_tuple=False).shape[0]
    return hit_num


def get_metric(y_true, y_pred):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict
    """
    result = {}
    for top_k in [10, 20, 30, 40]:
        result.update({
            f'recall_{top_k}': recall_score(y_true, y_pred, top_k=top_k),
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    return result