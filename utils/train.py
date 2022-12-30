import random

import numpy as np
import torch
import torch.nn.functional as F

def fix_seed(seed):
    # random seed initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k*100)
    return res
