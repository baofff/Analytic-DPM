
__all__ = ["clip_grad_norm_", "clip_grad_element_wise_"]


import torch
from typing import List, Union
import math


def clip_grad_norm_(grads: Union[torch.Tensor, List[torch.Tensor]], max_norm: float, norm_type: float = 2.):
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(grad.data.abs().max() for grad in grads)
    else:
        total_norm = 0
        for grad in grads:
            param_norm = grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.data.mul_(clip_coef)
    return total_norm


def clip_grad_element_wise_(grads: Union[torch.Tensor, List[torch.Tensor]], max_norm: float):
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    for grad in grads:
        grad.data.clamp_(-max_norm, max_norm)
