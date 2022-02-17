
__all__ = ["RequiresGradContext", "differential"]


import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Union, List


def judge_requires_grad(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    else:
        raise TypeError


class RequiresGradContext(object):
    def __init__(self, *objs: Union[torch.Tensor, nn.Module], requires_grad: Union[List[bool], bool]):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]
        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError
        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)


def differential(fn, v, retain_graph=None, create_graph=False):
    r""" d fn / dv
    Args:
        fn: a batch of tensor -> a batch of scalar
        v: a batch of tensor
        retain_graph: see autograd.grad, default to create_graph
        create_graph: see autograd.grad
    """
    if retain_graph is None:
        retain_graph = create_graph
    with RequiresGradContext(v, requires_grad=True):
        return autograd.grad(fn(v).sum(), v, retain_graph=retain_graph, create_graph=create_graph)[0]
