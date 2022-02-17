import torch.nn as nn
from typing import Union, Iterator
import torch
from .device_utils import device_of, global_device


def grad_norm_inf(inputs: Union[nn.Module, Iterator[torch.Tensor]]) -> float:
    if isinstance(inputs, nn.Module):
        inputs = inputs.parameters()
    s = float("-inf")
    for p in inputs:
        if p.grad is not None:
            s = max(s, p.grad.data.abs().max().item())
    return s


def grad_norm(inputs: Union[nn.Module, Iterator[torch.Tensor]], norm_type: float = 2.) -> float:
    if norm_type == float('inf'):
        return grad_norm_inf(inputs)

    if isinstance(inputs, nn.Module):
        inputs = inputs.parameters()
    s = torch.tensor(0., device=next(inputs).device)
    for p in inputs:
        if p.grad is not None:
            s += p.grad.data.pow(norm_type).sum()
    return s.pow(1. / norm_type).item()


def probe_output_shape(model: nn.Module, input_shape):
    inputs = torch.ones(1, *input_shape, device=device_of(model))
    return model(inputs).shape[1:]


def make_xyz(fn, left, right, bottom, top, steps, exp):
    assert left < right
    assert bottom < top
    xs = torch.linspace(left, right, steps=steps)
    ys = torch.linspace(bottom, top, steps=steps)
    xs, ys = torch.meshgrid([xs, ys])
    xs, ys = xs.flatten().unsqueeze(dim=-1), ys.flatten().unsqueeze(dim=-1)
    inputs = torch.cat([xs, ys], dim=-1).to(global_device())
    zs = fn(inputs)
    if exp:
        zs = (zs - zs.max()).exp()
    xs, ys, = xs.view(steps, steps).detach().cpu().numpy(), ys.view(steps, steps).detach().cpu().numpy()
    zs = zs.view(steps, steps).detach().cpu().numpy()
    return xs, ys, zs


def tensor_info(ts, label):
    print(label, '{:.4f}'.format(ts.max().item()), '{:.4f}'.format(ts.min().item()), '{:.4f}'.format(ts.mean().item()))
