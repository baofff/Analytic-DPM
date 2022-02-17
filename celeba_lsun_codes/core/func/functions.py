
__all__ = ["stp", "sos", "mos", "inner_product", "duplicate", "unsqueeze_like", "logsumexp", "log_discretized_normal",
           "binary_cross_entropy_with_logits", "log_bernoulli", "kl_between_normal"]


import numpy as np
import torch.nn.functional as F
import torch


def stp(s: np.ndarray, ts: torch.Tensor):  # scalar tensor product
    s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def sos(a, start_dim=1):  # sum of square
    return a.pow(2).flatten(start_dim=start_dim).sum(dim=-1)


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def inner_product(a, b, start_dim=1):
    return (a * b).flatten(start_dim=start_dim).sum(dim=-1)


def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)


def unsqueeze_like(tensor, template, start="left"):
    if start == "left":
        tensor_dim = tensor.dim()
        template_dim = template.dim()
        assert tensor.shape == template.shape[:tensor_dim]
        return tensor.view(*tensor.shape, *([1] * (template_dim - tensor_dim)))
    elif start == "right":
        tensor_dim = tensor.dim()
        template_dim = template.dim()
        assert tensor.shape == template.shape[-tensor_dim:]
        return tensor.view(*([1] * (template_dim - tensor_dim)), *tensor.shape)
    else:
        raise ValueError


def logsumexp(tensor, dim, keepdim=False):
    # the logsumexp of pytorch is not stable!
    tensor_max, _ = tensor.max(dim=dim, keepdim=True)
    ret = (tensor - tensor_max).exp().sum(dim=dim, keepdim=True).log() + tensor_max
    if not keepdim:
        ret.squeeze_(dim=dim)
    return ret


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def log_discretized_normal(x, mu, var):  # element-wise
    centered_x = x - mu
    std = var ** 0.5
    left = (centered_x - 1. / 255) / std
    right = (centered_x + 1. / 255) / std

    cdf_right = approx_standard_normal_cdf(right)
    cdf_left = approx_standard_normal_cdf(left)
    cdf_delta = cdf_right - cdf_left

    return torch.where(
        x < -0.999,
        cdf_right.clamp(min=1e-12).log(),
        torch.where(x > 0.999, (1. - cdf_left).clamp(min=1e-12).log(), cdf_delta.clamp(min=1e-12).log()),
    )


def binary_cross_entropy_with_logits(logits, inputs):
    r""" -inputs * log (sigmoid(logits)) - (1 - inputs) * log (1 - sigmoid(logits)) element wise
        with automatically expand dimensions
    """
    if inputs.dim() < logits.dim():
        inputs = inputs.expand_as(logits)
    else:
        logits = logits.expand_as(inputs)
    return F.binary_cross_entropy_with_logits(logits, inputs, reduction="none")


def log_bernoulli(inputs, logits, n_data_dim):
    return -binary_cross_entropy_with_logits(logits, inputs).flatten(-n_data_dim).sum(dim=-1)


def kl_between_normal(mu_0, var_0, mu_1, var_1):  # element-wise
    tensor = None
    for obj in (mu_0, var_0, mu_1, var_1):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None

    var_0, var_1 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (var_0, var_1)
    ]

    return 0.5 * (var_0 / var_1 + (mu_0 - mu_1).pow(2) / var_1 + var_1.log() - var_0.log() - 1.)
