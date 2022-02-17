
__all__ = ["device_of", "global_device"]


import torch.nn as nn
import torch
from typing import Union
from .managers import ModelsManager


def device_of(inputs: Union[nn.Module, torch.Tensor, ModelsManager]) -> torch.device:
    if isinstance(inputs, nn.Module):
        return next(inputs.parameters()).device
    elif isinstance(inputs, torch.Tensor):
        return inputs.device
    elif isinstance(inputs, ModelsManager):
        return device_of(next(iter(ModelsManager.__dict__.values())))
    else:
        raise TypeError


def global_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
