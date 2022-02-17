from . import diagnose
from . import managers
from .clip_grad import *
from .device_utils import *


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]
