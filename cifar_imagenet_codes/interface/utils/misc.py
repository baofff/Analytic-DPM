
__all__ = ["get_root_by_time", "sample_from_dataset"]


import os
import datetime
import re
import random


def valid_prefix_time(prefix_time: str, prefix: str):
    res = re.search(r"%s_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}" % prefix, prefix_time)
    return res is not None and res.span() == (0, len(prefix_time))


def get_root_by_time(path, prefix, time=None, strategy=None):
    r"""
    Args:
        path: the root is path/prefix_time
        prefix: the root is path/prefix_time
        time: a time tag with the format %Y-%m-%d-%H-%M-%S
        strategy: how to infer the time when time is None (only works when time is None )
    """
    assert strategy is None or strategy in ["latest", "now"]
    if time is None:
        if strategy == "latest":
            _all = filter(lambda s: valid_prefix_time(s, prefix), os.listdir(path))
            latest = sorted(_all)[-1]
            prefix = os.path.join(path, latest)
        elif strategy == "now":
            time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            prefix = os.path.join(path, prefix + "_" + time)
        else:
            raise ValueError
    else:
        prefix = os.path.join(path, prefix + "_" + time)
    return prefix


def sample_from_dataset(dataset):
    idx = random.sample(range(len(dataset)), 1)[0]
    return dataset[idx]
