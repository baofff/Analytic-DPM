import os
import logging
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def global_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_logger(fname):
    os.makedirs(os.path.split(fname)[0], exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def cnt_png(path):
    png_files = filter(lambda x: x.endswith(".png"), os.listdir(path))
    return len(list(png_files))


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(path, n_samples, batch_size, sample_fn, unpreprocess_fn=None, persist=True):
    os.makedirs(path, exist_ok=True)
    idx = n_png = cnt_png(path) if persist else 0
    for _batch_size in amortize(n_samples - n_png, batch_size):
        samples = sample_fn(_batch_size)
        samples = unpreprocess_fn(samples)
        for sample in samples:
            Image.fromarray(sample).save(os.path.join(path, "{}.png".format(idx)))
            idx += 1


def score_on_dataset(dataset: Dataset, score_fn, batch_size):
    r"""
    Args:
        dataset: an instance of Dataset
        score_fn: a batch of data -> a batch of scalars
        batch_size: the batch size
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    total_score = None
    tuple_output = None
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for idx, v in enumerate(dataloader):
        v = v.to(device)
        score = score_fn(v)
        if idx == 0:
            tuple_output = isinstance(score, tuple)
            total_score = (0.,) * len(score) if tuple_output else 0.
        if tuple_output:
            total_score = tuple([a + b.sum().detach().item() for a, b in zip(total_score, score)])
        else:
            total_score += score.sum().detach().item()
    if tuple_output:
        mean_score = tuple([a / len(dataset) for a in total_score])
    else:
        mean_score = total_score / len(dataset)
    return mean_score
