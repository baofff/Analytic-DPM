
__all__ = ["grid_sample", "sample2dir"]

import os
import torch
from torchvision.utils import make_grid, save_image
from core.utils import amortize


def grid_sample(fname, nrow, ncol, sample_fn, unpreprocess_fn=None):
    r""" Sample images in a grid
    Args:
        fname: the file name
        nrow: the number of rows of the grid
        ncol: the number of columns of the grid
        sample_fn: the sampling function, n_samples -> samples
        unpreprocess_fn: the function to unpreprocess data
    """
    root, name = os.path.split(fname)
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "tensor"), exist_ok=True)
    n_samples = nrow * ncol
    samples = sample_fn(n_samples)
    if unpreprocess_fn is not None:
        samples = unpreprocess_fn(samples)
    grid = make_grid(samples, nrow)
    save_image(grid, fname)
    torch.save(samples, os.path.join(root, "tensor", "%s.pth" % name))  # save the tensor data


def cnt_png(path):
    png_files = filter(lambda x: x.endswith(".png"), os.listdir(path))
    return len(list(png_files))


def sample2dir(path, n_samples, batch_size, sample_fn, unpreprocess_fn=None, persist=True):
    os.makedirs(path, exist_ok=True)
    idx = n_png = cnt_png(path) if persist else 0
    for _batch_size in amortize(n_samples - n_png, batch_size):
        samples = sample_fn(_batch_size)
        samples = unpreprocess_fn(samples)
        for sample in samples:
            save_image(sample, os.path.join(path, "{}.png".format(idx)))
            idx += 1
