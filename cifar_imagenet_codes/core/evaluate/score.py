
__all__ = ["score_on_dataset"]

import torch
from torch.utils.data import DataLoader, Dataset


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
