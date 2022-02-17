from .fit import infinite_loader
import logging
import time
from interface.utils.task_schedule import gpu_memory_consumption


def timing(criterion, train_dataset, batch_size, n_its):
    r""" Loops of Learning
    Args:
        criterion: a Criterion instance
        train_dataset: the training dataset
        batch_size: the batch size of training
        n_its: the number of iterations
    """
    criterion.models.to(criterion.device)

    it = 0
    train_dataset_loader = infinite_loader(train_dataset, batch_size=batch_size)

    st = time.time()
    while it < n_its:
        criterion.models.train()
        criterion.update(train_dataset_loader)
        it += 1
    ed = time.time()

    logging.info("%d iterations take %.2f s" % (n_its, ed - st))
    logging.info("Taking %d MB" % gpu_memory_consumption())
