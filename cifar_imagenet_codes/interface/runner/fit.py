from torch.utils.data import DataLoader
from interface.utils.ckpt import CKPT
import os
import logging
import math
from core.utils.managers import ModelsManager


def infinite_loader(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    while True:
        for data in loader:
            yield data


def check_anomaly(statistics: dict, it: int):
    for k, v in statistics.items():
        if math.isnan(float(v)):
            statistics_str = {k: "{:.5e}".format(float(statistics[k])) for k in statistics}
            logging.info('Exit at it {}, {}'.format(it, statistics_str))
            exit(0)


def naive_fit(criterion, train_dataset, batch_size, n_its, n_ckpts, ckpt_root, interact,
              evaluator=None, val_dataset=None, val_fn=None, ckpt=None,
              ema_models: ModelsManager = None, ema_rate=None):
    r""" Loops of Learning
    Args:
        criterion: a Criterion instance
        train_dataset: the training dataset
        batch_size: the batch size of training
        n_its: the number of iterations
        n_ckpts: the number of ckpts to save
        ckpt_root: the directory root of ckpts
        interact: an Interact instance
        evaluator: an Evaluator instance
        val_dataset: the validation dataset
        val_fn: the function used for validation
        ckpt: a CKPT instance
        ema_models: the exponential moving average models
        ema_rate: theta <- rate * theta + (1 - rate) * theta_src
    """
    os.makedirs(ckpt_root, exist_ok=True)
    criterion.models.to(criterion.device)
    if ema_models is not None:
        ema_models.to(criterion.device)

    it = 0
    best_val_loss = float('inf')  # the smaller the better
    if ckpt is not None:
        it = ckpt.it
        best_val_loss = ckpt.best_val_loss
        ckpt.to_criterion(criterion)
        if ema_models is not None:
            ckpt.to_ema_models(ema_models)
        del ckpt

    logging.info("Start fitting, it=%d" % it)

    train_dataset_loader = infinite_loader(train_dataset, batch_size=batch_size)
    period = n_its // n_ckpts  # the period of saving ckpts

    while it < n_its:
        criterion.models.train()
        criterion.update(train_dataset_loader)
        if ema_models is not None:  # exponential moving average
            ema_models.ema(criterion.models, rate=ema_rate)
        it += 1

        interact.report_train(criterion.statistics, it)
        check_anomaly(criterion.statistics, it)
        if evaluator is not None:
            criterion.models.eval()
            evaluator.evaluate_train(it)

        if it % period == 0 or it == n_its:
            criterion.models.eval()
            if val_dataset is not None and val_fn is not None:  # validation
                loss = val_fn(dataset=val_dataset)
                interact.report_val(loss, it)
                if loss < best_val_loss:
                    CKPT(models_states=criterion.models.get_states()). \
                        save(os.path.join(ckpt_root, "best.pth"))  # update the best model
                    best_val_loss = loss
            CKPT(it=it, best_val_loss=best_val_loss, ema_models_states=None if ema_models is None else ema_models.get_states()).\
                from_criterion(criterion).save(os.path.join(ckpt_root, "%d.ckpt.pth" % it))  # save ckpt

    logging.info("Finish fitting, it=%d" % it)
