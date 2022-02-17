import torch
import os
import core.utils.managers as managers
import logging


class CKPT(object):
    def __init__(self, it=None, best_val_loss=None, models_states=None, optimizers_states=None, lr_schedulers_states=None,
                 ema_models_states=None):
        r""" Record the states of training
        Args:
            it: iteration
            best_val_loss: the best validation loss
            models_states: a dict of state_dicts of models
            optimizers_states: a dict of state_dicts of optimizers
            lr_schedulers_states: a dict of state_dicts of lr_schedulers
            ema_models_states: a dict of state_dicts of ema_models
        """
        self.it = it
        self.best_val_loss = best_val_loss
        self.models_states = models_states
        self.optimizers_states = optimizers_states
        self.lr_schedulers_states = lr_schedulers_states
        self.ema_models_states = ema_models_states

    def save(self, fname):
        logging.info("save ckpt to {}".format(fname))
        torch.save(self.__dict__, fname)

    def load(self, fname):
        logging.info("load ckpt from {}".format(fname))
        ckpt = torch.load(fname)
        for k, val in ckpt.items():
            self.__dict__[k] = val
        return self

    def from_criterion(self, criterion):
        self.models_states = criterion.models.get_states()
        self.optimizers_states = criterion.optimizers.get_states()
        self.lr_schedulers_states = criterion.lr_schedulers.get_states()
        return self

    def to_criterion(self, criterion):
        criterion.models.load_states(self.models_states)
        criterion.optimizers.load_states(self.optimizers_states)
        criterion.lr_schedulers.load_states(self.lr_schedulers_states)

    def to_models(self, models: managers.ModelsManager):
        logging.info("load models_states")
        models.load_states(self.models_states)

    def to_ema_models(self, ema_models: managers.ModelsManager):
        logging.info("load ema_models_states")
        ema_models.load_states(self.ema_models_states)


def list_ckpts(ckpt_root):
    fnames = list(filter(lambda x: x.endswith(".ckpt.pth"), os.listdir(ckpt_root)))
    fnames = sorted(fnames, key=lambda x: int(x.split(".")[0]))
    return fnames


def get_ckpt_by_it(ckpt_root, it=None):
    r""" Get the ckpt at a iteration 'it' from ckpt_root
        If 'it' is None, try to get the latest ckpt
        If 'it' is None and there is no ckpt, return None

    Args:
        ckpt_root: the root of ckpts
        it: the iteration
    """
    if not os.path.exists(ckpt_root):
        return None
    if it is None:
        fnames = list_ckpts(ckpt_root)
        if fnames:
            return CKPT().load(os.path.join(ckpt_root, fnames[-1]))
        else:
            return None
    else:
        return CKPT().load(os.path.join(ckpt_root, "%d.ckpt.pth" % it))
