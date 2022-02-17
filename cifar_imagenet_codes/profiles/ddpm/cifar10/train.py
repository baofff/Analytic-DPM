
__all__ = ["train", "train_ddpm_dsm"]


import core.criterions as criterions
from interface.utils import dict_utils
import profiles.common as common
from .base import unet_model, dataset, default_betas, ddpm_naive_evaluator_train
import torch.optim as optim


train = {
    "seed": 1234,
    "betas": default_betas,
    "rescale_timesteps": True,
    "ckpt_root": "os.path.join($(workspace_root), 'train/ckpts/')",
    "backup_root": "os.path.join($(workspace_root), 'train/reproducibility/')",
    "training": {
        "n_ckpts": 50,
        "n_its": 500000,
        "batch_size": 128,
    },
    "ema": {
        "rate": 0.9999
    },
    "dataset": dataset,
    "optimizers": {
        "all": {
            "class": optim.AdamW,
            "kwargs": {
                "lr": 0.0001,
                "weight_decay": 0.
            }
        }
    },
    "interact": common.interact_datetime_train(period=10),
}


train_ddpm_dsm = dict_utils.merge_dict(train, {
    "models": {
        "eps_model": unet_model,
    },
    "criterion": {
        "class": criterions.DDPMDSM,
        "kwargs": {
            "betas": "$(betas)",
            "rescale_timesteps": "$(rescale_timesteps)",
        }
    },
    "evaluator": ddpm_naive_evaluator_train
})
