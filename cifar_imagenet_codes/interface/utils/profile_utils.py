import copy
import traceback
import core.utils.managers as managers
import torch.optim as optim
from .interact import Interact
from .dict_utils import get_val
from core.evaluate import score_on_dataset
import functools
import torch


def _is_instance_profile(profile):
    r""" Judge whether the profile defines an instance of a class
    """
    return isinstance(profile, dict) and "class" in profile


def _create_instance_recursively(profile):
    assert _is_instance_profile(profile)
    kwargs = profile.get("kwargs", {})
    for k, val in kwargs.items():
        if _is_instance_profile(val):
            kwargs[k] = _create_instance_recursively(val)
    try:
        return profile["class"](**kwargs)
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


################################################################################
# Create models from a profile
################################################################################

def create_model(profile: dict):
    r""" Create an instance of the model described in the profile
    Args:
        profile: a parsed profile describing the model
    """
    return _create_instance_recursively(copy.deepcopy(profile))


def create_models(profile: dict):
    r""" Create models (an instance of ModelsManager) described in the profile
    Args:
        profile: a parsed profile describing models
    """
    profile = copy.deepcopy(profile)
    models = {}
    for k, val in profile.items():
        models[k] = create_model(val)
        if "init_ckpt_path" in val:
            path = val["init_ckpt_path"]
            models[k].load_state_dict(torch.load(path)["models_states"][k])
    return managers.ModelsManager(**models)


################################################################################
# Create optimizers from a profile
################################################################################

def create_optimizer(profile: dict, models: managers.ModelsManager):
    r""" Create an instance of the optimizer described in the profile
    Args:
        profile: a parsed profile describing the optimizer
            Example: { "class": optim.Adam,
                       "model_keys": ["lvm", "q"],
                       "kwargs": { "lr": 0.0001 } }
            If 'model_keys' is missing, the corresponding optimizer will include all parameters
        models: an object of ModelsManager
    """
    assert _is_instance_profile(profile)
    params = models.parameters(*profile.get("model_keys", []))
    try:
        return profile["class"](params, **profile["kwargs"])
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


def create_optimizers(profile: dict, models: managers.ModelsManager):
    r""" Create optimizers (an instance of OptimizersManager) described in the profile
    Args:
        profile: a parsed profile describing optimizers
        models: an object of ModelsManager
    """
    profile = copy.deepcopy(profile)
    optimizers = {}
    for k, val in profile.items():
        optimizers[k] = create_optimizer(val, models)
    return managers.OptimizersManager(**optimizers)


################################################################################
# Create lr_schedulers from a profile
################################################################################

def create_lr_scheduler(profile: dict, optimizer: optim.Optimizer):
    r""" Create an instance of the optimizer described in the profile
    Args:
        profile: a parsed profile describing the optimizer
        optimizer: the optimizer to apply
    """
    assert _is_instance_profile(profile)
    try:
        return profile["class"](optimizer, **profile["kwargs"])
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


def create_lr_schedulers(profile: dict, optimizers: managers.OptimizersManager):
    r""" Create optimizers (an instance of OptimizersManager) described in the profile
    Args:
        profile: a parsed profile describing optimizers
        optimizers: an object of OptimizersManager
    """
    profile = copy.deepcopy(profile)
    lr_schedulers = {}
    for k, val in profile.items():
        lr_schedulers[k] = create_lr_scheduler(val, optimizers.get(k))
    return managers.LRSchedulersManager(**lr_schedulers)


################################################################################
# Create criterion from a profile
################################################################################

def create_criterion(profile: dict,
                     models: managers.ModelsManager,
                     optimizers: managers.OptimizersManager,
                     lr_schedulers: managers.LRSchedulersManager):
    r""" Create an instance of the criterion described in the profile
    Args:
        profile: a parsed profile describing the criterion
        models: an object of ModelsManager
        optimizers: an object of OptimizersManager
        lr_schedulers: an object of LRSchedulersManager
    """
    assert _is_instance_profile(profile)
    try:
        return profile["class"](**profile.get("kwargs", {}),
                                models=models,
                                optimizers=optimizers,
                                lr_schedulers=lr_schedulers)
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


################################################################################
# Create dataset from a profile
################################################################################

def create_dataset(profile):
    assert _is_instance_profile(profile)
    try:
        return profile["class"](**profile["kwargs"])
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


################################################################################
# Create evaluator from a profile
################################################################################

def create_evaluator(profile, models, dataset, interact):
    assert _is_instance_profile(profile)
    try:
        return profile["class"](**profile["kwargs"], models=models, dataset=dataset, interact=interact)
    except TypeError:
        traceback.print_exc()
        print(profile["class"])
        exit(1)


################################################################################
# Create interact from a profile
################################################################################

def create_interact(profile: dict) -> Interact:
    return Interact(**profile)


################################################################################
# Create the validation function
################################################################################

def create_val_fn(profile, criterion):
    if profile.get("disable_val_fn", False):  # no val_fn
        return None
    elif "val_fn" not in profile:  # default val_fn
        return functools.partial(score_on_dataset, score_fn=criterion.default_val_fn,
                                 batch_size=get_val(profile, "training", "batch_size"))
    else:
        profile_val_fn = profile["val_fn"]
        batch_size = get_val(profile_val_fn, "batch_size", default=get_val(profile, "training", "batch_size"))
        kwargs = profile_val_fn.get("kwargs", {})
        apply_to = profile_val_fn.get("apply_to", "tensor")
        if apply_to == "tensor":
            def score_fn(v):
                return profile_val_fn["fn"](models=criterion.models, v=v, **kwargs)
            return functools.partial(score_on_dataset, score_fn=score_fn, batch_size=batch_size)
        elif apply_to == "dataset":
            return functools.partial(profile_val_fn["fn"], models=criterion.models, batch_size=batch_size)
        else:
            raise ValueError


################################################################################
# Create ema
################################################################################

def create_ema(profile, models):
    if profile.get("disable_ema", False):
        return None, None
    else:
        ema_keys = get_val(profile, "ema", "keys", default=list(models.__dict__.keys()))
        ema_rate = get_val(profile, "ema", "rate", default=0.9999)
        ema_models = create_models({key: profile["models"][key] for key in ema_keys})
        ema_models.ema(models, rate=0)
        return ema_models, ema_rate
