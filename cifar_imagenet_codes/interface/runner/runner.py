
__all__ = ["run_train_profile", "run_evaluate_profile", "run_timing_profile"]


from interface.runner.fit import naive_fit
from interface.runner.timing import timing
from interface.utils import set_seed, set_deterministic, backup_codes, backup_profile
from interface.utils import ckpt, profile_utils, dict_utils
from interface.utils.dict_utils import get_val
from core.utils import global_device
from core.utils.managers import ModelsManager


def merge_models(dest: ModelsManager, src: ModelsManager):
    if src is None:
        return dest
    _dict = {key: dest.__dict__[key] for key in dest.__dict__.keys()}
    for key, val in src.__dict__.items():
        _dict[key] = val
    return ModelsManager(**_dict)


def run_train_profile(profile: dict):
    r"""
    Args:
        profile: a parsed or unparsed profile
    """
    profile = dict_utils.parse_self_ref_dict(profile)
    set_seed(get_val(profile, "seed", default=None))
    set_deterministic(get_val(profile, "deterministic", default=False))
    backup_codes(profile["backup_root"])
    backup_profile(profile, profile["backup_root"])

    interact = profile_utils.create_interact(profile["interact"])
    interact.report_machine()

    models = profile_utils.create_models(profile["models"])
    ema_models, ema_rate = profile_utils.create_ema(profile, models)
    optimizers = profile_utils.create_optimizers(profile["optimizers"], models)
    lr_schedulers = profile_utils.create_lr_schedulers(profile.get("lr_schedulers", {}), optimizers)
    criterion = profile_utils.create_criterion(profile["criterion"], models, optimizers, lr_schedulers)

    dataset = profile_utils.create_dataset(profile["dataset"])
    if profile["dataset"].get("use_val", True):
        train_dataset = dataset.get_train_data()
        val_dataset = dataset.get_val_data()
    else:
        train_dataset = dataset.get_train_val_data()
        val_dataset = None

    evaluator = None
    if "evaluator" in profile:
        evaluator = profile_utils.create_evaluator(profile["evaluator"], merge_models(models, ema_models), dataset, interact)

    naive_fit(criterion=criterion,
              train_dataset=train_dataset,
              batch_size=get_val(profile, "training", "batch_size"),
              n_its=get_val(profile, "training", "n_its"),
              n_ckpts=get_val(profile, "training", "n_ckpts", default=10),
              ckpt_root=profile["ckpt_root"],
              interact=interact,
              evaluator=evaluator,
              val_dataset=val_dataset,
              val_fn=profile_utils.create_val_fn(profile, criterion),
              ckpt=ckpt.get_ckpt_by_it(profile["ckpt_root"]),
              ema_models=ema_models,
              ema_rate=ema_rate
              )


def run_evaluate_profile(profile: dict):
    r"""
    Args:
        profile: a parsed or unparsed profile
    """
    profile = dict_utils.parse_self_ref_dict(profile)
    set_seed(get_val(profile, "seed", default=None))
    set_deterministic(get_val(profile, "deterministic", default=False))
    backup_codes(profile["backup_root"])
    backup_profile(profile, profile["backup_root"])

    interact = profile_utils.create_interact(profile["interact"])
    interact.report_machine()

    models = profile_utils.create_models(profile["models"])
    ckpt_path = profile['ckpt_path'] if isinstance(profile['ckpt_path'], list) else [profile['ckpt_path']]
    for path in ckpt_path:
        if profile.get("ema", False):
            ckpt.CKPT().load(path).to_ema_models(models)
        else:
            ckpt.CKPT().load(path).to_models(models)
    dataset = profile_utils.create_dataset(profile["dataset"])

    evaluator = profile_utils.create_evaluator(profile["evaluator"], models, dataset, interact)
    models.to(global_device())
    models.eval()
    evaluator.evaluate()


def run_timing_profile(profile: dict):
    r"""
    Args:
        profile: a parsed or unparsed profile
    """
    profile = dict_utils.parse_self_ref_dict(profile)
    set_seed(get_val(profile, "seed", default=None))
    set_deterministic(get_val(profile, "deterministic", default=False))
    backup_codes(profile["backup_root"])
    backup_profile(profile, profile["backup_root"])

    interact = profile_utils.create_interact(profile["interact"])
    interact.report_machine()

    models = profile_utils.create_models(profile["models"])
    optimizers = profile_utils.create_optimizers(profile["optimizers"], models)
    lr_schedulers = profile_utils.create_lr_schedulers(profile.get("lr_schedulers", {}), optimizers)
    criterion = profile_utils.create_criterion(profile["criterion"], models, optimizers, lr_schedulers)

    dataset = profile_utils.create_dataset(profile["dataset"])
    if profile["dataset"].get("use_val", True):
        train_dataset = dataset.get_train_data()
    else:
        train_dataset = dataset.get_train_val_data()

    timing(criterion=criterion,
           train_dataset=train_dataset,
           batch_size=get_val(profile, "training", "batch_size"),
           n_its=get_val(profile, "training", "n_its")
           )
