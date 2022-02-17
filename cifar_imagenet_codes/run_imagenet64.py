from interface.runner import run_evaluate_profile
from profiles.ddpm.imagenet64.evaluate import sample2dir, nll, save_ms_eps, nll_ms_eps, sample2dir_ms_eps, save_nll_terms, sample2dir_ddim, sample2dir_ddim_ms_eps
from interface.utils.exp_templates import sample_one_ckpt, nll_one_ckpt, save_ms_eps_one_ckpt, save_nll_terms_one_ckpt
import profiles.ddpm.beta_schedules as beta_schedules
import numpy as np
from interface.utils.dict_utils import merge_dict
_prefix = "L_hybrid"
batch_size_per_card = 125
_n_devices = 8


def _sample_one_ckpt(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir, path="workspace/runner/imagenet64/improved_diffusion",
                    prefix=prefix, names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings,
                    n_devices=n_devices, common_setting=common_setting, devices=devices, time=time)


def _nll_one_ckpt(prefix, names, best_ckpts, settings, nll_settings, n_devices=1, common_setting=None, devices=None, time=None):
    nll_one_ckpt(run_fn=run_evaluate_profile, profile=nll, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                 names=names, ckpts=best_ckpts, settings=settings, nll_settings=nll_settings, n_devices=n_devices,
                 common_setting=common_setting, devices=devices, time=time)


def _save_ms_eps_one_ckpt(prefix, names, best_ckpts, settings, save_ms_eps_settings, n_devices=1, common_setting=None, devices=None, time=None):
    save_ms_eps_one_ckpt(run_fn=run_evaluate_profile, profile=save_ms_eps, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                         names=names, ckpts=best_ckpts, settings=settings, save_ms_eps_settings=save_ms_eps_settings, n_devices=n_devices,
                         common_setting=common_setting, devices=devices, time=time)


def _nll_one_ckpt_ms_eps(prefix, names, best_ckpts, settings, nll_settings, n_devices=1, common_setting=None, devices=None, time=None):
    nll_one_ckpt(run_fn=run_evaluate_profile, profile=nll_ms_eps, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                 names=names, ckpts=best_ckpts, settings=settings, nll_settings=nll_settings, n_devices=n_devices,
                 common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ddim(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ddim, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ms_eps(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ms_eps, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ddim_ms_eps(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ddim_ms_eps, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _save_nll_terms_one_ckpt(prefix, names, best_ckpts, settings, save_nll_terms_settings, n_devices=1, common_setting=None, devices=None, time=None):
    save_nll_terms_one_ckpt(run_fn=run_evaluate_profile, profile=save_nll_terms, path="workspace/runner/imagenet64/improved_diffusion", prefix=prefix,
                            names=names, ckpts=best_ckpts, settings=settings, save_nll_terms_settings=save_nll_terms_settings, n_devices=n_devices,
                            common_setting=common_setting, devices=devices, time=time)


if __name__ == "__main__":
    phase = "sample_analytic_ddim"
    _settings = {"cosine4000": {
        "betas": np.append(0., beta_schedules.get_named_beta_schedule("cosine", 4000))
    }}

    if phase == "save_ms_eps":  # save \Gamma
        _save_ms_eps_settings = []
        for n_samples in [10000]:
            _save_ms_eps_settings.append({"n_samples": n_samples})

        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        _save_ms_eps_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                              best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                              settings=list(_settings.values()), save_ms_eps_settings=_save_ms_eps_settings, n_devices=_n_devices,
                              common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_analytic_ddpm":
        _sample_settings = []
        for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
            _sample_settings.append({"tag": "ms_eps", "steps_type": "linear", "sample_steps": sample_steps,
                                     "n_samples": 50000, "clip_sigma_idx": 1, "clip_pixel": 1, "seed": 1234})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        names_ms_eps_path = {
            "cosine4000": {
                "ms_eps_path": "workspace/runner/imagenet64/improved_diffusion/"
                               "L_hybrid_2021-08-29-20-26-00/cosine4000/train/ms_eps/"
                               "imagenet64_uncond_100M_1500K_n_samples_10000.ms_eps.pth"
            },
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _sample_one_ckpt_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                                best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                                common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_analytic_ddim":
        _sample_settings = []
        for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
            _sample_settings.append({"tag": "ddim_ms_eps", "sample_steps": sample_steps, "n_samples": 50000,
                                     "eta": 0., "clip_sigma_idx": 1, "clip_pixel": 1, "seed": 1234})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        names_ms_eps_path = {
            "cosine4000": {
                "ms_eps_path": "workspace/runner/imagenet64/improved_diffusion/"
                               "L_hybrid_2021-08-29-20-26-00/cosine4000/train/ms_eps/"
                               "imagenet64_uncond_100M_1500K_n_samples_10000.ms_eps.pth"
            },
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _sample_one_ckpt_ddim_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                                     best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                     settings=list(_settings.values()), sample_settings=_sample_settings,
                                     n_devices=_n_devices,
                                     common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "nll_analytic_ddpm":
        _nll_settings = []
        for steps_type in ["dp_seg", "linear"]:
            for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
                _nll_settings.append({"tag": "ms_eps", "steps_type": steps_type, "sample_steps": sample_steps,
                                      "n_samples": None, "partition": "test"})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        names_ms_eps_path = {
            "cosine4000": {
                "ms_eps_path": "workspace/runner/imagenet64/improved_diffusion/"
                               "L_hybrid_2021-08-29-20-26-00/cosine4000/train/ms_eps/"
                               "imagenet64_uncond_100M_1500K_n_samples_10000.ms_eps.pth"
            },
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _nll_one_ckpt_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                             best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())], n_devices=_n_devices,
                             settings=list(_settings.values()), nll_settings=_nll_settings,
                             common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_ddpm":
        _sample_settings = []
        for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
            _sample_settings.append({"sample_steps": sample_steps, "n_samples": 50000, "small_sigma": True, "seed": 1234})
            _sample_settings.append({"sample_steps": sample_steps, "n_samples": 50000, "small_sigma": False, "clip_sigma_idx": 1, "clip_pixel": 1, "seed": 1234})

        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        _sample_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                         best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                         settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                         common_setting={"batch_size": _n_devices * batch_size_per_card}, devices=[0, 1, 2, 5])

    elif phase == "sample_ddim":
        _sample_settings = []
        for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
            _sample_settings.append({"tag": "ddim", "sample_steps": sample_steps, "n_samples": 50000,
                                     "eta": 0., "seed": 1234})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        _sample_one_ckpt_ddim(prefix=_prefix, names=list(_settings.keys()),
                              best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                              settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                              common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "nll_ddpm":
        _nll_settings = []
        for sample_steps in [25, 50, 100, 200, 400, 1000, 4000]:
            for small_sigma in [True, False]:
                _nll_settings.append({"sample_steps": sample_steps, "small_sigma": small_sigma,
                                      "n_samples": None, "partition": "test"})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        _nll_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                      best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                      settings=list(_settings.values()), nll_settings=_nll_settings, n_devices=_n_devices,
                      common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == 'save_nll_terms':  # used in DP for big_sigma
        _save_nll_terms_settings = []
        for partition, n_samples in [("train", 16384), ("test", None)]:
            _save_nll_terms_settings.append({"small_sigma": False, "n_samples": n_samples, "partition": partition})
        names_best_ckpts = {
            "cosine4000": "imagenet64_uncond_100M_1500K.ckpt.pth",
        }
        _save_nll_terms_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                                 best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                 settings=list(_settings.values()), save_nll_terms_settings=_save_nll_terms_settings,
                                 n_devices=_n_devices, common_setting={"batch_size": _n_devices * batch_size_per_card})
