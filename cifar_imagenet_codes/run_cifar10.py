from interface.runner import run_train_profile, run_evaluate_profile
from profiles.ddpm.cifar10 import train_ddpm_dsm
from profiles.ddpm.cifar10.naive_evaluate import sample2dir_eps, sample2dir_ms_eps, sample2dir_ddim, \
    nll_eps, nll_ms_eps, save_ms_eps, save_nll_terms, sample2dir_ddim_ms_eps
from interface.utils.exp_templates import run_on_different_settings, sample_ckpts, sample_one_ckpt, nll_one_ckpt, \
    save_ms_eps_one_ckpt, save_nll_terms_one_ckpt, _dict2str
import profiles.ddpm.beta_schedules as beta_schedules
import numpy as np
from interface.utils.dict_utils import merge_dict
_prefix = "betas"
batch_size_per_card = 500
_n_devices = 8


def _run_on_different_setting_train(names, settings, prefix, n_devices=1, common_setting=None, devices=None, time=None):
    run_on_different_settings(run_fn=run_train_profile, profile=train_ddpm_dsm, path="workspace/runner/cifar10/ddpm_dsm",
                              prefix=prefix, names=names, settings=settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="now")


def _sample_ckpts(prefix, names, settings, ckpts, dirname, n_devices=1, common_setting=None, devices=None, time=None):
    r""" 1000 samples for validation
    """
    sample_ckpts(run_fn=run_evaluate_profile, profile=sample2dir_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                 names=names, settings=settings, ckpts=ckpts, dirname=dirname, n_devices=n_devices,
                 common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ddim(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ddim, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ms_eps(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ms_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _sample_one_ckpt_ddim_ms_eps(prefix, names, best_ckpts, settings, sample_settings, n_devices=1, common_setting=None, devices=None, time=None):
    sample_one_ckpt(run_fn=run_evaluate_profile, profile=sample2dir_ddim_ms_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                    names=names, ckpts=best_ckpts, settings=settings, sample_settings=sample_settings, n_devices=n_devices,
                    common_setting=common_setting, devices=devices, time=time)


def _nll_one_ckpt(prefix, names, best_ckpts, settings, nll_settings, n_devices=1, common_setting=None, devices=None, time=None):
    nll_one_ckpt(run_fn=run_evaluate_profile, profile=nll_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                 names=names, ckpts=best_ckpts, settings=settings, nll_settings=nll_settings, n_devices=n_devices,
                 common_setting=common_setting, devices=devices, time=time)


def _nll_one_ckpt_ms_eps(prefix, names, best_ckpts, settings, nll_settings, n_devices=1, common_setting=None, devices=None, time=None):
    nll_one_ckpt(run_fn=run_evaluate_profile, profile=nll_ms_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                 names=names, ckpts=best_ckpts, settings=settings, nll_settings=nll_settings, n_devices=n_devices,
                 common_setting=common_setting, devices=devices, time=time)


def _save_ms_eps_one_ckpt(prefix, names, best_ckpts, settings, save_ms_eps_settings, n_devices=1, common_setting=None, devices=None, time=None):
    save_ms_eps_one_ckpt(run_fn=run_evaluate_profile, profile=save_ms_eps, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                         names=names, ckpts=best_ckpts, settings=settings, save_ms_eps_settings=save_ms_eps_settings, n_devices=n_devices,
                         common_setting=common_setting, devices=devices, time=time)


def _save_nll_terms_one_ckpt(prefix, names, best_ckpts, settings, save_nll_terms_settings, n_devices=1, common_setting=None, devices=None, time=None):
    save_nll_terms_one_ckpt(run_fn=run_evaluate_profile, profile=save_nll_terms, path="workspace/runner/cifar10/ddpm_dsm", prefix=prefix,
                            names=names, ckpts=best_ckpts, settings=settings, save_nll_terms_settings=save_nll_terms_settings, n_devices=n_devices,
                            common_setting=common_setting, devices=devices, time=time)


if __name__ == "__main__":
    phase = "sample_analytic_ddim"
    _settings = {}
    for beta_schedule, num_diffusion in [("linear", 1000), ("cosine", 1000)]:
        _key = "beta_schedule_%s_num_diffusion_%d" % (beta_schedule, num_diffusion)
        _settings[_key] = {
            "betas": np.append(0., beta_schedules.get_named_beta_schedule(beta_schedule, num_diffusion))
        }
    if phase == "train":  # you can also use the pretrained model provided in README
        _run_on_different_setting_train(list(_settings.keys()), list(_settings.values()), prefix=_prefix, n_devices=2)

    elif phase == "selection":  # select a model according to FID
        tuned = {"n_samples": 1000, "small_sigma": False, "sample_steps": None}
        _dirname = "selection_" + _dict2str(tuned)
        _sample_ckpts(prefix=_prefix, names=list(_settings.keys()), settings=list(_settings.values()),
                      ckpts=["%d.ckpt.pth" % k for k in range(110000, 500001, 10000)],
                      dirname=_dirname, common_setting={"batch_size": 500, **tuned})

    elif phase == "save_ms_eps":  # save \Gamma
        _save_ms_eps_settings = []
        for n_samples in [None]:
            _save_ms_eps_settings.append({"n_samples": n_samples})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _save_ms_eps_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                              best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                              settings=list(_settings.values()), save_ms_eps_settings=_save_ms_eps_settings, n_devices=_n_devices,
                              common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_analytic_ddpm":
        _sample_settings = []
        for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
            _sample_settings.append({"tag": "ms_eps", "steps_type": "linear", "sample_steps": sample_steps,
                                     "n_samples": 50000, "clip_sigma_idx": 1, "clip_pixel": 2, "seed": 1234})  # clip_pixel: 2 for LS and 1 for CS
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        names_ms_eps_path = {
            "beta_schedule_linear_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_linear_num_diffusion_1000/train/ms_eps/400000_n_samples_None.ms_eps.pth"
            },
            "beta_schedule_cosine_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_cosine_num_diffusion_1000/train/ms_eps/160000_n_samples_None.ms_eps.pth"
            }
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _sample_one_ckpt_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                                best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                                common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "nll_analytic_ddpm":
        _nll_settings = []
        for steps_type in ["dp_seg", "linear"]:
            for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
                _nll_settings.append({"tag": "ms_eps", "steps_type": steps_type, "sample_steps": sample_steps,
                                      "n_samples": None, "partition": "test"})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        names_ms_eps_path = {
            "beta_schedule_linear_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_linear_num_diffusion_1000/train/ms_eps/400000_n_samples_None.ms_eps.pth"
            },
            "beta_schedule_cosine_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_cosine_num_diffusion_1000/train/ms_eps/160000_n_samples_None.ms_eps.pth"
            }
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _nll_one_ckpt_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                             best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())], n_devices=_n_devices,
                             settings=list(_settings.values()), nll_settings=_nll_settings,
                             common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_analytic_ddim":
        _sample_settings = []
        for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
            _sample_settings.append({"tag": "ddim_ms_eps", "sample_steps": sample_steps, "n_samples": 50000,
                                     "eta": 0., "clip_sigma_idx": 1, "clip_pixel": 1, "seed": 1234})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        names_ms_eps_path = {
            "beta_schedule_linear_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_linear_num_diffusion_1000/train/ms_eps/400000_n_samples_None.ms_eps.pth"
            },
            "beta_schedule_cosine_num_diffusion_1000": {
                "ms_eps_path": "workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/"
                               "beta_schedule_cosine_num_diffusion_1000/train/ms_eps/160000_n_samples_None.ms_eps.pth"
            }
        }
        _settings = merge_dict(_settings, {key: names_ms_eps_path[key] for key in _settings.keys()})
        _sample_one_ckpt_ddim_ms_eps(prefix=_prefix, names=list(_settings.keys()),
                                     best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                     settings=list(_settings.values()), sample_settings=_sample_settings,
                                     n_devices=_n_devices,
                                     common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_ddpm":
        _sample_settings = []
        for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
            _sample_settings.append({"sample_steps": sample_steps, "n_samples": 50000, "small_sigma": True, "seed": 1234})
            _sample_settings.append({"sample_steps": sample_steps, "n_samples": 50000, "small_sigma": False, "clip_sigma_idx": 1, "clip_pixel": 1, "seed": 1234})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _sample_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                         best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                         settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                         common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_ddpm_quad":  # using the quadratic trajectory in DDIM for a stronger baseline
        _sample_settings = []
        for small_sigma in [True]:
            for sample_steps in [90]:
                _sample_settings.append({"sample_steps": sample_steps, "n_samples": 50000, "steps_type": "quad_ddim",
                                         "small_sigma": small_sigma, "seed": 1234})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _sample_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                         best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                         settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                         common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_ddim":
        _sample_settings = []
        for eta in [0., 1.]:
            for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
                _sample_settings.append({"tag": "ddim", "sample_steps": sample_steps, "n_samples": 50000,
                                         "eta": eta, "seed": 1234})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _sample_one_ckpt_ddim(prefix=_prefix, names=list(_settings.keys()),
                              best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                              settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                              common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "sample_ddim_quad":  # using the quadratic trajectory in DDIM for a stronger baseline
        _sample_settings = []
        for eta in [0.]:
            for sample_steps in [20, 25, 30]:
                _sample_settings.append({"tag": "ddim", "sample_steps": sample_steps, "steps_type": "quad_ddim",
                                         "n_samples": 50000, "eta": eta, "seed": 1234})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _sample_one_ckpt_ddim(prefix=_prefix, names=list(_settings.keys()),
                              best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                              settings=list(_settings.values()), sample_settings=_sample_settings, n_devices=_n_devices,
                              common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == "nll_ddpm":
        _nll_settings = []
        for small_sigma in [True, False]:
            for sample_steps in [10, 25, 50, 100, 200, 400, 1000]:
                _nll_settings.append({"sample_steps": sample_steps, "small_sigma": small_sigma,
                                      "n_samples": None, "partition": "test"})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _nll_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                      best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                      settings=list(_settings.values()), nll_settings=_nll_settings, n_devices=_n_devices,
                      common_setting={"batch_size": _n_devices * batch_size_per_card})

    elif phase == 'save_nll_terms':  # used in DP for big_sigma
        _save_nll_terms_settings = []
        for partition in ["train", "test"]:
            _save_nll_terms_settings.append({"small_sigma": False, "n_samples": None, "partition": partition})
        names_best_ckpts = {
            "beta_schedule_linear_num_diffusion_1000": "400000.ckpt.pth",
            "beta_schedule_cosine_num_diffusion_1000": "160000.ckpt.pth",
        }
        _save_nll_terms_one_ckpt(prefix=_prefix, names=list(_settings.keys()),
                                 best_ckpts=[names_best_ckpts[name] for name in list(_settings.keys())],
                                 settings=list(_settings.values()), save_nll_terms_settings=_save_nll_terms_settings, n_devices=_n_devices,
                                 common_setting={"batch_size": _n_devices * batch_size_per_card})
