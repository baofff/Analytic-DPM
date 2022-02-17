
__all__ = ["run_on_different_settings", "sample_ckpts", "sample_one_ckpt", "evaluate_one_ckpt", "nll_ckpts", "nll_one_ckpt", "save_ms_eps_one_ckpt"]


from .misc import get_root_by_time
from .dict_utils import merge_dict
import os
from multiprocessing import Process
from .task_schedule import Task, wait_schedule, available_devices
from typing import List, Union
import copy


def run_on_different_settings(run_fn, profile: dict, path: str, prefix: str, names: List[str], settings: List[dict],
                              n_devices: Union[int, List[int]] = 1, common_setting: dict = None,
                              devices=None, time: str = None, time_strategy: str = None):
    r"""
    Args:
        run_fn: the running function
        profile: the profile template
        path: the result of each experiment will be saved in "path/prefix_time/name"
        prefix: the result of each experiment will be saved in "path/prefix_time/name"
        names: the result of each experiment will be saved in "path/prefix_time/name"
        settings: the settings of experiments
        n_devices: the number of devices for each experiment
        common_setting: the common experimental setting
        devices: the devices to use
        time: a time tag with the format %Y-%m-%d-%H-%M-%S
        time_strategy: how to infer the time when time is None (only works when time is None )
    """
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(settings) == len(n_devices)
    path_prefix_time = get_root_by_time(path, prefix, time, time_strategy)

    tasks = []
    for setting, name, n_device in zip(settings, names, n_devices):
        _profile = profile
        if common_setting is not None:
            _profile = merge_dict(_profile, common_setting)
        _profile = merge_dict(_profile, {
            "workspace_root": os.path.join(path_prefix_time, name),
            **setting
        })
        p = Process(target=run_fn, args=(_profile,))
        tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def sample_ckpts(run_fn, profile: dict, path: str, prefix: str, names: List[str], settings: List[dict],
                 ckpts: List[str], dirname: str, ref_ckpt_paths: List[List[str]] = None,
                 n_devices: Union[int, List[int]] = 1, common_setting: dict = None, devices=None, time: str = None):
    r""" generally for validation
    """
    if ref_ckpt_paths is None:
        ref_ckpt_paths = [[] for _ in names]
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(settings) == len(ref_ckpt_paths) == len(n_devices)
    path_prefix_time = get_root_by_time(path, prefix, time, "latest")

    tasks = []
    for setting, name, ref_ckpt_path, n_device in zip(settings, names, ref_ckpt_paths, n_devices):
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        for ckpt in ckpts:
            _profile = profile
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/{}.log".format(ckpt, dirname)),
            }
            _profile = merge_dict(_profile, {
                "path": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/{}".format(ckpt, dirname)),
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/reproducibility/{}".format(ckpt, dirname)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt), *ref_ckpt_path],
                **setting,
            })
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def nll_ckpts(run_fn, profile: dict, path: str, prefix: str, names: List[str], settings: List[dict],
              ckpts: List[str], filename: str, ref_ckpt_paths: List[List[str]] = None,
              n_devices: Union[int, List[int]] = 1, common_setting: dict = None, devices=None, time: str = None):
    r""" generally for validation
    """
    if ref_ckpt_paths is None:
        ref_ckpt_paths = [[] for _ in names]
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(settings) == len(ref_ckpt_paths) == len(n_devices)
    path_prefix_time = get_root_by_time(path, prefix, time, "latest")

    tasks = []
    for setting, name, ref_ckpt_path, n_device in zip(settings, names, ref_ckpt_paths, n_devices):
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        for ckpt in ckpts:
            _profile = profile
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/{}.log".format(ckpt, filename)),
            }
            _profile = merge_dict(_profile, {
                "fname": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/{}.pth".format(ckpt, filename)),
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/reproducibility/{}".format(ckpt, filename)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt), *ref_ckpt_path],
                **setting,
            })
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def _dict2str(dct):
    pairs = []
    for key, val in dct.items():
        pairs.append("{}_{}".format(key, val))
    return "_".join(pairs)


def sample_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str], ckpts: List[str], settings: List[dict],
                    sample_settings: List[dict], ref_ckpt_paths: List[List[str]] = None, n_devices: Union[int, List[int]] = 1,
                    common_setting: dict = None, devices=None, time: str = None):
    if ref_ckpt_paths is None:
        ref_ckpt_paths = [[] for _ in names]
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(ckpts) == len(settings) == len(ref_ckpt_paths) == len(n_devices)

    tasks = []
    for setting, name, ckpt, ref_ckpt_path, n_device in zip(settings, names, ckpts, ref_ckpt_paths, n_devices):
        for sample_setting in sample_settings:
            path_prefix_time = get_root_by_time(path, prefix, time, "latest")
            workspace_root = os.path.join(path_prefix_time, name)
            ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
            dirname = _dict2str(sample_setting)

            _profile = copy.deepcopy(profile)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/{}.log".format(ckpt, dirname)),
            }
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile = merge_dict(_profile, {
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/reproducibility/{}".format(ckpt, dirname)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt), *ref_ckpt_path],
                "path": os.path.join(workspace_root, "evaluate/evaluator/sample2dir/{}/{}".format(ckpt, dirname)),
            })
            _profile = merge_dict(_profile, setting)
            _profile = merge_dict(_profile, sample_setting)
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def nll_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str], ckpts: List[str], settings: List[dict],
                 nll_settings: List[dict], ref_ckpt_paths: List[List[str]] = None, n_devices: Union[int, List[int]] = 1,
                 common_setting: dict = None, devices=None, time: str = None):
    if ref_ckpt_paths is None:
        ref_ckpt_paths = [[] for _ in names]
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(ckpts) == len(settings) == len(ref_ckpt_paths) == len(n_devices)

    tasks = []
    for setting, name, ckpt, ref_ckpt_path, n_device in zip(settings, names, ckpts, ref_ckpt_paths, n_devices):
        for nll_setting in nll_settings:
            path_prefix_time = get_root_by_time(path, prefix, time, "latest")
            workspace_root = os.path.join(path_prefix_time, name)
            ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
            filename = _dict2str(nll_setting)

            _profile = copy.deepcopy(profile)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/{}.log".format(ckpt, filename)),
            }
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile = merge_dict(_profile, {
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/reproducibility/{}".format(ckpt, filename)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt), *ref_ckpt_path],
                "fname": os.path.join(workspace_root, "evaluate/evaluator/nll/{}/{}.pth".format(ckpt, filename)),
            })
            _profile = merge_dict(_profile, setting)
            _profile = merge_dict(_profile, nll_setting)
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def save_ms_eps_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str], ckpts: List[str], settings: List[dict],
                         save_ms_eps_settings: List[dict], n_devices: Union[int, List[int]] = 1,
                         common_setting: dict = None, devices=None, time: str = None):
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(ckpts) == len(settings) == len(n_devices)

    tasks = []
    for setting, name, ckpt, n_device in zip(settings, names, ckpts, n_devices):
        for save_ms_eps_setting in save_ms_eps_settings:
            path_prefix_time = get_root_by_time(path, prefix, time, "latest")
            workspace_root = os.path.join(path_prefix_time, name)
            ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
            save_ms_eps_setting_s = _dict2str(save_ms_eps_setting)
            fname = "{}_{}.ms_eps.pth".format(ckpt.split('.')[0], save_ms_eps_setting_s)

            _profile = copy.deepcopy(profile)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "train/ms_eps/{}.log".format(fname)),
            }
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile = merge_dict(_profile, {
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "train/ms_eps/reproducibility/{}".format(fname)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt)],
                "fname": os.path.join(workspace_root, "train/ms_eps/{}".format(fname)),
            })
            _profile = merge_dict(_profile, setting)
            _profile = merge_dict(_profile, save_ms_eps_setting)
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def save_nll_terms_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str], ckpts: List[str], settings: List[dict],
                            save_nll_terms_settings: List[dict], n_devices: Union[int, List[int]] = 1,
                            common_setting: dict = None, devices=None, time: str = None):
    if isinstance(n_devices, int):
        n_devices = [n_devices] * len(settings)
    assert len(names) == len(ckpts) == len(settings) == len(n_devices)

    tasks = []
    for setting, name, ckpt, n_device in zip(settings, names, ckpts, n_devices):
        for save_nll_terms_setting in save_nll_terms_settings:
            path_prefix_time = get_root_by_time(path, prefix, time, "latest")
            workspace_root = os.path.join(path_prefix_time, name)
            ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
            save_nll_terms_setting_s = _dict2str(save_nll_terms_setting)
            fname = "{}_{}.nll_terms.pth".format(ckpt.split('.')[0], save_nll_terms_setting_s)

            _profile = copy.deepcopy(profile)
            _profile["interact"] = {
                "fname_log": os.path.join(workspace_root, "train/nll_terms/{}.log".format(fname)),
            }
            if common_setting is not None:
                _profile = merge_dict(_profile, common_setting)
            _profile = merge_dict(_profile, {
                "workspace_root": workspace_root,
                "backup_root": os.path.join(workspace_root, "train/nll_terms/reproducibility/{}".format(fname)),
                "ckpt_path": [os.path.join(ckpt_root, ckpt)],
                "fname": os.path.join(workspace_root, "train/nll_terms/{}".format(fname)),
            })
            _profile = merge_dict(_profile, setting)
            _profile = merge_dict(_profile, save_nll_terms_setting)
            p = Process(target=run_fn, args=(_profile,))
            tasks.append(Task(p, n_device))

    wait_schedule(tasks, devices=devices or available_devices())


def evaluate_one_ckpt(run_fn, profile: dict, path: str, prefix: str, names: List[str], ckpts: List[str], settings: List[dict],
                      ref_ckpt_paths: List[List[str]] = None, n_devices: Union[int, List[int]] = 1, common_setting: dict = None,
                      devices=None, time: str = None):
    if ref_ckpt_paths is None:
        ref_ckpt_paths = [[] for _ in names]
    assert len(names) == len(ckpts) == len(settings) == len(ref_ckpt_paths)
    _settings = []
    for setting, name, ckpt, ref_ckpt_path in zip(settings, names, ckpts, ref_ckpt_paths):
        path_prefix_time = get_root_by_time(path, prefix, time, "latest")
        workspace_root = os.path.join(path_prefix_time, name)
        ckpt_root = os.path.join(workspace_root, 'train/ckpts/')
        _settings.append(merge_dict(setting, {
            "backup_root": os.path.join(workspace_root, "evaluate/reproducibility"),
            "ckpt_path": [os.path.join(ckpt_root, ckpt), *ref_ckpt_path],
        }))

    run_on_different_settings(run_fn=run_fn, profile=profile, path=path, prefix=prefix, names=names, settings=_settings,
                              n_devices=n_devices, common_setting=common_setting,
                              devices=devices, time=time, time_strategy="latest")
