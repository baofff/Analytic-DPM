from interface.utils.misc import get_root_by_time
import os
from .fid_score import calculate_fid_given_paths


def fid_ckpts(stat, path, prefix, names, ckpts, dirname, device=None, batch_size=200, time=None):
    path_prefix_time = get_root_by_time(path, prefix, time, "latest")
    for name in names:
        root = os.path.join(path_prefix_time, name, 'evaluate/evaluator/sample2dir')
        with open(os.path.join(root, "%s_fid.txt" % dirname), 'w') as f:
            for ckpt in ckpts:
                samples_dir = os.path.join(root, ckpt, dirname)
                fid = calculate_fid_given_paths((stat, samples_dir), device=device, batch_size=batch_size)
                f.write("{}: {}\n".format(ckpt, fid))


def fid_one_ckpt(stat, path, prefix, names, ckpts, dirname, device=None, batch_size=200, time=None):
    path_prefix_time = get_root_by_time(path, prefix, time, "latest")
    for name, ckpt in zip(names, ckpts):
        root = os.path.join(path_prefix_time, name, 'evaluate/evaluator/sample2dir')
        with open(os.path.join(root, "%s_fid.txt" % dirname), 'w') as f:
            samples_dir = os.path.join(root, ckpt, dirname)
            fid = calculate_fid_given_paths((stat, samples_dir), device=device, batch_size=batch_size)
            f.write("{}: {}\n".format(ckpt, fid))


def fid_ckpts_cifar10(path, prefix, names, ckpts, dirname, device=None, batch_size=200, time=None):
    fid_ckpts('workspace/fid_stats/fid_stats_cifar10_train_pytorch.npz', path, prefix, names, ckpts, dirname, device, batch_size, time)


def fid_one_ckpt_cifar10(path, prefix, names, ckpts, dirname, device=None, batch_size=200, time=None):
    fid_one_ckpt('workspace/fid_stats/fid_stats_cifar10_train_pytorch.npz', path, prefix, names, ckpts, dirname, device, batch_size, time)
