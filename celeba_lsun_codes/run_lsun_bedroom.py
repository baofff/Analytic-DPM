import os
from multiprocessing import Process
from interface.runner import run_sample, run_nll, run_save_ms_eps, run_sample_ms_eps, run_nll_ms_eps, run_sample_ddim, run_sample_ddim_ms_eps
from interface.task_schedule import Task, wait_schedule, available_devices
batch_size_per_card = 10
n_devices = 8


if __name__ == "__main__":

    phase = "sample_analytic_ddpm"

    if phase == "save_ms_eps":
        n_samples = 1000
        pretrained_model = "ema_lsun_bedroom"
        profile = {
            "fname_log": os.path.join("ms_eps", "%s_%d.log" % (pretrained_model, n_samples)),
            "pretrained_model": pretrained_model,
            "train_dataset": "lsun_bedroom",
            "batch_size": batch_size_per_card * n_devices,
            "fname": os.path.join("ms_eps", "%s_%d.pth" % (pretrained_model, n_samples)),
            "n_samples": n_samples
        }
        p = Process(target=run_save_ms_eps, args=(profile,))
        tasks = [Task(p, n_devices=n_devices)]

        wait_schedule(tasks, devices=[] or available_devices())

    elif phase == "sample_analytic_ddpm":
        tasks = []
        clip_sigma_idx = 1
        clip_pixel = 1
        n_samples = 50000
        steps_type = "linear"
        for sample_steps in [10, 25, 50, 100, 200]:
            tag = "clip_sigma_idx_{}_clip_pixel_{}_n_samples_{}_steps_type_{}_sample_steps_{}"\
                .format(clip_sigma_idx, clip_pixel, n_samples, steps_type, sample_steps)
            pretrained_model = "ema_lsun_bedroom"
            root = os.path.join("samples", pretrained_model)
            profile = {
                "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                "seed": 1234,
                "pretrained_model": pretrained_model,
                "sample_steps": sample_steps,
                "batch_size": batch_size_per_card * n_devices,
                "path": os.path.join(root, tag),
                "n_samples": n_samples,
                "fid_stat": "workspace/fid_stats/fid_stats_lsun_bedroom_train_50000_ddim.npz",
                "ms_eps_path": "ms_eps/ema_lsun_bedroom_1000.pth",
                "steps_type": steps_type,
                "clip_sigma_idx": clip_sigma_idx,
                "clip_pixel": clip_pixel,
            }
            p = Process(target=run_sample_ms_eps, args=(profile,))
            tasks.append(Task(p, n_devices=n_devices))
        wait_schedule(tasks, devices=[] or available_devices())

    elif phase == "sample_ddpm":
        tasks = []
        for n_samples in [50000]:
            for small_sigma in [True]:
                for sample_steps in [10, 25, 50, 100, 200]:
                    tag = "n_samples_{}_small_sigma_{}_sample_steps_{}".format(n_samples, small_sigma, sample_steps)
                    pretrained_model = "ema_lsun_bedroom"
                    root = os.path.join("samples", pretrained_model)
                    profile = {
                        "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                        "seed": 1234,
                        "pretrained_model": pretrained_model,
                        "small_sigma": small_sigma,
                        "sample_steps": sample_steps,
                        "batch_size": batch_size_per_card * n_devices,
                        "path": os.path.join(root, tag),
                        "n_samples": n_samples,
                        "fid_stat": "workspace/fid_stats/fid_stats_lsun_bedroom_train_50000_ddim.npz",
                    }
                    p = Process(target=run_sample, args=(profile,))
                    tasks.append(Task(p, n_devices=n_devices))
        wait_schedule(tasks, devices=available_devices())

    elif phase == "sample_ddim":
        tasks = []
        for n_samples in [50000]:
            for eta in [0.]:
                for sample_steps in [10, 25, 50, 100, 200]:
                    tag = "ddim_n_samples_{}_eta_{}_sample_steps_{}".format(n_samples, eta, sample_steps)
                    pretrained_model = "ema_lsun_bedroom"
                    root = os.path.join("samples", pretrained_model)
                    profile = {
                        "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                        "seed": None,
                        "pretrained_model": pretrained_model,
                        "eta": eta,
                        "sample_steps": sample_steps,
                        "batch_size": batch_size_per_card * n_devices,
                        "path": os.path.join(root, tag),
                        "n_samples": n_samples,
                        "fid_stat": "workspace/fid_stats/fid_stats_lsun_bedroom_train_50000_ddim.npz",
                    }
                    p = Process(target=run_sample_ddim, args=(profile,))
                    tasks.append(Task(p, n_devices=n_devices))

        wait_schedule(tasks, devices=[] or available_devices())
