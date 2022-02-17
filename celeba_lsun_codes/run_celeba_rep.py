import os
from multiprocessing import Process
from interface.runner import run_sample, run_sample_ms_eps, run_sample_ddim, run_sample_ddim_ms_eps
from interface.task_schedule import Task, wait_schedule, available_devices
batch_size_per_card = 125
n_devices = 8


# run experiments with a slightly different implementation of ET used in DDIM
def add_tasks(phase, tasks):
    steps_type = "linear_ddim"
    seed = 1234

    if phase == "sample_analytic_ddpm":
        clip_sigma_idx = 1
        clip_pixel = 2
        n_samples = 50000
        for sample_steps in [10, 20, 50, 100]:
            tag = "clip_sigma_idx_{}_clip_pixel_{}_n_samples_{}_steps_type_{}_sample_steps_{}" \
                .format(clip_sigma_idx, clip_pixel, n_samples, steps_type, sample_steps)
            pretrained_model = "ema_celeba"
            root = os.path.join("samples", pretrained_model)
            profile = {
                "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                "seed": seed,
                "pretrained_model": pretrained_model,
                "sample_steps": sample_steps,
                "batch_size": batch_size_per_card * n_devices,
                "path": os.path.join(root, tag),
                "n_samples": n_samples,
                "fid_stat": "workspace/fid_stats/fid_stats_celeba64_train_50000_ddim.npz",
                "ms_eps_path": "ms_eps/ema_celeba_10000.pth",
                "steps_type": steps_type,
                "clip_sigma_idx": clip_sigma_idx,
                "clip_pixel": clip_pixel,
            }
            p = Process(target=run_sample_ms_eps, args=(profile,))
            tasks.append(Task(p, n_devices=n_devices))

    elif phase == "sample_analytic_ddim":
        clip_sigma_idx = 1
        clip_pixel = 1
        for n_samples in [50000]:
            for eta in [0.]:
                for sample_steps in [10, 20, 50, 100]:
                    tag = "ddim_ms_eps_clip_sigma_idx_{}_clip_pixel_{}_n_samples_{}_eta_{}_steps_type_{}_sample_steps_{}" \
                        .format(clip_sigma_idx, clip_pixel, n_samples, eta, steps_type, sample_steps)
                    pretrained_model = "ema_celeba"
                    root = os.path.join("samples", pretrained_model)
                    profile = {
                        "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                        "seed": seed,
                        "pretrained_model": pretrained_model,
                        "eta": eta,
                        "sample_steps": sample_steps,
                        "steps_type": steps_type,
                        "batch_size": batch_size_per_card * n_devices,
                        "path": os.path.join(root, tag),
                        "n_samples": n_samples,
                        "fid_stat": "workspace/fid_stats/fid_stats_celeba64_train_50000_ddim.npz",
                        "ms_eps_path": "ms_eps/ema_celeba_10000.pth",
                        "clip_sigma_idx": clip_sigma_idx,
                        "clip_pixel": clip_pixel,
                    }
                    p = Process(target=run_sample_ddim_ms_eps, args=(profile,))
                    tasks.append(Task(p, n_devices=n_devices))

    elif phase == "sample_ddpm":
        for n_samples in [50000]:
            for small_sigma in [True, False]:
                for sample_steps in [10, 20, 50, 100]:
                    tag = "n_samples_{}_small_sigma_{}_steps_type_{}_sample_steps_{}".format(n_samples, small_sigma, steps_type, sample_steps)
                    pretrained_model = "ema_celeba"
                    root = os.path.join("samples", pretrained_model)
                    profile = {
                        "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                        "seed": seed,
                        "pretrained_model": pretrained_model,
                        "small_sigma": small_sigma,
                        "sample_steps": sample_steps,
                        "batch_size": batch_size_per_card * n_devices,
                        "path": os.path.join(root, tag),
                        "n_samples": n_samples,
                        "steps_type": steps_type,
                        "fid_stat": "workspace/fid_stats/fid_stats_celeba64_train_50000_ddim.npz",
                    }
                    p = Process(target=run_sample, args=(profile,))
                    tasks.append(Task(p, n_devices=n_devices))

    elif phase == "sample_ddim":
        for n_samples in [50000]:
            for eta in [0.]:
                for sample_steps in [10, 20, 50, 100]:
                    tag = "ddim_n_samples_{}_eta_{}_steps_type_{}_sample_steps_{}".format(n_samples, eta, steps_type, sample_steps)
                    pretrained_model = "ema_celeba"
                    root = os.path.join("samples", pretrained_model)
                    profile = {
                        "fname_log": os.path.join(root, os.path.join("%s.log" % tag)),
                        "seed": seed,
                        "pretrained_model": pretrained_model,
                        "eta": eta,
                        "sample_steps": sample_steps,
                        "steps_type": steps_type,
                        "batch_size": batch_size_per_card * n_devices,
                        "path": os.path.join(root, tag),
                        "n_samples": n_samples,
                        "fid_stat": "workspace/fid_stats/fid_stats_celeba64_train_50000_ddim.npz",
                    }
                    p = Process(target=run_sample_ddim, args=(profile,))
                    tasks.append(Task(p, n_devices=n_devices))


def main():
    tasks = []
    add_tasks("sample_analytic_ddim", tasks)
    add_tasks("sample_analytic_ddpm", tasks)
    add_tasks("sample_ddim", tasks)
    add_tasks("sample_ddpm", tasks)
    wait_schedule(tasks, devices=[] or available_devices())


if __name__ == "__main__":
    main()
