from pytorch_diffusion import Diffusion
import numpy as np
from core.inference.sampler.reverse_ddpm import reverse_ddpm_naive, reverse_ddpm_ms_eps
from core.inference.sampler.reverse_ddim import reverse_ddim_naive, reverse_ddim_ms_eps
from core.inference.ll.elbo import nelbo_naive_ddpm, nelbo_ms_eps_ddpm
from core.inference.utils import _rescale_timesteps
from interface.utils import sample2dir, set_logger, set_seed, global_device, score_on_dataset
from interface.datasets import get_train_dataset, get_test_dataset
import torch
import os
import logging
import torch.nn as nn
import math
import core.func as func
from torch.utils.data import Subset
import random


def run_sample(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])
    set_seed(profile["seed"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    kwargs = sub_dict(profile, "sample_steps", "small_sigma", "clip_sigma_idx", "clip_pixel", "steps_type")

    def sample_fn(n_samples):
        x_init = torch.randn(n_samples, *data_shape, device=global_device())
        return reverse_ddpm_naive(x_init, betas, clip_denoise=True, rescale_timesteps=True, eps_model=eps_model, **kwargs)

    os.makedirs(os.path.split(profile["path"])[0], exist_ok=True)
    sample2dir(path=profile["path"], n_samples=profile["n_samples"], batch_size=profile["batch_size"], sample_fn=sample_fn,
               unpreprocess_fn=diffusion.torch2hwcuint8)

    if profile["fid_stat"]:
        from tools.fid_score import calculate_fid_given_paths
        fid = calculate_fid_given_paths((profile["fid_stat"], profile["path"]))
        logging.info("fid={}".format(fid))


def run_sample_ddim(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])
    set_seed(profile["seed"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    kwargs = sub_dict(profile, "sample_steps", "eta", "steps_type")

    def sample_fn(_n_samples):
        x_init = torch.randn(_n_samples, *data_shape, device=global_device())
        return reverse_ddim_naive(x_init, betas, rescale_timesteps=True, eps_model=eps_model, **kwargs)

    os.makedirs(os.path.split(profile["path"])[0], exist_ok=True)
    sample2dir(path=profile["path"], n_samples=profile["n_samples"], batch_size=profile["batch_size"],
               sample_fn=sample_fn,
               unpreprocess_fn=diffusion.torch2hwcuint8)

    if profile["fid_stat"]:
        from tools.fid_score import calculate_fid_given_paths
        fid = calculate_fid_given_paths((profile["fid_stat"], profile["path"]))
        logging.info("fid={}".format(fid))


def run_sample_ddim_ms_eps(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])
    set_seed(profile["seed"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)
    ms_eps = torch.load(profile["ms_eps_path"])

    kwargs = sub_dict(profile, "sample_steps", "eta", "clip_sigma_idx", "clip_pixel", "steps_type")

    def sample_fn(_n_samples):
        x_init = torch.randn(_n_samples, *data_shape, device=global_device())
        return reverse_ddim_ms_eps(x_init, betas, rescale_timesteps=True, eps_model=eps_model,
                                   ms_eps=ms_eps, **kwargs)

    os.makedirs(os.path.split(profile["path"])[0], exist_ok=True)
    sample2dir(path=profile["path"], n_samples=profile["n_samples"], batch_size=profile["batch_size"],
               sample_fn=sample_fn,
               unpreprocess_fn=diffusion.torch2hwcuint8)

    if profile["fid_stat"]:
        from tools.fid_score import calculate_fid_given_paths
        fid = calculate_fid_given_paths((profile["fid_stat"], profile["path"]))
        logging.info("fid={}".format(fid))


def run_nll(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    data_dim = int(np.prod(data_shape))
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    def score_fn(x_0):
        nelbo, terms = nelbo_naive_ddpm(x_0, betas, profile["small_sigma"], clip_denoise=True, rescale_timesteps=True,
                                        eps_model=eps_model, sample_steps=profile["sample_steps"])
        return tuple([nelbo, *terms])

    dataset = get_test_dataset(profile["test_dataset"])
    n_samples = profile.get("n_samples", len(dataset)) or len(dataset)
    idxes = random.sample(list(range(len(dataset))), n_samples)
    dataset = Subset(dataset, idxes)

    logging.info("nll with {} test samples".format(n_samples))

    outputs = score_on_dataset(dataset, score_fn, profile["batch_size"])
    outputs_bpd = [a / (data_dim * math.log(2.)) for a in outputs]
    nelbo_bpd = outputs_bpd[0]
    terms_bpd = outputs_bpd[1:]
    logging.info('bpd={}'.format(nelbo_bpd))
    logging.info('continuous_part={}'.format(sum(terms_bpd[1:])))
    logging.info('discrete_part={}'.format(terms_bpd[0]))
    torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, profile["fname"])


def run_save_ms_eps(profile):
    set_logger(profile["fname_log"])
    dataset = get_train_dataset(profile["train_dataset"])
    n_samples = profile.get("n_samples", len(dataset)) or len(dataset)
    idxes = random.sample(list(range(len(dataset))), n_samples)
    dataset = Subset(dataset, idxes)

    logging.info("save_ms_eps with {} samples".format(len(dataset)))

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    N = len(betas) - 1
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    ms_eps = np.zeros(N + 1, dtype=np.float32)
    for n in range(1, N + 1):
        @torch.no_grad()
        def score_fn(x_0):
            eps = torch.randn_like(x_0)
            x_n = cum_alphas[n] ** 0.5 * x_0 + cum_betas[n] ** 0.5 * eps
            eps_pred = eps_model(x_n, _rescale_timesteps(torch.tensor([n] * x_n.size(0)).type_as(x_n), N, True))
            return func.mos(eps_pred)

        ms_eps[n] = score_on_dataset(dataset, score_fn, profile["batch_size"])
        logging.info("[n: {}] [ms_eps[{}]: {}]".format(n, n, ms_eps[n]))

    torch.save(ms_eps, profile["fname"])
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(list(range(1, N + 1)), ms_eps[1:])
    plt.savefig("{}.png".format(profile["fname"]))
    plt.close()


def sub_dict(dct, *keys):
    return {key: dct[key] for key in keys if key in dct}


def run_sample_ms_eps(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])
    set_seed(profile["seed"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)
    ms_eps = torch.load(profile["ms_eps_path"])

    kwargs = sub_dict(profile, "steps_type", "sample_steps", "clip_sigma_idx", "clip_pixel")

    def sample_fn(n_samples):
        x_init = torch.randn(n_samples, *data_shape, device=global_device())
        return reverse_ddpm_ms_eps(x_init, betas, rescale_timesteps=True, eps_model=eps_model, ms_eps=ms_eps, **kwargs)

    os.makedirs(os.path.split(profile["path"])[0], exist_ok=True)
    sample2dir(path=profile["path"], n_samples=profile["n_samples"], batch_size=profile["batch_size"], sample_fn=sample_fn,
               unpreprocess_fn=diffusion.torch2hwcuint8)

    from tools.fid_score import calculate_fid_given_paths
    fid = calculate_fid_given_paths((profile["fid_stat"], profile["path"]))
    logging.info("fid={}".format(fid))


def run_nll_ms_eps(profile: dict):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    data_dim = int(np.prod(data_shape))
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    eps_model = lambda x, t: diffusion.model(x, t - 1)
    ms_eps = torch.load(profile["ms_eps_path"])

    def score_fn(x_0):
        nelbo, terms = nelbo_ms_eps_ddpm(x_0, betas, rescale_timesteps=True, steps_type=profile['steps_type'],
                                         eps_model=eps_model, ms_eps=ms_eps, sample_steps=profile["sample_steps"])
        return tuple([nelbo, *terms])

    dataset = get_test_dataset(profile["test_dataset"])
    n_samples = profile.get("n_samples", len(dataset)) or len(dataset)
    idxes = random.sample(list(range(len(dataset))), n_samples)
    dataset = Subset(dataset, idxes)

    logging.info("nll_ms_eps with {} test samples".format(n_samples))
    logging.info("load ms_eps from {}".format(profile["ms_eps_path"]))

    outputs = score_on_dataset(dataset, score_fn, profile["batch_size"])
    outputs_bpd = [a / (data_dim * math.log(2.)) for a in outputs]
    nelbo_bpd = outputs_bpd[0]
    terms_bpd = outputs_bpd[1:]
    logging.info('bpd={}'.format(nelbo_bpd))
    logging.info('continuous_part={}'.format(sum(terms_bpd[1:])))
    logging.info('discrete_part={}'.format(terms_bpd[0]))
    torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, profile["fname"])


def _sigma2_small(s, r, alphas, cum_alphas):
    skip_alpha = alphas[s + 1: r + 1].prod()
    skip_beta = 1. - skip_alpha
    cum_alpha_s, cum_alpha_r = cum_alphas[s], cum_alphas[r]
    sigma2_small = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r)
    return sigma2_small


def _x_0_pred(x, n, cum_alphas, rescale_timesteps, eps_model=None):  # estimate of E[x_0|x_n] w.r.t. q
    N = len(cum_alphas) - 1
    cum_alpha_n = cum_alphas[n]
    input_n = n
    eps_pred = eps_model(x, _rescale_timesteps(torch.tensor([input_n] * x.size(0)).type_as(x), N, rescale_timesteps))
    x_0_pred = cum_alpha_n ** -0.5 * x - (1. / cum_alpha_n - 1.) ** 0.5 * eps_pred
    return x_0_pred, eps_pred


@ torch.no_grad()
def get_nelbo_terms(profile):
    os.makedirs(os.path.split(profile["fname_log"])[0], exist_ok=True)
    set_logger(profile["fname_log"])

    diffusion = Diffusion.from_pretrained(profile["pretrained_model"])
    data_shape = diffusion.model.in_channels, diffusion.model.resolution, diffusion.model.resolution
    data_dim = int(np.prod(data_shape))
    diffusion.model = nn.DataParallel(diffusion.model)
    betas = np.append(0., diffusion.betas)
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    eps_model = lambda x, t: diffusion.model(x, t - 1)

    if profile["partition"] == "train":
        dataset = get_train_dataset(profile["dataset"])
    elif profile["partition"] == "test":
        dataset = get_test_dataset(profile["dataset"])
    else:
        raise ValueError
    n_samples = profile["n_samples"] or len(dataset)
    idxes = random.sample(list(range(len(dataset))), n_samples)
    dataset = Subset(dataset, idxes)

    N = len(betas) - 1
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("get_nelbo_terms with eps_model, small_sigma={}".format(profile["small_sigma"]))

    F = np.full((N + 1, N + 1), float('inf'))  # F[s, r] with 0 <= s < r <= N

    reconstructions = np.zeros([N + 1])
    for r in range(1, N + 1):
        def reconstruction_nll_fn(x_0):
            eps = torch.randn_like(x_0)
            x_r = cum_alphas[r] ** 0.5 * x_0 + cum_betas[r] ** 0.5 * eps
            x_0_pred, eps_pred = _x_0_pred(x_r, r, cum_alphas, True, eps_model=eps_model)
            x_0_pred_clamp = x_0_pred.clamp(-1., 1.)
            _reconstruction = func.sos(x_0 - x_0_pred_clamp, start_dim=1)

            mu_p = x_0_pred_clamp
            var_p = _sigma2_small(1, 2, alphas, cum_alphas) if r == 1 else cum_betas[r]
            _nll = -func.log_discretized_normal(x_0, mu_p, var_p).flatten(1).sum(1)  # - E_q log p(x_0|x_r)

            return _reconstruction, _nll
        reconstructions[r], F[0, r] = score_on_dataset(dataset, reconstruction_nll_fn, profile["batch_size"])
        logging.info("reconstructions[{}]={}, nll[{}]={}".format(r, reconstructions[r], r, F[0, r]))

    for s in range(1, N + 1):  # F is kl when 1 <= s < r <= N
        for r in range(s + 1, N + 1):
            skip_alpha = alphas[s + 1: r + 1].prod()
            skip_beta = 1. - skip_alpha
            tilde_beta = skip_beta * cum_betas[s] / cum_betas[r]
            sigma2 = tilde_beta if profile["small_sigma"] else skip_beta
            c = 0.5 * data_dim * (tilde_beta / sigma2 + np.log(sigma2 / tilde_beta) - 1.)
            coeff = 0.5 * cum_alphas[s] * skip_beta ** 2 / (cum_betas[r] ** 2 * sigma2)
            F[s, r] = c + coeff * reconstructions[r]

    def last_term_fn(x_0):
        mu_q = cum_alphas[N] ** 0.5 * x_0
        var_q = cum_betas[N]
        mu_p = torch.zeros_like(mu_q)
        var_p = 1.
        term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
        return term
    last_term = score_on_dataset(dataset, last_term_fn, profile["batch_size"])

    res = {"F": F, "last_term": last_term}

    torch.save(res, profile["fname"])

    N = len(betas) - 1
    terms = [*[res['F'][n, n + 1] for n in range(0, N)], res['last_term']]
    terms_bpd = [a / (data_dim * math.log(2.)) for a in terms]
    nelbo_bpd = sum(terms_bpd)
    logging.info("bpd/continuous_part/discrete_part: {}/{}/{}".format(nelbo_bpd, sum(terms_bpd[1:]), terms_bpd[0]))
