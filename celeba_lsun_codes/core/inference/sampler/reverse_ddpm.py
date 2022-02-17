import torch
import numpy as np
import logging
from core.inference.utils import _choice_steps, _x_0_pred, _report_statistics
import math


@ torch.no_grad()
def reverse_ddpm_naive(x_init, betas, small_sigma, clip_denoise, rescale_timesteps, steps_type='linear', clip_sigma_idx=0, clip_pixel=2,
                       eps_model=None, d_model=None, sample_steps=None):
    assert (eps_model is None and d_model is not None) or (eps_model is not None and d_model is None)
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, steps_type)
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("reverse_ddpm_naive with {}, small_sigma={}, clip_denoise={}, rescale_timesteps={}, "
                 "sample_steps={}, steps_type={}, clip_sigma_idx={}, clip_pixel={}"
                 .format("eps_model" if eps_model is not None else "d_model",
                         small_sigma, clip_denoise, rescale_timesteps, sample_steps, steps_type, clip_sigma_idx, clip_pixel))

    x = x_init
    for s, r in list(zip([0] + ns, ns))[::-1]:
        statistics = {}
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_s, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[s], cum_betas[r]
        statistics['skip_beta'] = skip_beta
        statistics['cum_beta_s'] = cum_beta_s
        statistics['cum_beta_r'] = cum_beta_r
        statistics['cum_alpha_s'] = cum_alpha_s

        x_0_pred, eps_pred = _x_0_pred(x, r, cum_alphas, rescale_timesteps, eps_model, d_model)
        if clip_denoise:
            x_0_pred = x_0_pred.clamp(-1., 1.)
        coeff1 = skip_beta * cum_alpha_s ** 0.5 / cum_beta_r
        coeff2 = skip_alpha ** 0.5 * cum_beta_s / cum_beta_r
        x_mean = coeff1 * x_0_pred + coeff2 * x
        if s != 0:
            sigma2 = skip_beta
            if small_sigma:
                sigma2 *= cum_beta_s / cum_beta_r
            if s < ns[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
                statistics['sigma2_unclip'] = sigma2.item()
                sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                sigma2 = np.clip(sigma2, 0., sigma2_threshold)
                statistics['sigma2_threshold'] = sigma2_threshold
            x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            statistics['sigma2'] = sigma2
        else:
            x = x_mean
        _report_statistics(s, r, statistics)
    return x


@ torch.no_grad()
def reverse_ddpm_ms_eps(x_init, betas, rescale_timesteps, steps_type='linear', clip_sigma_idx=0, clip_pixel=2, eps_model=None, ms_eps=None, sample_steps=None):
    assert eps_model is not None and ms_eps is not None
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, steps_type, ms_eps=ms_eps, betas=betas)
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("reverse_ddpm_ms_eps with eps_model, rescale_timesteps={}, sample_steps={}, steps_type={}, clip_sigma_idx={}, clip_pixel={}"
                 .format(rescale_timesteps, sample_steps, steps_type, clip_sigma_idx, clip_pixel))

    x = x_init
    for s, r in list(zip([0] + ns, ns))[::-1]:
        statistics = {}
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_s, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[s], cum_betas[r]
        statistics['skip_beta'] = skip_beta
        statistics['cum_beta_s'] = cum_beta_s
        statistics['cum_beta_r'] = cum_beta_r
        statistics['cum_alpha_s'] = cum_alpha_s

        x_0_pred, eps_pred = _x_0_pred(x, r, cum_alphas, rescale_timesteps, eps_model)
        x_0_pred_clamp = x_0_pred.clamp(-1., 1.)
        coeff1 = skip_beta * cum_alpha_s ** 0.5 / cum_beta_r
        coeff2 = skip_alpha ** 0.5 * cum_beta_s / cum_beta_r
        x_mean = coeff1 * x_0_pred_clamp + coeff2 * x
        if s != 0:
            sigma2_small = skip_beta * cum_beta_s / cum_beta_r
            cov_x_0_pred = cum_beta_r / cum_alpha_r * (1. - ms_eps[r])
            cov_x_0_pred_clamp = np.clip(cov_x_0_pred, 0., 1.)
            coeff_cov_x_0 = cum_alpha_s * skip_beta ** 2 / cum_beta_r ** 2
            offset = coeff_cov_x_0 * cov_x_0_pred_clamp
            sigma2 = sigma2_small + offset
            statistics['sigma2_small'] = sigma2_small
            statistics['cov_x_0'] = cov_x_0_pred.item()
            statistics['cov_x_0_clamp'] = cov_x_0_pred_clamp.item()
            statistics['coeff_cov_x_0'] = coeff_cov_x_0
            statistics['offset'] = offset.item()
            statistics['sigma2_small/offset'] = statistics['sigma2_small'] / statistics['offset']
            if s < ns[clip_sigma_idx]:  # clip_sigma_idx = 0 <=> not clip
                statistics['sigma2_unclip'] = sigma2.item()
                sigma2_threshold = (clip_pixel * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                sigma2 = np.clip(sigma2, 0., sigma2_threshold)
                statistics['sigma2_threshold'] = sigma2_threshold
            x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            statistics['sigma2'] = sigma2
        else:
            x = x_mean
        _report_statistics(s, r, statistics)
    return x
