import torch
import numpy as np
import logging
import math
from core.inference.utils import _choice_steps, _x_0_pred, _report_statistics


@ torch.no_grad()
def reverse_ddim_naive(x_init, betas, rescale_timesteps, eta=0., steps_type='linear', eps_model=None, sample_steps=None):
    assert eps_model is not None
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, typ=steps_type)
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("reverse_ddim_naive with eps_model, rescale_timesteps={}, eta={}, sample_steps={}, steps_type={}"
                 .format(rescale_timesteps, eta, sample_steps, steps_type))

    x = x_init
    for s, r in list(zip([0] + ns, ns))[::-1]:
        statistics = {}
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_s, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[s], cum_betas[r]
        sigma2_small = skip_beta * cum_beta_s / cum_beta_r
        lamb2 = eta ** 2 * sigma2_small
        statistics['skip_beta'] = skip_beta
        statistics['cum_beta_s'] = cum_beta_s
        statistics['cum_beta_r'] = cum_beta_r
        statistics['cum_alpha_s'] = cum_alpha_s

        x_0_pred, eps_pred = _x_0_pred(x, r, cum_alphas, rescale_timesteps, eps_model=eps_model)
        x_0_pred_clamp = x_0_pred.clamp(-1., 1.)
        coeff1 = cum_alpha_s ** 0.5
        coeff2 = (cum_beta_s - lamb2) ** 0.5
        x_mean = coeff1 * x_0_pred_clamp + coeff2 * eps_pred
        if s != 0:
            sigma2 = lamb2
            x = x_mean + sigma2 ** 0.5 * torch.randn_like(x)
            statistics['sigma2'] = sigma2
        else:
            x = x_mean
        _report_statistics(s, r, statistics)
    return x


@ torch.no_grad()
def reverse_ddim_ms_eps(x_init, betas, rescale_timesteps, steps_type='linear', eta=0., eps_model=None,
                        ms_eps=None, sample_steps=None, clip_sigma_idx=0, clip_pixel=2):
    assert eps_model is not None and ms_eps is not None
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, typ=steps_type, ms_eps=ms_eps, betas=betas)
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("reverse_ddim_ms_eps with eps_model, rescale_timesteps={}, eta={}, sample_steps={}, steps_type={}, clip_sigma_idx={}, clip_pixel={}"
                 .format(rescale_timesteps, eta, sample_steps, steps_type, clip_sigma_idx, clip_pixel))

    x = x_init
    for s, r in list(zip([0] + ns, ns))[::-1]:
        statistics = {}
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_s, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[s], cum_betas[r]
        sigma2_small = skip_beta * cum_beta_s / cum_beta_r
        lamb2 = eta ** 2 * sigma2_small
        statistics['skip_beta'] = skip_beta
        statistics['cum_beta_s'] = cum_beta_s
        statistics['cum_beta_r'] = cum_beta_r
        statistics['cum_alpha_s'] = cum_alpha_s

        x_0_pred, eps_pred = _x_0_pred(x, r, cum_alphas, rescale_timesteps, eps_model=eps_model)
        x_0_pred_clamp = x_0_pred.clamp(-1., 1.)
        coeff1 = cum_alpha_s ** 0.5
        coeff2 = (cum_beta_s - lamb2) ** 0.5
        x_mean = coeff1 * x_0_pred_clamp + coeff2 * eps_pred
        if s != 0:
            cov_x_0_pred = cum_beta_r / cum_alpha_r * (1. - ms_eps[r])
            cov_x_0_pred_clamp = np.clip(cov_x_0_pred, 0., 1.)
            coeff_cov_x_0 = (cum_alpha_s ** 0.5 - ((cum_beta_s - lamb2) * cum_alpha_r / cum_beta_r) ** 0.5) ** 2
            offset = coeff_cov_x_0 * cov_x_0_pred_clamp
            sigma2 = lamb2 + offset
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
