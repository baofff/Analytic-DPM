import core.func as func
import numpy as np
import torch
from tqdm import tqdm
from core.inference.utils import _x_0_pred, _choice_steps, _report_statistics
import logging


@ torch.no_grad()
def nelbo_naive_ddpm(x_0, betas, small_sigma, clip_denoise, rescale_timesteps, eps_model=None, d_model=None, sample_steps=None):
    assert (eps_model is None and d_model is not None) or (eps_model is not None and d_model is None)
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, 'linear')
    assert ns[0] == 1 and ns[-1] == N and len(ns) == sample_steps
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("nelbo_naive_ddpm with {}, small_sigma={}, clip_denoise={}, sample_steps={}"
                 .format("eps_model" if eps_model is not None else "d_model", small_sigma, clip_denoise, sample_steps))

    nelbo = torch.zeros(x_0.size(0), device=x_0.device)
    rev_terms = []

    mu_q = cum_alphas[N] ** 0.5 * x_0
    var_q = cum_betas[N]
    mu_p = torch.zeros_like(mu_q)
    var_p = 1.
    term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
    nelbo += term
    rev_terms.append(term)

    for s, r in tqdm(list(zip([0] + ns, ns))[::-1]):
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[r]
        eps = torch.randn_like(x_0)
        x_r = cum_alpha_r ** 0.5 * x_0 + cum_beta_r ** 0.5 * eps

        coeff1 = skip_beta * cum_alpha_s ** 0.5 / (1. - cum_alpha_r)
        coeff2 = skip_alpha ** 0.5 * (1. - cum_alpha_s) / (1. - cum_alpha_r)
        x_0_pred, eps_pred = _x_0_pred(x_r, r, cum_alphas, rescale_timesteps, eps_model, d_model)
        if clip_denoise:
            x_0_pred = x_0_pred.clamp(-1., 1.)
        mu_p = coeff1 * x_0_pred + coeff2 * x_r

        # if small_sigma:
        #     var_p = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r) if s != 0 else var_p
        # else:
        #     var_p = skip_beta

        if s != 0:
            var_p = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r) if small_sigma else skip_beta
        else:
            var_p = _sigma2_small(1, 2, alphas, cum_alphas)

        if s != 0:
            mu_q = coeff1 * x_0 + coeff2 * x_r
            var_q = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r)
            term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
        else:
            term = -func.log_discretized_normal(x_0, mu_p, var_p).flatten(1).sum(1)
        nelbo += term
        rev_terms.append(term)

    return nelbo, rev_terms[::-1]


def _sigma2_small(s, r, alphas, cum_alphas):
    skip_alpha = alphas[s + 1: r + 1].prod()
    skip_beta = 1. - skip_alpha
    cum_alpha_s, cum_alpha_r = cum_alphas[s], cum_alphas[r]
    sigma2_small = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r)
    return sigma2_small


@ torch.no_grad()
def nelbo_ms_eps_ddpm(x_0, betas, rescale_timesteps, steps_type='linear', eps_model=None, ms_eps=None, sample_steps=None):
    assert eps_model is not None and ms_eps is not None
    assert isinstance(betas, np.ndarray) and betas[0] == 0
    N = len(betas) - 1
    sample_steps = sample_steps or N
    ns = _choice_steps(N, sample_steps, steps_type, ms_eps=ms_eps, betas=betas)
    assert ns[0] == 1 and ns[-1] == N and len(ns) == sample_steps
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas

    logging.info("nelbo_ms_eps_ddpm with eps_model, sample_steps={}, steps_type={}".format(sample_steps, steps_type))

    nelbo = torch.zeros(x_0.size(0), device=x_0.device)
    rev_terms = []

    mu_q = cum_alphas[N] ** 0.5 * x_0
    var_q = cum_betas[N]
    mu_p = torch.zeros_like(mu_q)
    var_p = 1.
    term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
    nelbo += term
    rev_terms.append(term)

    for s, r in list(zip([0] + ns, ns))[::-1]:
        statistics = {}
        skip_alpha = alphas[s + 1: r + 1].prod()
        skip_beta = 1. - skip_alpha
        cum_alpha_s, cum_alpha_r, cum_beta_r = cum_alphas[s], cum_alphas[r], cum_betas[r]
        statistics['skip_beta'] = skip_beta
        statistics['cum_beta_r'] = cum_beta_r
        statistics['cum_alpha_s'] = cum_alpha_s

        eps = torch.randn_like(x_0)
        x_r = cum_alpha_r ** 0.5 * x_0 + cum_beta_r ** 0.5 * eps

        coeff1 = skip_beta * cum_alpha_s ** 0.5 / (1. - cum_alpha_r)
        coeff2 = skip_alpha ** 0.5 * (1. - cum_alpha_s) / (1. - cum_alpha_r)
        x_0_pred, eps_pred = _x_0_pred(x_r, r, cum_alphas, rescale_timesteps, eps_model)
        x_0_pred_clamp = x_0_pred.clamp(-1., 1.)
        mu_p = coeff1 * x_0_pred_clamp + coeff2 * x_r

        if s != 0:
            sigma2_small = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r)
            cov_x_0_pred = cum_beta_r / cum_alpha_r * (1. - ms_eps[r])
            cov_x_0_pred_clamp = np.clip(cov_x_0_pred, 0., 1.)
            coeff_cov_x_0 = cum_alpha_s * skip_beta ** 2 / cum_beta_r ** 2
            offset = coeff_cov_x_0 * cov_x_0_pred_clamp
            var_p = sigma2_small + offset
            statistics['sigma2_small'] = sigma2_small
            statistics['cov_x_0'] = cov_x_0_pred.item()
            statistics['cov_x_0_clamp'] = cov_x_0_pred_clamp.item()
            statistics['coeff_cov_x_0'] = coeff_cov_x_0
            statistics['offset'] = offset.item()
        else:
            var_p = _sigma2_small(1, 2, alphas, cum_alphas)
        statistics['var_p'] = var_p.item()

        if s != 0:
            mu_q = coeff1 * x_0 + coeff2 * x_r
            var_q = skip_beta * (1. - cum_alpha_s) / (1. - cum_alpha_r)
            statistics['var_q'] = var_q
            term = func.kl_between_normal(mu_q, var_q, mu_p, var_p).flatten(1).sum(1)
        else:
            term = -func.log_discretized_normal(x_0, mu_p, var_p).flatten(1).sum(1)

        nelbo += term
        rev_terms.append(term)
        _report_statistics(s, r, statistics)

    return nelbo, rev_terms[::-1]
