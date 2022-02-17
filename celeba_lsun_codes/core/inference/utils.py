import torch
import logging
import numpy as np
import math


def _rescale_timesteps(n, N, flag):
    if flag:
        return n * 1000.0 / float(N)
    return n


def _report_statistics(s, r, statistics):
    statistics_str = {k: "{:.5e}".format(v) for k, v in statistics.items()}
    logging.info("[(s, r): ({}, {})] [{}]".format(s, r, statistics_str))


def _x_0_pred(x, n, cum_alphas, rescale_timesteps, eps_model=None, d_model=None):  # estimate of E[x_0|x_n] w.r.t. q
    N = len(cum_alphas) - 1
    cum_alpha_n = cum_alphas[n]
    cum_beta_n = 1. - cum_alpha_n
    if eps_model is not None:
        eps_pred = eps_model(x, _rescale_timesteps(torch.tensor([n] * x.size(0)).type_as(x), N, rescale_timesteps))
        x_0_pred = cum_alpha_n ** -0.5 * x - (1. / cum_alpha_n - 1.) ** 0.5 * eps_pred
    else:
        x_0_pred = d_model(x, _rescale_timesteps(torch.tensor([n] * x.size(0)).type_as(x), N, rescale_timesteps))
        eps_pred = - (cum_alpha_n / cum_beta_n) ** 0.5 * x_0_pred + (1. / cum_beta_n ** 0.5) * x
    return x_0_pred, eps_pred


def _cov_x_0_pred(x, n, cum_alphas, cum_betas, rescale_timesteps, tau_model=None, eps_pred=None, kappa_model=None, x_0_pred=None):  # estimate Cov[x_0|x_n] w.r.t. q
    N = len(cum_alphas) - 1
    cum_alpha_n, cum_beta_n = cum_alphas[n], cum_betas[n]
    if tau_model is not None:
        tau_pred = tau_model(x, _rescale_timesteps(torch.tensor([n] * x.size(0)).type_as(x), N, rescale_timesteps))
        delta_pred = tau_pred - eps_pred.pow(2)
        cov_x_0_pred = cum_beta_n / cum_alpha_n * delta_pred
    else:
        x_0_2_pred = kappa_model(x, _rescale_timesteps(torch.tensor([n] * x.size(0)).type_as(x), N, rescale_timesteps))
        cov_x_0_pred = x_0_2_pred - x_0_pred.pow(2)
    return cov_x_0_pred


def _choice_steps_linear(N, sample_steps):
    assert sample_steps > 1
    frac_stride = (N - 1) / (sample_steps - 1)
    cur_idx = 1.0
    steps = []
    for _ in range(sample_steps):
        steps.append(round(cur_idx))
        cur_idx += frac_stride
    return steps


def _choice_steps_linear_ddim(N, sample_steps):
    skip = N // sample_steps
    seq = list(range(1, N + 1, skip))
    return seq


def _choice_steps_quad_ddim(N, sample_steps):
    seq = np.linspace(0, np.sqrt(N * 0.8), sample_steps) ** 2
    seq = [int(s) + 1 for s in list(seq)]
    return seq


def _split(ms_eps, N, K):
    idx_g1 = N + 1
    for n in range(1, N):  # Theoretically, ms_eps <= 1. Remove points of poor estimation
        if ms_eps[n] > 1:
            idx_g1 = n
            break
    num_bad = 2 * (N - idx_g1 + 1)
    bad_ratio = num_bad / N

    N1 = N - num_bad
    K1 = math.ceil((1. - 0.8 * bad_ratio) * K)
    K2 = K - K1
    if K1 > N1:
        K1 = N1
        K2 = K - K1
    if K2 > num_bad:
        K2 = num_bad
        K1 = K - K2
    if num_bad > 0 and K2 == 0:
        K2 = 1
        K1 = K - K2
    assert num_bad <= N
    assert K1 <= N1 and K2 <= N - N1
    return K1, N1, K2, num_bad


def _ms_score(ms_eps, betas):
    alphas = 1. - betas
    cum_alphas = alphas.cumprod()
    cum_betas = 1. - cum_alphas
    ms_score = np.zeros_like(ms_eps)
    ms_score[1:] = ms_eps[1:] / cum_betas[1:]
    return ms_score


def _solve_fn_dp(fn, N, K):  # F[st, ed] with 1 <= st < ed <= N, other elements is inf
    if N == K:
        return list(range(1, N + 1))

    F = fn[: N + 1, : N + 1]

    C = np.full((K + 1, N + 1), float('inf'))  # C[k, n] with 2 <= k <= K, k <= n <= N
    D = np.full((K + 1, N + 1), -1)  # D[k, n] with 2 <= k <= K, k <= n <= N

    C[2, 2: N] = F[1, 2: N]
    D[2, 2: N] = 1

    for k in range(3, K + 1):
        # {C[k-1, s] + F[s, r]}_{0 <= s, r <= N} = {C[k-1, s] + F[s, r]}_{k-1 <= s < r <= N}
        tmp = C[k - 1, :].reshape(N + 1, 1) + F
        C[k, k: N + 1] = np.min(tmp, axis=0)[k: N + 1]
        D[k, k: N + 1] = np.argmin(tmp, axis=0)[k: N + 1]

    res = [N]
    n, k = N, K
    while k > 2:
        n = D[k, n]
        res.append(n)
        k -= 1
    res.append(1)
    return res[::-1]


def _get_fn_m(ms_score, alphas, N):
    F = np.full((N + 1, N + 1), float('inf'))  # F[st, ed] with 1 <= st < ed <= N
    for s in range(1, N + 1):
        skip_alphas = alphas[s + 1: N + 1].cumprod()
        skip_betas = 1. - skip_alphas
        before_log = 1. - skip_betas * ms_score[s + 1: N + 1]
        F[s, s + 1: N + 1] = np.log(before_log)
    return F


def _dp_seg(ms_eps, betas, N, K):
    K1, N1, K2, num_bad = _split(ms_eps, N, K)

    alphas = 1. - betas
    ms_score = _ms_score(ms_eps, betas)
    F = _get_fn_m(ms_score, alphas, N1)

    steps1 = _solve_fn_dp(F, N1, K1)
    if K2 > 0:
        frac = (N - N1) / K2
        steps2 = [round(N - frac * k) for k in range(K2)][::-1]
        assert steps1[-1] < steps2[0]
        assert len(steps1) + len(steps2) == K
        assert steps1[0] == 1 and steps1[-1] == N1
        assert steps2[-1] == N
    else:
        steps2 = []
    steps = steps1 + steps2
    assert steps[0] == 1 and steps[-1] == N
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))
    return steps


def _choice_steps(N, sample_steps, typ, ms_eps=None, betas=None):
    if typ == 'linear':
        steps = _choice_steps_linear(N, sample_steps)
    elif typ == 'linear_ddim':
        steps = _choice_steps_linear_ddim(N, sample_steps)
    elif typ == 'quad_ddim':
        steps = _choice_steps_quad_ddim(N, sample_steps)
    elif typ == 'dp_seg':
        steps = _dp_seg(ms_eps, betas, N, sample_steps)
    else:
        raise NotImplementedError

    assert len(steps) == sample_steps and steps[0] == 1
    if typ not in ["linear_ddim", "quad_ddim"]:
        assert steps[-1] == N

    return steps
