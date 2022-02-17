# Get NLL for the baseline (OT, DDPM, $\sigma_n^2 = \beta_n$)

import torch
import os
import math
import numpy as np


def make_inf(F):  # make F[s, t] = inf for s >= t
    return np.triu(F, 1) + np.tril(np.full(F.shape, float('inf')))


def vectorized_dp(F, N):  # F[s, t] with 0 <= s < t <= N
    F = make_inf(F[: N + 1, : N + 1])

    C = np.full((N + 1, N + 1), float('inf'))
    D = np.full((N + 1, N + 1), -1)

    C[0, 0] = 0
    for k in range(1, N + 1):
        bpds = C[k - 1, :].reshape(N + 1, 1) + F
        C[k] = np.min(bpds, axis=0)
        D[k] = np.argmin(bpds, axis=0)

    return D


def fetch_path(D, N, K):  # find a path of length K (K+1 nodes)
    optpath = []
    t = N
    for k in reversed(range(K + 1)):
        optpath.append(t)
        t = D[k, t]
    return optpath[::-1]


@ torch.no_grad()
def nelbo_dp_ddpm(D_train, test_nll_terms, N, K, trajectory):
    if trajectory == "dp":
        ns = fetch_path(D_train, N, K)
    else:
        raise NotImplementedError

    nelbo = 0.
    rev_terms = []

    term = test_nll_terms['last_term']
    nelbo += term
    rev_terms.append(term)

    for s, r in list(zip(ns, ns[1:]))[::-1]:
        term = test_nll_terms['F'][s, r]
        nelbo += term
        rev_terms.append(term)

    return nelbo, rev_terms[::-1]


def main():
    dataset = "celeba"
    trajectory = "dp"

    if dataset == "cifar10_ls":
        root = 'workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/beta_schedule_linear_num_diffusion_1000/train/nll_terms'
        train_nll_terms_name = '400000_small_sigma_False_n_samples_None_partition_train.nll_terms.pth'
        test_nll_terms_name = '400000_small_sigma_False_n_samples_None_partition_test.nll_terms.pth'
        c = 3 * 32 * 32 * math.log(2.)
    elif dataset == "cifar10_cs":
        root = 'workspace/runner/cifar10/ddpm_dsm/betas_2021-08-19-23-55-32/beta_schedule_cosine_num_diffusion_1000/train/nll_terms'
        train_nll_terms_name = '160000_small_sigma_False_n_samples_None_partition_train.nll_terms.pth'
        test_nll_terms_name = '160000_small_sigma_False_n_samples_None_partition_test.nll_terms.pth'
        c = 3 * 32 * 32 * math.log(2.)
    elif dataset == "imagenet":
        root = "workspace/runner/imagenet64/improved_diffusion/L_hybrid_2021-08-29-20-26-00/cosine4000/train/nll_terms"
        train_nll_terms_name = 'imagenet64_uncond_100M_1500K_small_sigma_False_n_samples_16384_partition_train.nll_terms.pth'
        test_nll_terms_name = 'imagenet64_uncond_100M_1500K_small_sigma_False_n_samples_None_partition_test.nll_terms.pth'
        c = 3 * 64 * 64 * math.log(2.)
    elif dataset == 'celeba':
        root = '../celeba_lsun_codes/nelbo_terms/ema_celeba'
        train_nll_terms_name = 'partition_train_n_samples_16384.pth'
        test_nll_terms_name = 'partition_test_n_samples_None.pth'
        c = 3 * 64 * 64 * math.log(2.)
    else:
        raise ValueError

    train_nll_terms = torch.load(os.path.join(root, train_nll_terms_name))
    F_train = train_nll_terms['F']
    N = len(F_train) - 1
    if trajectory == "dp":
        D_train = vectorized_dp(F_train, N)
    else:
        D_train = None

    for sample_steps in sorted({10, 25, 50, 100, 200, 400, 1000, N}):
        test_nll_terms = torch.load(os.path.join(root, test_nll_terms_name))

        nelbo, terms = nelbo_dp_ddpm(D_train, test_nll_terms, N, sample_steps, trajectory)
        nelbo_bpd = nelbo / c
        terms_bpd = [a / c for a in terms]

        print('sample_steps', sample_steps, 'bpd/continuous_part/discrete_part',
              '{:.2f}/{:.2f}/{:.2f}'.format(nelbo_bpd, sum(terms_bpd[1:]), terms_bpd[0]))


if __name__ == "__main__":
    main()
