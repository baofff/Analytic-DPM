import os
from core.evaluate import grid_sample, sample2dir
from .base import Evaluator
import torch.nn as nn
from core.criterions.ddpm import _rescale_timesteps, _bipartition
from core.inference.sampler.reverse_ddpm import reverse_ddpm_naive, reverse_ddpm_ms_eps
from core.inference.sampler.reverse_ddim import reverse_ddim_naive, reverse_ddim_ms_eps
from core.inference.ll.elbo import nelbo_naive_ddpm, nelbo_ms_eps_ddpm, get_nelbo_terms
from core.evaluate.score import score_on_dataset
from interface.utils.interact import Interact
from interface.datasets import DatasetFactory
from core.utils.managers import ModelsManager
import torch
from core.utils import global_device
from torch.utils.data import Subset
import numpy as np
import logging
import math
import core.func as func
import random


class DDPMNaiveEvaluator(Evaluator):
    def __init__(self, models: ModelsManager, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Only estimate the mean of optimal Gaussian reverse in DDPM
        Args:
            models: an object of ModelsManager
            options: a dict, evaluation function name -> arguments of the function
                Example: {"grid_sample": {"nrow": 10, "ncol": 10}}
            dataset: an instance of DatasetFactory
            interact: an instance of Interact
        """
        super().__init__(options)
        self.models = models
        self.eps_model = nn.DataParallel(models.eps_model) if "eps_model" in models else None
        self.d_model = nn.DataParallel(models.d_model) if "d_model" in models else None
        self.dataset = dataset
        self.unpreprocess_fn = None if self.dataset is None else self.dataset.unpreprocess
        self.interact = interact

    def grid_sample(self, it, betas, small_sigma, clip_denoise, rescale_timesteps, path, sample_steps=None, nrow=10, ncol=10):
        fname = os.path.join(path, "%d.png" % it)

        def sample_fn(n_samples):
            x_init = torch.randn(n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddpm_naive(x_init, betas, small_sigma, clip_denoise, rescale_timesteps,
                                      eps_model=self.eps_model, d_model=self.d_model, sample_steps=sample_steps)
        grid_sample(fname, nrow, ncol, sample_fn, self.unpreprocess_fn)

    def sample2dir(self, path, n_samples, batch_size, betas, small_sigma, clip_denoise,
                   rescale_timesteps, sample_steps=None, steps_type='linear', clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir with {} samples".format(n_samples))

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddpm_naive(x_init, betas, small_sigma, clip_denoise, rescale_timesteps, steps_type=steps_type,
                                      clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, eps_model=self.eps_model,
                                      d_model=self.d_model, sample_steps=sample_steps)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def sample2dir_ddim(self, path, n_samples, batch_size, betas, rescale_timesteps, eta=0., steps_type='linear',
                        sample_steps=None, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ddim with {} samples".format(n_samples))

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddim_naive(x_init, betas, rescale_timesteps, eta=eta, eps_model=self.eps_model,
                                      sample_steps=sample_steps, steps_type=steps_type)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def sample2dir_ms_eps(self, path, n_samples, batch_size, betas, rescale_timesteps, ms_eps_path,
                          sample_steps=None, steps_type='linear', clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ms_eps with {} samples".format(n_samples))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddpm_ms_eps(x_init, betas, rescale_timesteps, steps_type=steps_type,
                                       clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel,
                                       eps_model=self.eps_model, ms_eps=ms_eps, sample_steps=sample_steps)

        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def sample2dir_ddim_ms_eps(self, path, n_samples, batch_size, betas, rescale_timesteps, ms_eps_path, eta=0.,
                               sample_steps=None, clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ddim_ms_eps with {} samples".format(n_samples))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddim_ms_eps(x_init, betas, rescale_timesteps, eta=eta, eps_model=self.eps_model, ms_eps=ms_eps,
                                       sample_steps=sample_steps, clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel)

        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def nll(self, batch_size, betas, small_sigma, clip_denoise, rescale_timesteps, fname, sample_steps=None, n_samples=None, partition="test", it=None):
        if partition == "train":
            dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("nll with {} {} samples".format(n_samples, partition))

        def score_fn(x_0):
            nelbo, terms = nelbo_naive_ddpm(x_0, betas, small_sigma, clip_denoise, rescale_timesteps,
                                            eps_model=self.eps_model, d_model=self.d_model, sample_steps=sample_steps)
            return tuple([nelbo, *terms])
        outputs = score_on_dataset(dataset, score_fn, batch_size)
        outputs_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in outputs]
        nelbo_bpd = outputs_bpd[0]
        terms_bpd = outputs_bpd[1:]
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
        torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, fname)

    def nll_ms_eps(self, batch_size, betas, rescale_timesteps, fname, ms_eps_path,
                   sample_steps=None, steps_type='linear', n_samples=None, partition="test", it=None):
        if partition == "train":
            dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("nll_ms_eps with {} {} samples".format(n_samples, partition))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def score_fn(x_0):
            nelbo, terms = nelbo_ms_eps_ddpm(x_0, betas, rescale_timesteps, steps_type=steps_type,
                                             eps_model=self.eps_model, ms_eps=ms_eps, sample_steps=sample_steps)
            return tuple([nelbo, *terms])
        outputs = score_on_dataset(dataset, score_fn, batch_size)
        outputs_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in outputs]
        nelbo_bpd = outputs_bpd[0]
        terms_bpd = outputs_bpd[1:]
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
        torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, fname)

    def save_ms_eps(self, fname, batch_size, betas, rescale_timesteps, include_val=True, n_samples=None, it=None):
        if include_val:
            dataset = self.dataset.get_train_val_data(labelled=False)
        else:
            dataset = self.dataset.get_train_data(labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("save_ms_eps with {} samples".format(n_samples))

        N = len(betas) - 1
        alphas = 1. - betas
        cum_alphas = alphas.cumprod()
        cum_betas = 1. - cum_alphas

        ms_eps = np.zeros(N + 1, dtype=np.float32)
        for n in range(1, N + 1):
            @ torch.no_grad()
            def score_fn(x_0):
                eps = torch.randn_like(x_0)
                x_n = cum_alphas[n] ** 0.5 * x_0 + cum_betas[n] ** 0.5 * eps
                eps_pred = self.eps_model(x_n, _rescale_timesteps(torch.tensor([n] * x_n.size(0)).type_as(x_n), N, rescale_timesteps))
                return func.mos(eps_pred)
            ms_eps[n] = score_on_dataset(dataset, score_fn, batch_size)
            logging.info("[n: {}] [ms_eps[{}]: {}]".format(n, n, ms_eps[n]))

        torch.save(ms_eps, fname)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(list(range(1, N + 1)), ms_eps[1:])
        plt.savefig("{}.png".format(fname))
        plt.close()

    def save_nll_terms(self, batch_size, betas, small_sigma, rescale_timesteps, fname,
                       partition="test", include_val=True, n_samples=None, it=None):
        if partition == "train":
            if include_val:
                dataset = self.dataset.get_train_val_data(labelled=False)
            else:
                dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("save_nll_terms with {} {} samples".format(n_samples, partition))
        res = get_nelbo_terms(dataset, batch_size, self.dataset.data_dim, betas, small_sigma, rescale_timesteps, eps_model=self.eps_model)
        torch.save(res, fname)

        N = len(betas) - 1
        terms = [*[res['F'][n, n + 1] for n in range(0, N)], res['last_term']]
        terms_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in terms]
        nelbo_bpd = sum(terms_bpd)
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')


class ImprovedDDPMEvaluator(Evaluator):
    def __init__(self, models: ModelsManager, options: dict,
                 dataset: DatasetFactory = None, interact: Interact = None):
        r""" Only estimate the mean of optimal Gaussian reverse in DDPM
        Args:
            models: an object of ModelsManager
            options: a dict, evaluation function name -> arguments of the function
                Example: {"grid_sample": {"nrow": 10, "ncol": 10}}
            dataset: an instance of DatasetFactory
            interact: an instance of Interact
        """
        super().__init__(options)
        self.models = models
        self.model = nn.DataParallel(models.model)  # the L_hybrid model
        self.dataset = dataset
        self.unpreprocess_fn = None if self.dataset is None else self.dataset.unpreprocess
        self.interact = interact

    def sample2dir(self, path, n_samples, batch_size, betas, small_sigma, clip_denoise,
                   rescale_timesteps, sample_steps=None, steps_type='linear', clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir with {} samples".format(n_samples))

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddpm_naive(x_init, betas, small_sigma, clip_denoise, rescale_timesteps, steps_type=steps_type,
                                      clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, eps_model=eps_model, sample_steps=sample_steps, shift1=True)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def sample2dir_ddim(self, path, n_samples, batch_size, betas, rescale_timesteps, eta=0.,
                        sample_steps=None, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ddim with {} samples".format(n_samples))

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddim_naive(x_init, betas, rescale_timesteps, eta=eta, eps_model=eps_model,
                                      sample_steps=sample_steps, shift1=True)
        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def sample2dir_ddim_ms_eps(self, path, n_samples, batch_size, betas, rescale_timesteps, ms_eps_path, eta=0.,
                               sample_steps=None, clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ddim_ms_eps with {} samples".format(n_samples))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddim_ms_eps(x_init, betas, rescale_timesteps, eta=eta, eps_model=eps_model, ms_eps=ms_eps,
                                       sample_steps=sample_steps, clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, shift1=True)

        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def nll(self, batch_size, betas, small_sigma, clip_denoise, rescale_timesteps, fname, sample_steps=None,
            n_samples=None, partition="test", it=None):
        if partition == "train":
            dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("nll with {} {} samples".format(n_samples, partition))

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def score_fn(x_0):
            nelbo, terms = nelbo_naive_ddpm(x_0, betas, small_sigma, clip_denoise, rescale_timesteps,
                                            eps_model=eps_model, sample_steps=sample_steps, shift1=True)
            return tuple([nelbo, *terms])

        outputs = score_on_dataset(dataset, score_fn, batch_size)
        outputs_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in outputs]
        nelbo_bpd = outputs_bpd[0]
        terms_bpd = outputs_bpd[1:]
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
        torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, fname)

    def save_ms_eps(self, fname, batch_size, betas, rescale_timesteps, include_val=True, n_samples=None, it=None):
        if include_val:
            dataset = self.dataset.get_train_val_data(labelled=False)
        else:
            dataset = self.dataset.get_train_data(labelled=False)
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("save_ms_eps with {} samples".format(n_samples))

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        N = len(betas) - 1
        alphas = 1. - betas
        cum_alphas = alphas.cumprod()
        cum_betas = 1. - cum_alphas

        ms_eps = np.zeros(N + 1, dtype=np.float32)
        for n in range(1, N + 1):
            @ torch.no_grad()
            def score_fn(x_0):
                eps = torch.randn_like(x_0)
                x_n = cum_alphas[n] ** 0.5 * x_0 + cum_betas[n] ** 0.5 * eps
                input_n = n - 1
                eps_pred = eps_model(x_n, _rescale_timesteps(torch.tensor([input_n] * x_n.size(0)).type_as(x_n), N, rescale_timesteps))
                return func.mos(eps_pred)
            ms_eps[n] = score_on_dataset(dataset, score_fn, batch_size)
            logging.info("[n: {}] [ms_eps[{}]: {}]".format(n, n, ms_eps[n]))

        torch.save(ms_eps, fname)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.plot(list(range(1, N + 1)), ms_eps[1:])
        plt.savefig("{}.png".format(fname))
        plt.close()

    def sample2dir_ms_eps(self, path, n_samples, batch_size, betas, rescale_timesteps, ms_eps_path,
                          sample_steps=None, steps_type='linear', clip_sigma_idx=0, clip_pixel=2, persist=True, it=None):
        os.makedirs(path, exist_ok=True)

        logging.info("sample2dir_ms_eps with {} samples".format(n_samples))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def sample_fn(_n_samples):
            x_init = torch.randn(_n_samples, *self.dataset.data_shape, device=global_device())
            return reverse_ddpm_ms_eps(x_init, betas, rescale_timesteps, steps_type=steps_type,
                                       clip_sigma_idx=clip_sigma_idx, clip_pixel=clip_pixel, eps_model=eps_model,
                                       ms_eps=ms_eps, sample_steps=sample_steps, shift1=True)

        sample2dir(path, n_samples, batch_size, sample_fn, self.unpreprocess_fn, persist)
        if self.dataset.fid_stat is not None:
            from tools.fid_score import calculate_fid_given_paths
            fid = calculate_fid_given_paths((self.dataset.fid_stat, path))
            logging.info("fid={}".format(fid))

    def nll_ms_eps(self, batch_size, betas, rescale_timesteps, fname, ms_eps_path,
                   sample_steps=None, steps_type='linear', n_samples=None, partition="test", it=None):
        if partition == "train":
            dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        logging.info("nll_ms_eps with {} {} samples".format(n_samples, partition))
        logging.info("load ms_eps from {}".format(ms_eps_path))
        ms_eps = torch.load(ms_eps_path)

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        def score_fn(x_0):
            nelbo, terms = nelbo_ms_eps_ddpm(x_0, betas, rescale_timesteps, steps_type=steps_type,
                                             eps_model=eps_model, ms_eps=ms_eps, sample_steps=sample_steps, shift1=True)
            return tuple([nelbo, *terms])
        outputs = score_on_dataset(dataset, score_fn, batch_size)
        outputs_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in outputs]
        nelbo_bpd = outputs_bpd[0]
        terms_bpd = outputs_bpd[1:]
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
        torch.save({"nelbo_bpd": nelbo_bpd, "terms_bpd": terms_bpd}, fname)

    def save_nll_terms(self, batch_size, betas, small_sigma, rescale_timesteps, fname,
                       partition="test", include_val=True, n_samples=None, it=None):
        if partition == "train":
            if include_val:
                dataset = self.dataset.get_train_val_data(labelled=False)
            else:
                dataset = self.dataset.get_train_data(labelled=False)
        elif partition == "test":
            dataset = self.dataset.get_test_data(labelled=False)
        else:
            raise ValueError
        n_samples = n_samples or len(dataset)
        idxes = random.sample(list(range(len(dataset))), n_samples)
        dataset = Subset(dataset, idxes)

        def eps_model(*inputs):
            return _bipartition(self.model(*inputs))[0]

        logging.info("save_nll_terms with {} {} samples".format(n_samples, partition))
        res = get_nelbo_terms(dataset, batch_size, self.dataset.data_dim, betas, small_sigma, rescale_timesteps,
                              eps_model=eps_model, shift1=True)
        torch.save(res, fname)

        N = len(betas) - 1
        terms = [*[res['F'][n, n + 1] for n in range(0, N)], res['last_term']]
        terms_bpd = [a / (self.dataset.data_dim * math.log(2.)) for a in terms]
        nelbo_bpd = sum(terms_bpd)
        self.interact.report_scalar(nelbo_bpd, it, 'bpd')
        self.interact.report_scalar(sum(terms_bpd[1:]), it, 'continuous_part')
        self.interact.report_scalar(terms_bpd[0], it, 'discrete_part')
