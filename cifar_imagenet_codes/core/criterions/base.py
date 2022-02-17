import torch
import core.utils.managers as managers
from core.utils import global_device, diagnose


class Criterion(object):
    def __init__(self,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager
                 ):
        r""" Criterion does
            1. calculating objectives
            2. calculating gradients
            3. updating parameters

        Args:
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager
            lr_schedulers: an object of LRSchedulersManager
        """
        self.statistics = {}
        self.models = models
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.device = global_device()

    def objective(self, v, **kwargs):
        raise NotImplementedError

    def update(self, data_loader):
        raise NotImplementedError

    def default_val_fn(self, v):
        r""" Advise a validation function
        """
        return self.objective(v)

    def criterion_name(self):
        return self.__class__.__name__.lower()

    def record_grad_norm(self):
        for key, model in self.models.__dict__.items():
            self.statistics["grad_norm2_%s" % key] = diagnose.grad_norm(model, 2.)
            self.statistics["grad_norminf_%s" % key] = diagnose.grad_norm(model, float('inf'))


class MultilevelCriterion(Criterion):
    def __init__(self,
                 models: managers.ModelsManager,
                 optimizers: managers.OptimizersManager,
                 lr_schedulers: managers.LRSchedulersManager,
                 levels: list,
                 level_n_steps: dict,
                 level_model_keys: dict
                 ):
        r""" Sometimes the optimization might have multiple levels, e.g., bilevel optimization
            MultilevelCriterion does
            for level in levels:
                1. calculating objectives
                2. calculating gradients
                3. updating parameters

        Args:
            models: an object of ModelsManager
            optimizers: an object of OptimizersManager
                The optimizers inside are indexed by levels
            lr_schedulers: an object of LRSchedulersManager
                The schedulers inside are indexed by levels
            levels: the name of each level
                Example: levels = ["lower", "higher"]
            level_n_steps: the steps of each level
                Example: level_n_steps = {"lower": 5, "higher": 1}
            level_model_keys: the models to update in each level
                Example: level_model_keys = {"lower": ["discriminator"], "higher": ["generator"]}

        """
        super().__init__(models, optimizers, lr_schedulers)
        self.levels = levels
        self.level_n_steps = level_n_steps
        self.level_model_keys = level_model_keys

    def update(self, data_loader):
        r""" A demo of the multiple-level optimization
        """
        v = next(data_loader).to(self.device)
        for level in self.levels:
            self.models.toggle_grad(*self.level_model_keys[level])
            for i in range(self.level_n_steps[level]):
                objective = self.objective(v, level=level).mean()
                self.statistics[level] = objective.item()
                self.optimizers.get(level).zero_grad()
                objective.backward()
                self.optimizers.get(level).step()
            if level in self.lr_schedulers:
                self.lr_schedulers.get(level).step()


class NaiveCriterion(Criterion):
    def update(self, data_loader):
        v = next(data_loader).to(self.device)
        loss = self.objective(v).mean()
        self.statistics[self.criterion_name()] = loss.item()
        self.optimizers.all.zero_grad()
        loss.backward()
        self.optimizers.all.step()
        if "all" in self.lr_schedulers:
            self.lr_schedulers.all.step()
        # self.record_grad_norm()
