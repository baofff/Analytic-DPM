
__all__ = ["sample2dir_eps", "sample2dir_ms_eps", "sample2dir_d", "nll_eps", "nll_ms_eps", "nll_d", "save_ms_eps", "save_nll_terms"]


import interface.evaluators as evaluators
from .base import unet_model, dataset, default_betas
from interface.utils import dict_utils
import profiles.common as common


sample2dir = {
    "ema": True,
    "betas": default_betas,
    "small_sigma": False,  # the default setting used in model selection
    "sample_steps": None,  # the default setting used in model selection
    "rescale_timesteps": True,
    "clip_sigma_idx": 0,
    "clip_pixel": 2,
    "steps_type": "linear",
    "dataset": dataset,
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "sample2dir": {
                    "kwargs": {
                        "path": "$(path)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "small_sigma": "$(small_sigma)",
                        "sample_steps": "$(sample_steps)",
                        "clip_denoise": True,  # must be true since it will improve the sample quality
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "clip_sigma_idx": "$(clip_sigma_idx)",
                        "clip_pixel": "$(clip_pixel)",
                        "steps_type": "$(steps_type)"
                    }
                },
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


sample2dir_ddim = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,  # the default setting used in model selection
    "rescale_timesteps": True,
    "steps_type": "linear",
    "dataset": dataset,
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "sample2dir_ddim": {
                    "kwargs": {
                        "path": "$(path)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "sample_steps": "$(sample_steps)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "eta": "$(eta)",
                        "steps_type": "$(steps_type)"
                    }
                },
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


sample2dir_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,  # the default setting used in model selection
    "rescale_timesteps": True,
    "clip_sigma_idx": 0,
    "clip_pixel": 2,
    "dataset": dataset,
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "sample2dir_ms_eps": {
                    "kwargs": {
                        "path": "$(path)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "steps_type": "$(steps_type)",
                        "betas": "$(betas)",
                        "sample_steps": "$(sample_steps)",
                        "ms_eps_path": "$(ms_eps_path)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "clip_sigma_idx": "$(clip_sigma_idx)",
                        "clip_pixel": "$(clip_pixel)"
                    }
                },
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


sample2dir_ddim_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,  # the default setting used in model selection
    "rescale_timesteps": True,
    "clip_sigma_idx": 0,
    "clip_pixel": 2,
    "dataset": dataset,
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "sample2dir_ddim_ms_eps": {
                    "kwargs": {
                        "path": "$(path)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "sample_steps": "$(sample_steps)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "eta": "$(eta)",
                        "ms_eps_path": "$(ms_eps_path)",
                        "clip_sigma_idx": "$(clip_sigma_idx)",
                        "clip_pixel": "$(clip_pixel)"
                    }
                },
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


nll = {
    "ema": True,
    "betas": default_betas,
    "small_sigma": False,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "partition": "test",
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "nll": {
                    "kwargs": {
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "small_sigma": "$(small_sigma)",
                        "sample_steps": "$(sample_steps)",
                        "clip_denoise": True,
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "fname": "$(fname)",
                        "partition": "$(partition)"
                    }
                }
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


nll_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "partition": "test",
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "nll_ms_eps": {
                    "kwargs": {
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "sample_steps": "$(sample_steps)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "fname": "$(fname)",
                        "partition": "$(partition)",
                        "ms_eps_path": "$(ms_eps_path)",
                        "steps_type": "$(steps_type)"
                    }
                }
            }
        }
    },
    "interact": common.interact_datetime_evaluate()
}


save_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "save_ms_eps": {
                    "kwargs": {
                        "fname": "$(fname)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                    }
                }
            }
        }
    },
}


save_nll_terms = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "models": {
        "eps_model": unet_model
    },
    "evaluator": {
        "class": evaluators.DDPMNaiveEvaluator,
        "kwargs": {
            "options": {
                "save_nll_terms": {
                    "kwargs": {
                        "fname": "$(fname)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "small_sigma": "$(small_sigma)",
                        "partition": "$(partition)",
                    }
                }
            }
        }
    },
}


sample2dir_eps = dict_utils.merge_dict(sample2dir, {
    "models": {
        "eps_model": unet_model
    },
})


sample2dir_d = dict_utils.merge_dict(sample2dir, {
    "models": {
        "d_model": unet_model
    },
})


nll_eps = dict_utils.merge_dict(nll, {
    "models": {
        "eps_model": unet_model
    },
})


nll_d = dict_utils.merge_dict(nll, {
    "models": {
        "d_model": unet_model
    },
})
