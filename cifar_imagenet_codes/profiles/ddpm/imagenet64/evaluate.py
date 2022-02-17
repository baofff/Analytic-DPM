
__all__ = ["sample2dir", "nll", "save_ms_eps", "nll_ms_eps", "sample2dir_ms_eps"]


import interface.evaluators as evaluators
from .base import unet_model, default_betas, dataset
import profiles.common as common


sample2dir = {
    "ema": True,
    "betas": default_betas,
    "small_sigma": False,
    "sample_steps": None,
    "rescale_timesteps": True,
    "clip_sigma_idx": 0,
    "clip_pixel": 2,
    "dataset": dataset,
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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
                        "clip_pixel": "$(clip_pixel)"
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
    "dataset": dataset,
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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
                        "eta": "$(eta)"
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
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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


save_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
        "kwargs": {
            "options": {
                "save_ms_eps": {
                    "kwargs": {
                        "fname": "$(fname)",
                        "n_samples": "$(n_samples)",
                        "batch_size": "$(batch_size)",
                        "betas": "$(betas)",
                        "rescale_timesteps": "$(rescale_timesteps)",
                        "include_val": False
                    }
                }
            }
        }
    },
}


sample2dir_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "rescale_timesteps": True,
    "clip_sigma_idx": 0,
    "clip_pixel": 2,
    "dataset": dataset,
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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


nll_ms_eps = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "partition": "test",
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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


save_nll_terms = {
    "ema": True,
    "betas": default_betas,
    "sample_steps": None,
    "n_samples": None,
    "rescale_timesteps": True,
    "dataset": dataset,
    "models": {
        "model": unet_model
    },
    "evaluator": {
        "class": evaluators.ImprovedDDPMEvaluator,
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
                        "include_val": False
                    }
                }
            }
        }
    },
}
