import core.modules as modules
import interface.datasets as datasets
import interface.evaluators as evaluators
import numpy as np
import profiles.ddpm.beta_schedules as beta_schedules
from interface.utils import dict_utils


default_betas = np.append(0., beta_schedules.get_named_beta_schedule("cosine", 4000))


unet_model = {
    "class": modules.UNetModel,
    "kwargs": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": (32 // 16, 32 // 8),
        "dropout": 0.3,
        "channel_mult": (1, 2, 2, 2),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
    }
}


unet_model_double = dict_utils.merge_dict(unet_model, {
    "kwargs": {
        "out_channels": 6
    }
})


dataset = {
    "use_val": False,
    "class": datasets.Cifar10,
    "kwargs": {
        "data_path": "workspace/datasets/cifar10/",
    }
}


ddpm_naive_evaluator_train = {
    "class": evaluators.DDPMNaiveEvaluator,
    "kwargs": {
        "options": {
            "grid_sample": {
                "period": 5000,
                "kwargs": {  # fast sampling
                    "betas": "$(betas)",
                    "small_sigma": True,  # small sample steps require small sigma
                    "clip_denoise": True,  # must be true since it will improve the sample quality
                    "rescale_timesteps": "$(rescale_timesteps)",
                    "sample_steps": 50,
                    "path": "os.path.join($(workspace_root), 'train/evaluator/grid_sample')"
                }
            },
        }
    }
}
