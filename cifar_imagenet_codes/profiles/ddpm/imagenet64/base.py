import core.modules as modules
import numpy as np
import profiles.ddpm.beta_schedules as beta_schedules
import interface.datasets as datasets


default_betas = np.append(0., beta_schedules.get_named_beta_schedule("cosine", 4000))


unet_model = {
    "class": modules.UNetModel,
    "kwargs": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 6,
        "num_res_blocks": 3,
        "attention_resolutions": (64 // 16, 64 // 8),
        "dropout": 0.0,
        "channel_mult": (1, 2, 3, 4),
        "conv_resample": True,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
    }
}


dataset = {
    "use_val": False,
    "class": datasets.Imagenet64,
    "kwargs": {
        "path": "workspace/datasets/imagenet64"
    }
}
