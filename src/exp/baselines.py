from ..config import Config
from ..utils import files

experiments = {
    # ==========================================================================
    # ResNet50 Baseline Experiments
    # ==========================================================================
    # CDFv2 dataset
    "ResNet50-CDFv2": [
        Config(
            checkpoint="weights/ResNet50/resnet50_cdfv2.pth",
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    "ResNet50-imagenet-CDFv2": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",  # Only ImageNet pretrained
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    # DFDC dataset
    "ResNet50-DFDC": [
        Config(
            checkpoint="weights/ResNet50/resnet50_dfdc.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    "ResNet50-imagenet-DFDC": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    # UADFV dataset
    "ResNet50-UADFV": [
        Config(
            checkpoint="weights/ResNet50/resnet50_uadfv.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "ResNet50-imagenet-UADFV": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    # Multi-dataset test
    "ResNet50-all": [
        Config(
            checkpoint="weights/ResNet50/resnet50_ffpp.pth",
            tst_files={
                "CDFv2": files.CDFv2.test,
                "DFDC": files.DFDC.test,
                "UADFV": files.UADFV.test,
            },
        ),
    ],
}


def get_common():
    """Get common configuration settings for all baseline experiments."""
    config = Config()
    config.run_dir = "runs/baselines"
    config.num_workers = 12
    config.wandb = True
    config.wandb_tags = ["baseline", "comparison", "resnet50"]
    config.batch_size = 128
    config.mini_batch_size = 128
    config.devices = "auto"
    config.precision = "bf16-mixed"
    return config


def set_common_settings(experiments):
    """Apply common settings to all experiments."""
    for run_name, modifiers in experiments.items():
        experiments[run_name][0] = Config(
            **{
                **get_common().model_dump(exclude_unset=True),
                **modifiers[0].model_dump(exclude_unset=True),
            }
        )


# Apply common settings to all experiments
set_common_settings(experiments)
