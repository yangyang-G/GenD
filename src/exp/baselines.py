from ..config import Config
from ..utils import files


DFDC_PRE_X1_3_TH0_5_ALL_TEST = files.DFDC.test.map(lambda x: x.replace("/DFDC/", "/DFDC-pre-x1.3-th0.5-all/"))

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
    "ResNet50-UADFD": [
        Config(
            checkpoint="weights/ResNet50/resnet50_uadfv.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "ResNet50-FF": [
        Config(
            checkpoint="weights/ResNet50/resnet50_ffpp.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "ResNet50-FFIW": [
        Config(
            checkpoint="weights/ResNet50/resnet50_ffpp.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "ResNet50-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/ResNet50/resnet50_dfdc.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    "ResNet50-imagenet-UADFV": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "ResNet50-imagenet-UADFD": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "ResNet50-imagenet-FF": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "ResNet50-imagenet-FFIW": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "ResNet50-imagenet-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/ResNet50/resnet50_imagenet.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    # Multi-dataset test
    "ResNet50-all": [
        Config(
            checkpoint="weights/ResNet50/resnet50_ffpp.pth",
            tst_files={
                "CDFv2": files.CDFv2.test,
                "FF": files.FF.test,
                "DFDC": files.DFDC.test,
                "DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST,
                "UADFV": files.UADFV.test,
                "UADFD": files.UADFV.test,
                "FFIW": files.FFIW.test,
            },
        ),
    ],

    "EfficientNet-CDFv2": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_cdfv2.pth",
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    "EfficientNet-imagenet-CDFv2": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",  # Only ImageNet pretrained
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    # DFDC dataset
    "EfficientNet-DFDC": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_dfdc.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    "EfficientNet-imagenet-DFDC": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    # UADFV dataset
    "EfficientNet-UADFV": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_uadfv.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "EfficientNet-UADFD": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_uadfv.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "EfficientNet-FF": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_ffpp.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "EfficientNet-FFIW": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_ffpp.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "EfficientNet-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_dfdc.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    "EfficientNet-imagenet-UADFV": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "EfficientNet-imagenet-UADFD": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "EfficientNet-imagenet-FF": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "EfficientNet-imagenet-FFIW": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "EfficientNet-imagenet-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_imagenet.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    # Multi-dataset test
    "EfficientNet-all": [
        Config(
            checkpoint="weights/EfficientNet/efficientnet_ffpp.pth",
            tst_files={
                "CDFv2": files.CDFv2.test,
                "FF": files.FF.test,
                "DFDC": files.DFDC.test,
                "DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST,
                "UADFV": files.UADFV.test,
                "UADFD": files.UADFV.test,
                "FFIW": files.FFIW.test,
            },
        ),
    ],
    "Xception-CDFv2": [
        Config(
            checkpoint="weights/Xception/xception_cdfv2.pth",
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    "Xception-imagenet-CDFv2": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",  # Only ImageNet pretrained
            tst_files={"CDFv2": files.CDFv2.test},
        ),
    ],
    # DFDC dataset
    "Xception-DFDC": [
        Config(
            checkpoint="weights/Xception/xception_dfdc.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    "Xception-imagenet-DFDC": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"DFDC": files.DFDC.test},
        ),
    ],
    # UADFV dataset
    "Xception-UADFV": [
        Config(
            checkpoint="weights/Xception/xception_uadfv.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "Xception-UADFD": [
        Config(
            checkpoint="weights/Xception/xception_uadfv.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "Xception-FF": [
        Config(
            checkpoint="weights/Xception/xception_ffpp.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "Xception-FFIW": [
        Config(
            checkpoint="weights/Xception/xception_ffpp.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "Xception-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/Xception/xception_dfdc.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    "Xception-imagenet-UADFV": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"UADFV": files.UADFV.test},
        ),
    ],
    "Xception-imagenet-UADFD": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"UADFD": files.UADFV.test},
        ),
    ],
    "Xception-imagenet-FF": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"FF": files.FF.test},
        ),
    ],
    "Xception-imagenet-FFIW": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"FFIW": files.FFIW.test},
        ),
    ],
    "Xception-imagenet-DFDC-pre-x1.3-th0.5-all": [
        Config(
            checkpoint="weights/Xception/xception_imagenet.pth",
            tst_files={"DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST},
        ),
    ],
    # Multi-dataset test
    "Xception-all": [
        Config(
            checkpoint="weights/Xception/xception_ffpp.pth",
            tst_files={
                "CDFv2": files.CDFv2.test,
                "FF": files.FF.test,
                "DFDC": files.DFDC.test,
                "DFDC-pre-x1.3-th0.5-all": DFDC_PRE_X1_3_TH0_5_ALL_TEST,
                "UADFV": files.UADFV.test,
                "UADFD": files.UADFV.test,
                "FFIW": files.FFIW.test,
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
