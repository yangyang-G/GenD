from typing import override

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch import optim
from torchvision import models

from src.config import Config
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel, Batch, OutputsForMetrics
from src.utils import logger


class Xception(BaseDeepakeDetectionModel):
    """
    Xception (Inception v3 variant) baseline model for deepfake detection.
    Uses ImageNet pretrained weights and modifies the final FC layer for binary classification.
    """

    def __init__(self, config: Config):
        super().__init__(config, verbose=True)

        # Initialize Inception V3 with ImageNet pretrained weights
        # Xception architecture is similar to Inception with separable convolutions
        # Using InceptionV3 as the base since Xception is not in torchvision
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Modify final FC layer for 2-class classification (real/fake)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, config.num_classes)

        # Initialize output metrics collector
        self.test_step_outputs = OutputsForMetrics()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        logger.print_info(f"InceptionV3 (Xception-style) initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        logger.print_info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        """
        Forward pass through InceptionV3.

        Args:
            inputs: Input tensor [B, C, H, W]

        Returns:
            HeadOutput with logits_labels [B, num_classes]
        """
        # InceptionV3 returns tuple (output, aux_output) during training
        # We only need the main output
        output = self.model(inputs)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        return HeadOutput(logits_labels=logits)

    @override
    def training_step(self, batch, batch_idx):
        """
        Training step: forward pass, compute loss, log metrics.

        Args:
            batch: Batch dictionary containing images and labels
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss = self.criterion(outputs.logits_labels, batch.labels)
        probs = outputs.logits_labels.softmax(dim=1)

        # Log metrics
        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=len(batch.images))

        # Save outputs for metrics calculation
        self.train_step_outputs.labels.update(batch.labels)
        self.train_step_outputs.probs.update(probs.detach())
        self.train_step_outputs.idx.update(batch.idx)

        return loss

    @override
    def validation_step(self, batch, batch_idx):
        """
        Validation step: forward pass, compute loss, collect metrics.

        Args:
            batch: Batch dictionary containing images and labels
            batch_idx: Batch index
        """
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss = self.criterion(outputs.logits_labels, batch.labels)
        probs = outputs.logits_labels.softmax(dim=1)

        # Log metrics
        self.log("val/loss", loss, on_epoch=True, on_step=False, batch_size=len(batch.images))

        # Save outputs for metrics calculation
        self.val_step_outputs.labels.update(batch.labels)
        self.val_step_outputs.probs.update(probs.detach())
        self.val_step_outputs.idx.update(batch.idx)

    @override
    def test_step(self, batch, batch_idx):
        """
        Test step: perform inference and collect metrics.

        Args:
            batch: Batch dictionary containing images and labels
            batch_idx: Batch index
        """
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        probs = outputs.logits_labels.softmax(dim=1)

        # Save outputs for metrics calculation
        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    @override
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and optional lr_scheduler
        """
        config = self.config

        # Setup data to get dataloader length for scheduler
        self.trainer.fit_loop.setup_data()

        # Separate parameters for weight decay and no weight decay
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "bn" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Configure optimizer
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        optimizers = {"optimizer": optimizer}
        scheduler = None

        # Configure LR scheduler
        if config.lr_scheduler == "cosine":
            T_max = config.max_epochs * len(self.trainer.train_dataloader)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.min_lr)

        # Configure warmup
        if config.warmup_epochs > 0:
            total_warmup_steps = int(config.warmup_epochs * len(self.trainer.train_dataloader))
            warmup = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=config.min_lr / config.lr, total_iters=total_warmup_steps
            )

            if scheduler is not None:
                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, scheduler], milestones=[total_warmup_steps]
                )
            else:
                scheduler = warmup

        if scheduler is not None:
            optimizers["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }

        return optimizers

    @override
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (.pth or .ckpt)
        """
        logger.print_info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove "model." prefix if present (from Lightning checkpoint)
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Load state dict
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        """
        Get preprocessing function for input images.

        Returns:
            Preprocessing function that takes PIL Image and returns tensor
        """

        def preprocess(image: Image.Image) -> torch.Tensor:
            return _preprocess(image)

        return preprocess


# ImageNet normalization constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Default preprocessing pipeline for InceptionV3 (requires 299x299 input)
_preprocess = T.Compose(
    [
        T.Resize((299, 299), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


if __name__ == "__main__":
    # Test the model
    from src.config import Config

    config = Config(num_classes=2)
    model = Xception(config)

    # Test with dummy input
    dummy_input = torch.randn(2, 3, 299, 299)
    output = model(dummy_input)
    print(f"Output shape: {output.logits_labels.shape}")
    print(f"Output: {output.logits_labels.softmax(dim=1)}")
