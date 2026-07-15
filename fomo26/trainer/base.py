import math

from abc import ABC, abstractmethod

import torch
import lightning as pl

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from fomo26.utils.trainable import mark_trainable
from fomo26.utils.lora import load_lora_state_dict


class BaseTrainer(pl.LightningModule, ABC):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__()
        self.model = model
        self.config = config
        self.gpu_transforms = gpu_transforms
        self.norm_transforms = norm_transforms
        self.save_hyperparameters(ignore=["model", "gpu_transforms", "norm_transforms"])
        self.load_pretrained_checkpoint()
        if self.config.get("lora", False):
            mark_trainable(self.model)

    def load_pretrained_checkpoint(self):
        ckpt_path = self.config.get("pretrained_checkpoint", None)
        if ckpt_path is None:
            return
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if self.config.get("lora", False):
            load_lora_state_dict(self.model, state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def configure_optimizers(self):
        weight_decay = self.config.get("weight_decay", 0.05)
        lr = self.config.get("lr", 1e-4)
        betas = tuple(self.config.get("betas", (0.9, 0.999)))

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = AdamW(
            [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=betas,
        )

        total_steps = self.config.get("total_steps", None)
        if total_steps is None:
            total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(0.15 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def apply_gpu_transforms(self, batch, training=True):
        if self.norm_transforms is not None:
            batch = self.norm_transforms(batch)
        if training and self.gpu_transforms is not None:
            batch = self.gpu_transforms(batch)
        return batch

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def compute_loss(self, outputs, batch):
        raise NotImplementedError("Subclasses must implement the compute_loss method.")

    def training_step(self, batch, batch_idx):
        batch = self.apply_gpu_transforms(batch, training=True)
        outputs = self.forward(batch["image"])
        loss = self.compute_loss(outputs, batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.apply_gpu_transforms(batch, training=False)
        outputs = self.forward(batch["image"])
        loss = self.compute_loss(outputs, batch)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss
