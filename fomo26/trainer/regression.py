import torch
import torch.nn as nn
from torchmetrics.regression import MeanSquaredError

from fomo26.trainer.base import BaseTrainer


class RegressionTrainer(BaseTrainer):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__(model, config, gpu_transforms, norm_transforms)
        self.criterion = nn.HuberLoss()

        self.train_l2 = MeanSquaredError(squared=False)
        self.val_l2 = MeanSquaredError(squared=False)

    def on_train_epoch_start(self) -> None:
        self.train_l2.reset()
        self.val_l2.reset()

    def log_train_metrics(self, outputs, batch):
        labels = batch["target"].float()

        self.train_l2(outputs, labels)

        self.log(
            "train/l2",
            self.train_l2,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

    def log_val_metrics(self, outputs, batch):
        labels = batch["target"].float()

        self.val_l2(outputs, labels)

        self.log(
            "val/l2",
            self.val_l2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        labels = batch["target"].float()
        return self.criterion(outputs, labels)
