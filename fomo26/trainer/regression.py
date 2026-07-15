import torch.nn as nn

from fomo26.trainer.base import BaseTrainer


class RegressionTrainer(BaseTrainer):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__(model, config, gpu_transforms, norm_transforms)
        self.criterion = nn.HuberLoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        labels = batch["CLSREG_label"].float()
        return self.criterion(outputs, labels)
