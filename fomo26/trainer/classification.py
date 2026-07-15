import torch.nn as nn
import torch.nn.functional as F

from fomo26.trainer.base import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__(model, config, gpu_transforms, norm_transforms)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        num_classes = outputs.shape[-1]
        one_hot_labels = F.one_hot(
            batch["CLSREG_label"].long().squeeze(-1), num_classes
        ).to(outputs.dtype)
        return self.criterion(outputs, one_hot_labels)
