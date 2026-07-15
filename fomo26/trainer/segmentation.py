import torch
import torch.nn as nn
import torch.nn.functional as F

from fomo26.trainer.base import BaseTrainer


class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        target = target.long()
        if target.ndim == logits.ndim:
            target = target.squeeze(1)
        ce_loss = self.ce(logits, target)

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=num_classes)
        target_onehot = target_onehot.permute(
            0, -1, *range(1, target_onehot.ndim - 1)
        ).float()

        dims = (0,) + tuple(range(2, probs.ndim))
        intersection = torch.sum(probs * target_onehot, dim=dims)
        cardinality = torch.sum(probs + target_onehot, dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        )
        dice_loss = dice_loss.mean()

        return ce_loss + dice_loss


class SegmentationTrainer(BaseTrainer):
    def __init__(self, model, config, gpu_transforms=None, norm_transforms=None):
        super().__init__(model, config, gpu_transforms, norm_transforms)
        self.criterion = DiceCELoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        labels = batch["label"]
        return self.criterion(outputs, labels)
