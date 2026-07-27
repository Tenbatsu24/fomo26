import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.segmentation import MeanIoU, DiceScore

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

        num_classes = config["num_classes"]

        self.train_dice = DiceScore(
            num_classes=num_classes,
            average="macro",
        )

        self.val_dice = DiceScore(
            num_classes=num_classes,
            average="macro",
        )

        self.val_iou = MeanIoU(
            num_classes=num_classes,
        )

    def on_train_epoch_start(self) -> None:
        self.train_dice.reset()
        self.val_dice.reset()
        self.val_iou.reset()

    def log_train_metrics(self, outputs, batch):
        preds = outputs.argmax(dim=1)
        target = batch["target"]

        if target.ndim == preds.ndim + 1:
            target = target.squeeze(1)

        self.train_dice(preds, target)

        self.log(
            "train/dice",
            self.train_dice,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

    def log_val_metrics(self, outputs, batch):
        preds = outputs.argmax(dim=1)
        target = batch["target"]

        if target.ndim == preds.ndim + 1:
            target = target.squeeze(1)

        self.val_dice(preds, target)
        self.val_iou(preds, target)

        self.log(
            "val/dice",
            self.val_dice,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/iou",
            self.val_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, batch):
        labels = batch["target"]
        return self.criterion(outputs, labels)
