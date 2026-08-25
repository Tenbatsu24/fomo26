"""Classification trainer."""

from __future__ import annotations

import torch

from ml_collections import ConfigDict
from torchmetrics import Precision, Accuracy, Recall, F1Score, AUROC

from med_adapt.trainer.template import TemplateTrainer


class ClassificationTrainer(TemplateTrainer):
    def __init__(
        self,
        config: ConfigDict,
        model,
        gpu_augmentations,
    ):
        config["loss"] = {"type": "cross_entropy"}
        super().__init__(config, model, gpu_augmentations)

    def make_metrics(self):
        return {
            "acc": Accuracy(task="multiclass", num_classes=self.config.num_classes),
            "prec": Precision(
                task="multiclass", num_classes=self.config.num_classes, average="macro"
            ),
            "recall": Recall(
                task="multiclass", num_classes=self.config.num_classes, average="macro"
            ),
            "f1": F1Score(
                task="multiclass", num_classes=self.config.num_classes, average="macro"
            ),
            "auroc": AUROC(
                task="multiclass", num_classes=self.config.num_classes, average="macro"
            ),
        }

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)
        label = label.long()

        outputs = self(image)

        if isinstance(outputs, list):
            num_preds = len(outputs)
            total_loss = None
            for i, pred in enumerate(outputs):
                weight = 2 ** (i - (num_preds - 1))
                loss = self.criterion(pred, label)
                total_loss = loss if total_loss is None else total_loss + weight * loss
            logits = outputs[-1]
        else:
            logits = outputs
            total_loss = self.criterion(logits, label)

        return total_loss, (torch.softmax(logits, dim=1), label)
