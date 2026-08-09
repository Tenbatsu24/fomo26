"""Classification trainer."""

from __future__ import annotations

from typing import Optional

import torch

from ml_collections import ConfigDict

from med_adapt.trainer.template import TemplateTrainer


class ClassificationTrainer(TemplateTrainer):
    def __init__(
        self,
        config: ConfigDict,
        model,
        gpu_augmentations,
        normalisation: Optional[torch.nn.Module] = None,
    ):
        config["loss"] = {"type": "cross_entropy"}
        config["metrics"] = {
            "acc": {"type": "accuracy", "num_classes": config.num_classes}
        }
        super().__init__(config, model, gpu_augmentations, normalisation)

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

        return total_loss, (logits, label)
