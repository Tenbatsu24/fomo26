"""Regression trainer."""

from __future__ import annotations

from typing import Optional

import torch

from ml_collections import ConfigDict

from med_adapt.trainer.template import TemplateTrainer


class RegressionTrainer(TemplateTrainer):
    def __init__(
        self,
        config: ConfigDict,
        model,
        gpu_augmentations,
    ):
        config["loss"] = {"type": "huber"}
        config["metrics"] = {"l2": {"type": "rmse"}}
        super().__init__(config, model, gpu_augmentations)

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)
        label = label.float()

        outputs = self(image)

        if isinstance(outputs, list):
            num_preds = len(outputs)
            total_loss = None
            for i, pred in enumerate(outputs):
                weight = 2 ** (i - (num_preds - 1))
                if isinstance(pred, list):
                    pred_loss = sum(self.criterion(p, label) for p in pred) / len(pred)
                else:
                    pred_loss = self.criterion(pred, label)
                total_loss = (
                    pred_loss if total_loss is None else total_loss + weight * pred_loss
                )
            logits = outputs[-1]
        else:
            logits = (
                outputs
                if isinstance(outputs, torch.Tensor)
                else outputs.get("logits", outputs)
            )
            total_loss = self.criterion(logits, label)

        return total_loss, (logits, label)
