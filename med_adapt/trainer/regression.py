"""Regression trainer."""

from __future__ import annotations

from typing import Any, Optional

import torch

from med_adapt.trainer.template import TemplateTrainer


class RegressionTrainer(TemplateTrainer):
    def __init__(
        self,
        config: dict[str, Any],
        model,
        normalisation: Optional[torch.nn.Module] = None,
    ):
        config["loss"] = {"type": "huber"}
        config["metrics"] = {"l2": {"type": "rmse"}}
        super().__init__(config=config, model=model, normalisation=normalisation)

    def batch_to_loss(self, batch, train=False):
        x = batch["image"]
        y = batch["label"].float()
        if self.normalisation is not None:
            x = self.normalisation(x)
        if train and self.gpu_aug is not None:
            x = self.gpu_aug(x)
        outputs = self.model(x)
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        loss = self.criterion(logits, y)
        return loss, (logits, y)
