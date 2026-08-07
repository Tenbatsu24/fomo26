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
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        loss = self.criterion(logits, label)
        return loss, (logits, label)
