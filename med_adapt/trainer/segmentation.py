"""Segmentation trainer."""

from __future__ import annotations

from typing import Optional

import torch

from ml_collections import ConfigDict

from med_adapt.utils.config import get_logger
from med_adapt.inference import sliding_window_predict
from med_adapt.trainer.template import TemplateTrainer


logger = get_logger(__name__)


class SegmentationTrainer(TemplateTrainer):
    def __init__(
        self,
        config: ConfigDict,
        model,
        gpu_augmentations,
        normalisation: Optional[torch.nn.Module] = None,
    ):
        config["loss"] = {"type": "dice_ce"}
        config["metrics"] = {
            "iou": {"type": "mean_iou", "num_classes": config.num_classes},
        }

        super().__init__(config, model, gpu_augmentations, normalisation)

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)

        outputs = self(image)
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        loss = self.criterion(logits, label)
        return loss, (logits, label)

    def test_step(self, batch, batch_idx):
        """Run test evaluation with sliding-window inference."""
        if self.normalisation is not None:
            batch = self.normalisation(batch)

        x = batch["image"]
        y = batch["label"]

        ps = getattr(self.model, "patch_size", 14)
        patch_size = ps if isinstance(ps, tuple) else (ps, ps)
        logits = sliding_window_predict(
            self.model,
            x,
            patch_size=patch_size,
            device=x.device,
            batch_size=self.config.test.batch_size,
            amp=self.config.test.amp,
        )

        loss = self.criterion(logits, y)

        loss = self.log_loss(
            loss, prefix="test", prog_bar=True, on_epoch=True, on_step=False
        )

        if for_metrics := (logits, y):
            pred, gt = for_metrics
            try:
                self.test_metrics.update(pred, gt)
            except Exception as e:
                logger.error(
                    f"Error computing test metrics {pred.shape=}, {gt.shape=}: {e}"
                )

        return loss
