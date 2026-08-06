"""Segmentation trainer."""

from __future__ import annotations

from typing import Any, Optional

import torch

from med_adapt.inference import sliding_window_predict
from med_adapt.trainer.template import TemplateTrainer


class SegmentationTrainer(TemplateTrainer):
    def __init__(
        self,
        config: dict[str, Any],
        model,
        normalisation: Optional[torch.nn.Module] = None,
    ):
        config["loss"] = {"type": "dice_ce"}
        config["metrics"] = {
            "dice": {"type": "dice_score", "num_classes": config.num_classes},
            "iou": {"type": "mean_iou", "num_classes": config.num_classes},
        }
        super().__init__(config=config, model=model, normalisation=normalisation)

    def batch_to_loss(self, batch, train=False):
        x = batch["image"]
        y = batch["label"]
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

    def test_step(self, batch, batch_idx):
        """Run test evaluation with sliding-window inference."""
        x = batch["image"]
        y = batch["label"]

        if self.normalisation is not None:
            x = self.normalisation(x)

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
