"""Segmentation trainer with deep supervision."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

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
            "dice": {"type": "mean_dice", "num_classes": config.num_classes},
        }
        super().__init__(config, model, gpu_augmentations, normalisation)

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)
        outputs = self(image)

        # Deep supervision with list outputs (2D/3D adaptation models)
        num_preds = len(outputs)

        # Weighted intermediate presence losses
        all_losses = []
        all_weights = []
        for i, seg_pred in enumerate(outputs):
            dice_ce_loss = self.criterion(seg_pred, label)
            weight = 2 ** (i - (num_preds - 1))

            all_losses.append(dice_ce_loss)
            all_weights.append(weight)

        return (
            {
                "loss": sum([l * w for l, w in zip(all_losses, all_weights)]),
                **{
                    f"loss_{num_preds-i}": i_loss for i, i_loss in enumerate(all_losses)
                },
            },
            (outputs[-1], label),
        )

    def test_step(self, batch, batch_idx):
        """Run test evaluation.

        Sliding-window inference is kept in this method but disabled by
        default (see ``config.test.sliding_window``). When enabled it
        overrides the standard ``batch_to_loss`` forward pass.
        """
        use_sliding_window = self.config.test.get("sliding_window", False)

        if use_sliding_window:
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

        # Fall back to the same path as validation_step
        loss, for_metrics = self.batch_to_loss(batch, train=False)
        loss = self.log_loss(
            loss, prefix="test", prog_bar=True, on_epoch=True, on_step=False
        )
        if for_metrics:
            pred, gt = for_metrics
            try:
                self.test_metrics.update(pred, gt)
            except Exception as e:
                logger.error(f"Error computing test metrics: {e}")
        return loss
