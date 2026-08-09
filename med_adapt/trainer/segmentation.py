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
            "iou": {"type": "mean_iou", "num_classes": config.num_classes},
        }
        super().__init__(config, model, gpu_augmentations, normalisation)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def batch_to_loss(self, batch, train=False):
        image, label = self.preprocess_batch(batch, train)
        outputs = self(image)

        final_seg_logits = outputs["seg_logits"]  # (B, C, H, W, D)
        final_presence_logits = outputs["presence_logits"]  # (B, Q)
        intermediate = outputs.get(
            "intermediate", []
        )  # list of (mask_logits, presence_logits)

        # DiceCE on final segmentation
        dice_ce_loss = self.criterion(final_seg_logits, label)

        # Per-volume presence GT: 1 if class i appears anywhere in volume
        gt = label[:, 0].long() if label.ndim == 5 else label.long()
        presence_gt = torch.stack(
            [
                (gt == i).any(dim=(1, 2, 3)).float()
                for i in range(1, self.num_classes + 1)
            ],
            dim=1,
        )  # (B, num_classes)

        # BCE on final presence
        final_bce = self.bce_loss(final_presence_logits, presence_gt)

        # Deep supervision: BCE on intermediate presence logits
        inter_bce = torch.tensor(0.0, device=label.device)
        for _, pres_logits in intermediate:
            inter_bce += self.bce_loss(pres_logits, presence_gt)
        inter_bce /= max(len(intermediate), 1)

        loss = dice_ce_loss + final_bce + 0.5 * inter_bce
        return (
            {
                "loss": loss,
                "dice_ce": dice_ce_loss,
                "bce_final": final_bce,
                "bce_inter": inter_bce,
            },
            (final_seg_logits, label),
        )

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
