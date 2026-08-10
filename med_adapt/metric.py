"""Metric registry.

Provides ``get_metric(name)`` to instantiate torchmetrics by name.
"""

from __future__ import annotations

import torch

import torch.nn.functional as F

from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC


class DiceIoUMetric(Metric):
    """
    Sample-wise Dice and IoU metric.

    Predictions:
        [B, C, H, W, D]

    Targets:
        [B, 1, H, W, D]

    Notes
    -----
    - Uses argmax on predictions.
    - Computes TP/FP/FN per sample and class.
    - Classes absent in the GT for a sample are ignored
      (set to NaN before averaging), similar to nnU-Net.
    - Averages are sample-wise, not global TP/FP/FN aggregation.
    """

    full_state_update = False
    higher_is_better = True

    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        eps: float = 1e-8,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

        self.add_state(
            "tp",
            default=[],
            dist_reduce_fx="cat",
        )

        self.add_state(
            "fp",
            default=[],
            dist_reduce_fx="cat",
        )

        self.add_state(
            "fn",
            default=[],
            dist_reduce_fx="cat",
        )

        self.add_state(
            "gt_pixels",
            default=[],
            dist_reduce_fx="cat",
        )

    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        preds : torch.Tensor
            Shape [B, C, H, W, D]
        target : torch.Tensor
            Shape [B, 1, H, W, D]
        """

        pred_labels = preds.argmax(dim=1)
        target = target[:, 0].long()

        pred_oh = (
            F.one_hot(
                pred_labels,
                num_classes=self.num_classes,
            )
            .movedim(-1, 1)
            .bool()
        )

        target_oh = (
            F.one_hot(
                target,
                num_classes=self.num_classes,
            )
            .movedim(-1, 1)
            .bool()
        )

        spatial_dims = tuple(range(2, pred_oh.ndim))

        tp = (pred_oh & target_oh).sum(dim=spatial_dims)
        fp = (pred_oh & ~target_oh).sum(dim=spatial_dims)
        fn = (~pred_oh & target_oh).sum(dim=spatial_dims)

        gt_pixels = target_oh.sum(dim=spatial_dims)

        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.gt_pixels.append(gt_pixels)

    def compute(self):

        tp = torch.cat(self.tp, dim=0).float()
        fp = torch.cat(self.fp, dim=0).float()
        fn = torch.cat(self.fn, dim=0).float()

        gt_pixels = torch.cat(self.gt_pixels, dim=0)

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + self.eps)
        # iou = tp / (tp + fp + fn + self.eps)

        # Ignore classes absent in GT for a sample
        valid = gt_pixels > 0

        dice = dice.masked_fill(~valid, torch.nan)
        # iou = iou.masked_fill(~valid, torch.nan)

        start_idx = 0 if self.include_background else 1

        dice_fg = dice[:, start_idx:]
        # iou_fg = iou[:, start_idx:]

        return torch.nanmean(dice_fg)
        # {
        # average over samples
        # "dice_per_class": torch.nanmean(dice_fg, dim=0),
        # "iou_per_class": torch.nanmean(iou_fg, dim=0),
        # average over samples and classes
        # "mean_dice": torch.nanmean(dice_fg),
        # "mean_iou": torch.nanmean(iou_fg),
        # }


def get_metric(name: str, **params):
    """Return an instantiated metric.

    Args:
        name: Metric identifier.
        **params: Keyword arguments passed to the metric constructor.

    Returns:
        A torchmetrics :class:`Metric` instance.
    """
    metrics = {
        "accuracy": lambda **p: MulticlassAccuracy(
            **p, ignore_index=0, average="macro"
        ),
        "acc": lambda **p: MulticlassAccuracy(**p, ignore_index=0, average="macro"),
        "auroc": lambda **p: MulticlassAUROC(
            **p, ignore_index=0, average="macro", thresholds=11
        ),
        "mse": lambda **p: MeanSquaredError(squared=True, **p),
        "rmse": lambda **p: MeanSquaredError(squared=False, **p),
        "l2": lambda **p: MeanSquaredError(squared=False, **p),
        "mean_dice": lambda **p: DiceIoUMetric(**p),
    }
    if name not in metrics:
        raise ValueError(f"Unknown metric {name!r}. Available: {list(metrics)}")
    return metrics[name](**params)
