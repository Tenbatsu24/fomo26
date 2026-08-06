"""Metric registry.

Provides ``get_metric(name)`` to instantiate torchmetrics by name.
"""

from __future__ import annotations

from torchmetrics.regression import MeanSquaredError
from torchmetrics.segmentation import MeanIoU, DiceScore
from torchmetrics.classification import MulticlassAccuracy


def get_metric(name: str, **params):
    """Return an instantiated metric.

    Args:
        name: Metric identifier.
        **params: Keyword arguments passed to the metric constructor.

    Returns:
        A torchmetrics :class:`Metric` instance.
    """
    metrics = {
        "accuracy": lambda **p: MulticlassAccuracy(**p),
        "acc": lambda **p: MulticlassAccuracy(**p),
        "mse": lambda **p: MeanSquaredError(squared=True, **p),
        "rmse": lambda **p: MeanSquaredError(squared=False, **p),
        "l2": lambda **p: MeanSquaredError(squared=False, **p),
        "dice": lambda **p: DiceScore(average="macro", **p),
        "dice_score": lambda **p: DiceScore(average="macro", **p),
        "mean_iou": lambda **p: MeanIoU(**p),
        "iou": lambda **p: MeanIoU(**p),
    }
    if name not in metrics:
        raise ValueError(f"Unknown metric {name!r}. Available: {list(metrics)}")
    return metrics[name](**params)
