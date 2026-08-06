"""Loss function registry.

Provides ``get_loss(name)`` to instantiate losses by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def get_loss(name: str, **params):
    """Return an instantiated loss function.

    Args:
        name: Loss identifier.
        **params: Keyword arguments passed to the loss constructor.

    Returns:
        An :class:`torch.nn.Module` loss.
    """
    losses = {
        "cross_entropy": nn.CrossEntropyLoss,
        "huber": nn.HuberLoss,
        "mse": nn.MSELoss,
        "dice_ce": DiceCELoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}. Available: {list(losses)}")
    return losses[name](**params)


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for segmentation."""

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        if target.ndim == logits.ndim:
            target = target.squeeze(1)

        ce_loss = self.ce(logits, target)

        num_classes = logits.shape[1]
        probs = nn.functional.softmax(logits, dim=1)
        target_onehot = nn.functional.one_hot(target, num_classes=num_classes)
        target_onehot = target_onehot.permute(
            0, -1, *range(1, target_onehot.ndim - 1)
        ).float()

        dims = (0,) + tuple(range(2, probs.ndim))
        intersection = torch.sum(probs * target_onehot, dim=dims)
        cardinality = torch.sum(probs + target_onehot, dim=dims)
        dice_loss = 1.0 - (
            (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        )
        dice_loss = dice_loss.mean()

        return ce_loss + dice_loss
