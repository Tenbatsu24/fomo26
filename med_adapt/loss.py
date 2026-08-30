"""Loss function registry.

Provides ``get_loss(name)`` to instantiate losses by name.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        "bce": nn.BCEWithLogitsLoss,
        "huber": nn.HuberLoss,
        "mse": nn.MSELoss,
        "dice_ce": DiceCELoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}. Available: {list(losses)}")
    return losses[name](**params)


class MemoryEfficientSoftDiceLoss(nn.Module):
    """
    nnU-Net style MemoryEfficientSoftDiceLoss.

    Parameters
    ----------
    smooth : float
        Smoothing constant.
    do_bg : bool
        Include background class in Dice.
    batch_dice : bool
        Compute Dice across the whole batch.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        do_bg: bool = False,
        batch_dice: bool = True,
    ):
        super().__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_dice = batch_dice

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [B, C, H, W, D]
        target: [B, 1, H, W, D] or [B, H, W, D]
        """

        if target.ndim == logits.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]

        target = target.long()

        probs = F.softmax(logits, dim=1)

        target_oh = (
            F.one_hot(
                target,
                num_classes=logits.shape[1],
            )
            .movedim(-1, 1)
            .float()
        )

        if self.batch_dice:
            dims = (0,) + tuple(range(2, probs.ndim))
        else:
            dims = tuple(range(2, probs.ndim))

        intersection = torch.sum(
            probs * target_oh,
            dim=dims,
        )

        pred_sum = torch.sum(
            probs,
            dim=dims,
        )

        gt_sum = torch.sum(
            target_oh,
            dim=dims,
        )

        dice = (2.0 * intersection + self.smooth) / (pred_sum + gt_sum + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dice = dice[1:]
            else:
                dice = dice[:, 1:]

        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1e-5,
        do_bg: bool = False,
        batch_dice: bool = True,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = MemoryEfficientSoftDiceLoss(
            smooth=smooth,
            do_bg=do_bg,
            batch_dice=batch_dice,
        )

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        if target.ndim == logits.ndim:
            target_ce = target[:, 0].long()
        else:
            target_ce = target.long()

        ce_loss = self.ce(logits, target_ce)
        dice_loss = self.dice(logits, target)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
