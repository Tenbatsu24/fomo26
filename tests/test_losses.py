"""Tests for loss functions used in training.

Verifies that each loss computes correctly with known inputs and expected
outputs, and that shapes match what the trainers expect:
- classification: logits [B, C], target [B]
- regression:     logits [B, C], target [B, C]
- segmentation:   logits [B, C, H, D, W], target [B, 1, H, D, W] or [B, H, D, W]

Losses are imported from med_adapt.loss and are not re-implemented.
"""

from __future__ import annotations

import pytest
import torch

from med_adapt.loss import DiceCELoss, MemoryEfficientSoftDiceLoss, get_loss

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_device(t: torch.Tensor) -> torch.Tensor:
    return t.to(_device())


# ---------------------------------------------------------------------------
# cross_entropy  (classification)
# ---------------------------------------------------------------------------


class TestCrossEntropy:
    """nn.CrossEntropyLoss via get_loss('cross_entropy')."""

    def test_known_classes_gives_expected_loss(self):
        """All samples in batch are class 0 → loss should be -log(softmax[0])."""
        device = _device()
        loss_fn = get_loss("cross_entropy").to(device)

        # 4 samples, 3 classes. Logits deliberately set so softmax is known:
        # logits = [[0, 0, 0], ...]  →  softmax = [1/3, 1/3, 1/3]
        logits = torch.zeros(4, 3, device=device)
        target = torch.zeros(4, dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        expected = -torch.log(torch.tensor(1.0 / 3.0, device=device))
        assert torch.isclose(
            loss, expected, atol=1e-6
        ), f"got {loss.item():.6f}, expected {expected.item():.6f}"

    def test_perfect_prediction_gives_near_zero_loss(self):
        """One-hot-like logits with correct class → loss ≈ 0."""
        device = _device()
        loss_fn = get_loss("cross_entropy").to(device)

        logits = torch.tensor([[10.0, -1.0, -1.0], [-1.0, 10.0, -1.0]], device=device)
        target = torch.tensor([0, 1], dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss < 1e-4, f"Expected near-zero loss, got {loss.item():.6f}"

    def test_shape_B_C_and_B(self):
        """Correct input shapes: logits [B, C], target [B]."""
        device = _device()
        loss_fn = get_loss("cross_entropy").to(device)
        logits = torch.randn(8, 5, device=device)
        target = torch.randint(0, 5, (8,), dtype=torch.long, device=device)
        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([]), "Loss should be a scalar"

    def test_deep_supervision_list(self):
        """Trainer passes a list of logits; each element is [B, C]."""
        device = _device()
        loss_fn = get_loss("cross_entropy").to(device)

        preds = [
            torch.randn(4, 3, device=device),
            torch.randn(4, 3, device=device),
        ]
        target = torch.zeros(4, dtype=torch.long, device=device)

        total = sum(loss_fn(p, target) for p in preds)
        assert total.shape == torch.Size([])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_compatibility(self):
        device = "cuda"
        loss_fn = get_loss("cross_entropy").to(device)
        logits = torch.randn(2, 3, device=device)
        target = torch.tensor([0, 2], dtype=torch.long, device=device)
        loss = loss_fn(logits, target)
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# huber  (regression)
# ---------------------------------------------------------------------------


class TestHuber:
    """nn.HuberLoss via get_loss('huber')."""

    def test_perfect_prediction_gives_zero_loss(self):
        """pred == target → Huber loss = 0."""
        device = _device()
        loss_fn = get_loss("huber").to(device)

        pred = torch.tensor([[1.0, 2.0, 3.0]], device=device)
        target = pred.clone()

        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.zeros(1, device=device), atol=1e-6)

    def test_known_value(self):
        """Single element, delta=1.0: |error|=2 > delta, so Huber = delta*(|e|-0.5*delta) = 1.5."""
        device = _device()
        loss_fn = get_loss("huber", delta=1.0).to(device)

        pred = torch.tensor([[2.0]], device=device)
        target = torch.tensor([[0.0]], device=device)

        loss = loss_fn(pred, target)
        # |error| = 2.0 > delta=1.0 → linear regime: delta * (|error| - 0.5*delta)
        expected = torch.tensor(1.5, device=device)
        assert torch.isclose(loss, expected, atol=1e-6), f"got {loss.item():.6f}"

    def test_shape_B_C(self):
        """Correct input shapes: pred [B, C], target [B, C]."""
        device = _device()
        loss_fn = get_loss("huber").to(device)
        pred = torch.randn(8, 4, device=device)
        target = torch.randn(8, 4, device=device)
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([])

    def test_deep_supervision_list(self):
        """Regression trainer averages over per-class predictions in a list."""
        device = _device()
        loss_fn = get_loss("huber").to(device)

        preds = [
            torch.tensor([[1.0, 2.0]], device=device),
            torch.tensor([[1.5, 2.5]], device=device),
        ]
        target = torch.tensor([[1.0, 2.0]], device=device)

        total = sum(loss_fn(p, target) for p in preds) / len(preds)
        assert total.shape == torch.Size([])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_compatibility(self):
        device = "cuda"
        loss_fn = get_loss("huber").to(device)
        pred = torch.randn(2, 3, device=device)
        target = torch.randn(2, 3, device=device)
        loss = loss_fn(pred, target)
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# mse  (regression)
# ---------------------------------------------------------------------------


class TestMSE:
    """nn.MSELoss via get_loss('mse')."""

    def test_perfect_prediction_gives_zero_loss(self):
        device = _device()
        loss_fn = get_loss("mse").to(device)

        pred = torch.tensor([[3.0, 4.0]], device=device)
        target = pred.clone()

        loss = loss_fn(pred, target)
        assert torch.isclose(loss, torch.zeros(1, device=device), atol=1e-6)

    def test_known_value(self):
        """MSE([2], [0]) = (2-0)^2 = 4.0."""
        device = _device()
        loss_fn = get_loss("mse").to(device)

        pred = torch.tensor([[2.0]], device=device)
        target = torch.tensor([[0.0]], device=device)

        loss = loss_fn(pred, target)
        expected = torch.tensor(4.0, device=device)
        assert torch.isclose(loss, expected, atol=1e-6), f"got {loss.item():.6f}"

    def test_shape_B_C(self):
        device = _device()
        loss_fn = get_loss("mse").to(device)
        pred = torch.randn(8, 4, device=device)
        target = torch.randn(8, 4, device=device)
        loss = loss_fn(pred, target)
        assert loss.shape == torch.Size([])

    def test_deep_supervision_list(self):
        device = _device()
        loss_fn = get_loss("mse").to(device)

        preds = [
            torch.ones(2, 3, device=device) * 1.0,
            torch.ones(2, 3, device=device) * 2.0,
        ]
        target = torch.ones(2, 3, device=device)

        total = sum(loss_fn(p, target) for p in preds) / len(preds)
        assert total.shape == torch.Size([])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_compatibility(self):
        device = "cuda"
        loss_fn = get_loss("mse").to(device)
        pred = torch.randn(2, 3, device=device)
        target = torch.randn(2, 3, device=device)
        loss = loss_fn(pred, target)
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# MemoryEfficientSoftDiceLoss
# ---------------------------------------------------------------------------


class TestMemoryEfficientSoftDiceLoss:
    """Tests for the nnU-Net style Soft Dice loss."""

    def test_perfect_segmentation_gives_zero_loss(self):
        """Identical pred and target masks → Dice = 1 → loss = 0."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss().to(device)

        # 2 classes, perfect one-hot mask
        logits = torch.zeros(1, 2, 4, 4, 4, device=device)
        logits[:, 0] = 10.0  # background dominant
        target = torch.zeros(1, 1, 4, 4, 4, dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        # Softmax of [10, 0] ≈ [1, 0] but not exact; Dice ≈ 0.997 → loss ≈ 0.003
        assert loss < 1e-2, f"Expected near-zero loss, got {loss.item():.6f}"

    def test_complete_disjoint_classes(self):
        """Left half class 0, right half class 1 → known Dice per class."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss(smooth=1.0, batch_dice=False).to(device)

        B, C, H, D, W = 1, 2, 4, 4, 4
        logits = torch.zeros(B, C, H, D, W, device=device)
        # Class 0: left half, Class 1: right half (perfect separation)
        logits[:, 0, :, :, : W // 2] = 10.0
        logits[:, 1, :, :, W // 2 :] = 10.0

        # Target: left half class 0, right half class 1
        target = torch.zeros(B, H, D, W, dtype=torch.long, device=device)
        target[:, :, :, : W // 2] = 0
        target[:, :, :, W // 2 :] = 1

        loss = loss_fn(logits, target)
        # With perfect segmentation, each class Dice ≈ 1, so loss ≈ 0
        assert (
            loss < 1e-4
        ), f"Expected near-zero loss for perfect seg, got {loss.item():.6f}"

    def test_target_with_channel_dim(self):
        """Target [B, 1, H, D, W] should work (trainer passes this shape)."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss().to(device)

        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 1, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    def test_target_without_channel_dim(self):
        """Target [B, H, D, W] should also work."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss().to(device)

        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    def test_no_bg_excludes_background(self):
        """do_bg=False should return loss for foreground classes only."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss(do_bg=False, batch_dice=True).to(device)

        logits = torch.randn(2, 3, 4, 4, 4, device=device)
        target = torch.randint(0, 3, (2, 4, 4, 4), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        # With 3 classes and do_bg=False, batch_dice=True → scalar (mean over 2 fg classes)
        assert loss.shape == torch.Size([])

    def test_batch_dice_true(self):
        """batch_dice=True computes Dice across the whole batch."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss(batch_dice=True).to(device)

        logits = torch.randn(4, 2, 8, 8, 8, device=device)
        target = torch.randint(0, 2, (4, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    def test_batch_dice_false(self):
        """batch_dice=False computes per-sample Dice."""
        device = _device()
        loss_fn = MemoryEfficientSoftDiceLoss(batch_dice=False).to(device)

        logits = torch.randn(4, 2, 8, 8, 8, device=device)
        target = torch.randint(0, 2, (4, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_compatibility(self):
        device = "cuda"
        loss_fn = MemoryEfficientSoftDiceLoss().to(device)
        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 8, 8, 8), dtype=torch.long, device=device)
        loss = loss_fn(logits, target)
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# DiceCELoss
# ---------------------------------------------------------------------------


class TestDiceCELoss:
    """Tests for the combined Dice + CrossEntropy loss."""

    def test_perfect_segmentation_gives_ce_zero_plus_dice_zero(self):
        """Perfect mask → CE ≈ 0 and Dice ≈ 0, so total ≈ 0."""
        device = _device()
        loss_fn = DiceCELoss().to(device)

        B, C, H, D, W = 1, 2, 4, 4, 4
        logits = torch.zeros(B, C, H, D, W, device=device)
        # Perfect segmentation: left half class 0, right half class 1
        logits[:, 0, :, :, : W // 2] = 10.0
        logits[:, 1, :, :, W // 2 :] = 10.0
        target = torch.zeros(B, 1, H, D, W, dtype=torch.long, device=device)
        # Explicit 5-D slicing so the last dimension (W) is what gets split
        target[:, :, :, :, : W // 2] = 0
        target[:, :, :, :, W // 2 :] = 1

        loss = loss_fn(logits, target)
        assert loss < 1e-3, f"Expected near-zero loss, got {loss.item():.6f}"

    def test_known_ce_component(self):
        """With a single class and uniform logits, CE is known; verify magnitude."""
        device = _device()
        loss_fn = DiceCELoss(ce_weight=1.0, dice_weight=0.0).to(device)

        # 2 samples, 3 classes, logits = 0 → softmax uniform → CE = log(3)
        logits = torch.zeros(2, 3, device=device)
        target = torch.zeros(2, dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        expected_ce = torch.log(torch.tensor(3.0, device=device))
        assert torch.isclose(loss, expected_ce, atol=1e-5), f"got {loss.item():.6f}"

    def test_target_with_channel_dim(self):
        """Target [B, 1, H, D, W] is accepted by the trainer."""
        device = _device()
        loss_fn = DiceCELoss().to(device)

        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 1, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    def test_target_without_channel_dim(self):
        """Target [B, H, D, W] is also accepted."""
        device = _device()
        loss_fn = DiceCELoss().to(device)

        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 8, 8, 8), dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        assert loss.shape == torch.Size([])

    def test_weighted_combination(self):
        """ce_weight and dice_weight control the contribution of each term."""
        device = _device()
        # Use dice_weight=0 to isolate CE component
        loss_fn = DiceCELoss(ce_weight=2.0, dice_weight=0.0).to(device)

        logits = torch.zeros(1, 2, 4, 4, 4, device=device)
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long, device=device)

        loss = loss_fn(logits, target)
        # CE with uniform logits = log(2); weighted by 2.0 → 2*log(2)
        expected = 2.0 * torch.log(torch.tensor(2.0, device=device))
        assert torch.isclose(loss, expected, atol=1e-5), f"got {loss.item():.6f}"

    def test_deep_supervision_list(self):
        """Segmentation trainer passes a list of logits, each [B, C, H, D, W]."""
        device = _device()
        loss_fn = DiceCELoss().to(device)

        preds = [
            torch.randn(2, 3, 8, 8, 8, device=device),
            torch.randn(2, 3, 8, 8, 8, device=device),
        ]
        target = torch.randint(0, 3, (2, 1, 8, 8, 8), dtype=torch.long, device=device)

        losses = [loss_fn(p, target) for p in preds]
        assert all(l.shape == torch.Size([]) for l in losses)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_cuda_compatibility(self):
        device = "cuda"
        loss_fn = DiceCELoss().to(device)
        logits = torch.randn(2, 3, 8, 8, 8, device=device)
        target = torch.randint(0, 3, (2, 8, 8, 8), dtype=torch.long, device=device)
        loss = loss_fn(logits, target)
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# get_loss registry
# ---------------------------------------------------------------------------


class TestGetLoss:
    """Tests for the get_loss factory."""

    def test_known_names(self):
        for name in ("cross_entropy", "huber", "mse", "dice_ce"):
            loss = get_loss(name)
            assert isinstance(loss, torch.nn.Module), f"{name} did not return a Module"

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown loss"):
            get_loss("nonexistent_loss")

    def test_params_passed_through(self):
        loss = get_loss("huber", delta=0.5)
        assert loss.delta == 0.5, "Delta parameter not passed through"
