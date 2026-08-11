"""Shared constants and utilities for the visualisation package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project-root path (one level above this package)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Checkpoint & data paths
# ---------------------------------------------------------------------------
CHECKPOINT = _PROJECT_ROOT / "checkpoints" / "small" / "neco" / "encoder_teacher.ckpt"
CHECKPOINT_3D = (
    _PROJECT_ROOT / "checkpoints" / "small" / "neco_3d" / "encoder_teacher.ckpt"
)
DATASET_ROOT = _PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Canonical output directory — all plots land here, not scattered across the
# package itself.
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT_ROOT / "understand"

# ---------------------------------------------------------------------------
# Device & ImageNet normalisation
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MAX_SLICES = 9  # 3×3 grid
GRID_SIZE = 37  # 518 / 14
PATCH_SIZE = 14
IN_CHANS = 3
EMBED_DIM = 384

# ---------------------------------------------------------------------------
# Colours / palettes
# ---------------------------------------------------------------------------
PALETTE_SEQUENTIAL = "viridis"
PALETTE_DIVERGING = "coolwarm"
PALETTE_GRAY = "gray_r"

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _fig_kw(**override: object) -> dict:
    """Return default figure kwargs, allowing overrides."""
    base: dict = {"figsize": (8, 6), "dpi": 150}
    base.update(override)  # type: ignore[typeddict-item]
    return base


def _colorbar(ax, mappable, label: str = "") -> None:
    import matplotlib.pyplot as plt

    plt.colorbar(mappable, ax=ax, label=label, shrink=0.8)


# ---------------------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------------------


def preprocess_volume(volume: torch.Tensor) -> torch.Tensor:
    """Rescale channels to 3, rescale each channel to [0,1], then normalise
    per depth slice with ImageNet stats.

    Parameters
    ----------
    volume : torch.Tensor
        Input tensor of shape [C, H, W, D].

    Returns
    -------
    torch.Tensor
        Normalised volume of shape [C, H, W, D] on *DEVICE*.
    """
    volume = volume.to(DEVICE).float()
    C, H, W, D = volume.shape

    if C == 3:
        vol = volume
    elif C == 1:
        vol = volume.expand(3, H, W, D)
    else:
        vol = _resample_channels(volume, 3)

    vol = vol.reshape(3, H * W, D)
    ch_min = vol.min(dim=1, keepdim=True).values
    ch_max = vol.max(dim=1, keepdim=True).values
    denom = ch_max - ch_min
    denom[denom == 0] = 1.0
    vol = (vol - ch_min) / denom
    vol = vol.reshape(3, H, W, D)

    vol = vol.permute(0, 3, 1, 2)  # [C, D, H, W]
    vol = vol.reshape(3 * D, H, W)
    mean_rep = IMAGENET_MEAN.repeat(D, 1, 1)[:, 0, 0]
    std_rep = IMAGENET_STD.repeat(D, 1, 1)[:, 0, 0]
    vol = (vol - mean_rep[:, None, None]) / std_rep[:, None, None]
    vol = vol.reshape(3, D, H, W).permute(0, 2, 3, 1)  # back to [C, H, W, D]
    return vol


def _resample_channels(volume: torch.Tensor, target_c: int) -> torch.Tensor:
    """Bilinearly resample a [C, H, W, D] volume to *target_c* channels."""
    C, H, W, D = volume.shape
    out = torch.zeros(target_c, H, W, D, device=DEVICE, dtype=torch.float32)
    for d in range(D):
        slice_2d = volume[:, :, :, d]  # [C, H, W]
        if C >= target_c:
            out[:, :, :, d] = slice_2d[:target_c]
        else:
            slice_ch = slice_2d.permute(1, 2, 0)  # [H, W, C]
            slice_ch = slice_ch.unsqueeze(0)
            resized = F.interpolate(
                slice_ch.permute(0, 3, 1, 2),
                size=(target_c, H, W),
                mode="trilinear",
                align_corners=False,
            )
            out[:, :, :, d] = resized[0]
    return out


# ---------------------------------------------------------------------------
# PCA → RGB
# ---------------------------------------------------------------------------


def pca_to_rgb(
    patch_tokens: torch.Tensor, n_components: int = 3, whiten: bool = True
) -> np.ndarray:
    """Reduce patch tokens to *n_components* PCA axes and scale to [0, 1].

    Parameters
    ----------
    patch_tokens : torch.Tensor  shape [1, N, D]
    n_components : int
    whiten : bool
        If True, divide by the singular values (whitening).

    Returns
    -------
    np.ndarray  shape [N, n_components] with values in [0, 1]
    """
    flat = patch_tokens.squeeze(0).cpu().float()  # [N, D]
    mean = flat.mean(dim=0)
    centered = flat - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    components = U[:, :n_components] * S[:n_components]  # [N, n_components]
    if whiten:
        components = components / (S[:n_components] + 1e-8)
    for c in range(n_components):
        cmin = components[:, c].min()
        cmax = components[:, c].max()
        denom = cmax - cmin if (cmax - cmin) > 0 else 1.0
        components[:, c] = (components[:, c] - cmin) / denom
    return components.cpu().numpy()
