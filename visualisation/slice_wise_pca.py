"""3-D volume → per-slice PCA visualisation through a 2-D ViT encoder.

For every depth slice of a medical volume we run the slice through the
loaded 2-D ViT, collect the patch tokens, and reduce them to three
visualisable components.  Two panels are produced per slice:

  1. **PCA-RGB** — first 3 PCA components (whitened) reshaped as an
     RGB image whose spatial layout matches the patch grid.
  2. **CLS cosine** — cosine similarity of every patch token with the
     cls token, displayed as a heatmap over the same patch grid.

Up to nine slices (3 × 3 grid, sampled uniformly from top to bottom)
are shown.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = Path(__file__).resolve().parents[1] / "data"
CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "small" / "neco" / "encoder_teacher.ckpt"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

MAX_SLICES = 9  # 3×3 grid


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_volume(volume: torch.Tensor) -> torch.Tensor:
    """Rescale each channel to [0,1] then normalise per depth slice.

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

    # Duplicate or resample channels to 3
    if C == 3:
        vol = volume
    elif C == 1:
        vol = volume.expand(3, H, W, D)
    else:
        # Resample via linear interpolation in (H,W) for each depth slice
        vol = torch.zeros(3, H, W, D, device=DEVICE, dtype=torch.float32)
        for c in range(3):
            if c < C:
                vol[c] = volume[c]
            else:
                # Bilinear resample from existing channels
                src = volume[:C].unsqueeze(0)  # [1, C, H, W] per slice loop
                # Interpolate each depth slice
                for d in range(D):
                    slice_2d = src[:, :, :, d]  # [1, C, H, W]
                    resized = F.interpolate(
                        slice_2d, size=(H, W), mode="bilinear", align_corners=False
                    )
                    vol[c, :, :, d] = resized[0, c % C]
        # Actually simpler: just repeat/interpolate the whole volume
        vol = _resample_channels(volume, 3)

    # Rescale each of the 3 channels to [0, 1]
    vol = vol.reshape(3, H * W, D)
    ch_min = vol.min(dim=1, keepdim=True).values
    ch_max = vol.max(dim=1, keepdim=True).values
    denom = ch_max - ch_min
    denom[denom == 0] = 1.0
    vol = (vol - ch_min) / denom
    vol = vol.reshape(3, H, W, D)

    # Normalise per depth slice using ImageNet stats
    vol = vol.permute(0, 3, 1, 2)  # [C, D, H, W]
    vol = vol.reshape(3 * D, H, W)
    # Repeat ImageNet stats across depth slices: [3*D, 1, 1]
    mean_rep = IMAGENET_MEAN.repeat(D, 1, 1)[:, 0, 0]  # [3*D]
    std_rep = IMAGENET_STD.repeat(D, 1, 1)[:, 0, 0]  # [3*D]
    vol = (vol - mean_rep[:, None, None]) / std_rep[:, None, None]
    vol = vol.reshape(3, D, H, W).permute(0, 2, 3, 1)  # back to [C, H, W, D]

    return vol


def _resample_channels(volume: torch.Tensor, target_c: int) -> torch.Tensor:
    """Bilinearly resample a [C, H, W, D] volume to *target_c* channels."""
    C, H, W, D = volume.shape
    out = torch.zeros(target_c, H, W, D, device=DEVICE, dtype=torch.float32)
    src = volume.unsqueeze(0)  # [1, C, H, W, D]
    for d in range(D):
        slice_2d = src[0, :, :, :, d]  # [C, H, W]
        # Treat existing channels as a multi-channel image and resample
        if C >= target_c:
            out[:, :, :, d] = slice_2d[:target_c]
        else:
            slice_ch = slice_2d.permute(1, 2, 0)  # [H, W, C]
            slice_ch = slice_ch.unsqueeze(0)  # [1, H, W, C]
            # Interpolate the last dim by treating it as spatial
            resized_ch = F.interpolate(
                slice_ch.permute(0, 3, 1, 2),  # [1, C, H, W]
                size=(target_c, H, W),
                mode="trilinear",
                align_corners=False,
            )  # [1, target_c, H, W]
            out[:, :, :, d] = resized_ch[0]
    return out


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model() -> torch.nn.Module:
    """Build and load the 2-D ViT-S encoder from the checkpoint."""
    from med_adapt.models.base.vitv2 import vitv2_small

    model = vitv2_small(img_size=518, patch_size=14)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    filtered = {k: v for k, v in ckpt.items() if not k.startswith("projection_head")}
    model.load_state_dict(filtered, strict=False)
    model = model.to(DEVICE).eval()
    return model


def get_patch_tokens(model: torch.nn.Module, slice_2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Run a single 2-D slice through *model* and return patch + cls tokens.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded ViT encoder.
    slice_2d : torch.Tensor
        Single depth slice, shape [3, H, W], already on *DEVICE*.

    Returns
    -------
    patch_tokens : torch.Tensor  shape [1, N, D]
    cls_token : torch.Tensor  shape [1, 1, D]
    grid_h : int
    grid_w : int
    """
    slice_2d = slice_2d.unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        out = model(slice_2d)
    patch_tokens = out["patch_latent"]  # [1, N, D]
    cls_token = out["latent"].unsqueeze(1)  # [1, 1, D]
    H, W = slice_2d.shape[2], slice_2d.shape[3]
    grid_h = H // model.patch_size
    grid_w = W // model.patch_size
    return patch_tokens, cls_token, grid_h, grid_w


# ---------------------------------------------------------------------------
# PCA → RGB conversion
# ---------------------------------------------------------------------------

def pca_to_rgb(patch_tokens: torch.Tensor, n_components: int = 3, whiten: bool = True) -> np.ndarray:
    """Reduce patch tokens to *n_components* PCA axes and scale to [0, 1].

    Parameters
    ----------
    patch_tokens : torch.Tensor  shape [1, N, D]
    n_components : int
    whiten : bool
        If True, divide by the square root of each eigenvalue.

    Returns
    -------
    np.ndarray  shape [N, n_components] with values in [0, 1]
    """
    flat = patch_tokens.squeeze(0).cpu().float()  # [N, D]
    mean = flat.mean(dim=0)
    centered = flat - mean
    # Covariance-based PCA via SVD
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    components = U[:, :n_components] * S[:n_components]  # [N, n_components]
    if whiten:
        components = components / (S[:n_components] + 1e-8)
    # Scale to [0, 1] per component
    for c in range(n_components):
        cmin = components[:, c].min()
        cmax = components[:, c].max()
        denom = cmax - cmin
        denom = denom if denom > 0 else 1.0
        components[:, c] = (components[:, c] - cmin) / denom
    return components.cpu().numpy()


# ---------------------------------------------------------------------------
# Cosine-similarity heatmap
# ---------------------------------------------------------------------------

def cls_cosine_heatmap(
    patch_tokens: torch.Tensor, cls_token: torch.Tensor
) -> np.ndarray:
    """Cosine similarity between every patch token and the cls token.

    Returns
    -------
    np.ndarray  shape [1, N] with values in [-1, 1]
    """
    sim = F.cosine_similarity(patch_tokens, cls_token, dim=-1)  # [1, N]
    return sim.cpu().numpy()


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def plot_volume_pca(
    dataset_name: str = "CLS002_FOMO26_Infarct",
    output_dir: Path | str = OUTPUT_DIR,
    max_slices: int = MAX_SLICES,
    sample_index: int = 0,
) -> Path:
    """Run the full 3-D → per-slice PCA pipeline and save figures.

    Parameters
    ----------
    dataset_name : str
        Registered dataset name (must be available via the registry).
    output_dir : str | Path
        Directory in which to save the output PNGs.
    max_slices : int
        Maximum number of depth slices to visualise (displayed as a
        ⌈sqrt(max_slices)⌉ × ⌈sqrt(max_slices)⌉ grid).
    sample_index : int
        Which sample from the dataset to visualise.

    Returns
    -------
    Path to the saved composite figure.
    """
    from med_adapt.registry import STORE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset --------------------------------------------------------
    dataset_cls = STORE.get("datasets", dataset_name)
    dataset = dataset_cls(root=DATASET_ROOT, fold=None, seed=None, n_splits=5)
    sample = dataset[sample_index]
    volume = sample["image"]  # [C, H, W, D]
    print(f"Volume shape: {volume.shape}, label: {sample['label'].item()}, subject: {sample['subject']}")

    # --- Preprocess ----------------------------------------------------------
    vol = preprocess_volume(volume)  # [3, H, W, D]
    C, H, W, D = vol.shape
    print(f"Preprocessed volume: {vol.shape}, depth slices: {D}")

    # --- Select depth slices -------------------------------------------------
    if D <= max_slices:
        slice_indices = list(range(D))
    else:
        slice_indices = np.linspace(0, D - 1, max_slices, dtype=int).tolist()
    print(f"Selected depth slices: {slice_indices}")

    # --- Load model ----------------------------------------------------------
    model = load_model()
    print(f"Model loaded, patch_size={model.patch_size}")

    # --- Process each slice --------------------------------------------------
    pca_images: list[np.ndarray] = []
    cosine_images: list[np.ndarray] = []
    grid_dims: list[tuple[int, int]] = []
    depths: list[int] = []

    for d in slice_indices:
        slice_2d = vol[:, :, :, d]  # [3, H, W]
        patch_tokens, cls_token, gh, gw = get_patch_tokens(model, slice_2d)
        pca_rgb = pca_to_rgb(patch_tokens, n_components=3, whiten=True)
        pca_rgb = pca_rgb.reshape(gh, gw, 3)
        cos_sim = cls_cosine_heatmap(patch_tokens, cls_token).reshape(gh, gw)

        pca_images.append(pca_rgb)
        cosine_images.append(cos_sim)
        grid_dims.append((gh, gw))
        depths.append(d)

    # --- Build composite figure (PCA-RGB panel) -----------------------------
    n = len(pca_images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig_pca, axes_pca = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5), dpi=150)
    axes_pca = np.asarray(axes_pca).reshape(-1)
    fig_pca.suptitle(
        "ViT Patch-Token PCA (whitened) — RGB view per depth slice",
        fontsize=13, fontweight="bold", y=0.98,
    )
    for i, (ax, pca_img, depth) in enumerate(zip(axes_pca, pca_images, depths)):
        ax.imshow(pca_img, vmin=0, vmax=1)
        ax.set_title(f"depth z={depth}  ({pca_img.shape[0]}×{pca_img.shape[1]} patches)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes_pca)):
        axes_pca[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pca_path = output_dir / "volume_pca_rgb.png"
    fig_pca.savefig(pca_path, bbox_inches="tight")
    plt.close(fig_pca)
    print(f"Saved → {pca_path}")

    # --- Build composite figure (cosine-sim panel) --------------------------
    fig_cos, axes_cos = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5), dpi=150)
    axes_cos = np.asarray(axes_cos).reshape(-1)
    fig_cos.suptitle(
        "Patch-token ↔ CLS cosine similarity per depth slice",
        fontsize=13, fontweight="bold", y=0.98,
    )
    for i, (ax, cos_img, depth) in enumerate(zip(axes_cos, cosine_images, depths)):
        im = ax.imshow(cos_img, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"depth z={depth}  ({cos_img.shape[0]}×{cos_img.shape[1]} patches)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label="cosine sim")
    for j in range(n, len(axes_cos)):
        axes_cos[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    cos_path = output_dir / "volume_pca_cosine.png"
    fig_cos.savefig(cos_path, bbox_inches="tight")
    plt.close(fig_cos)
    print(f"Saved → {cos_path}")

    return pca_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_volume_pca()
