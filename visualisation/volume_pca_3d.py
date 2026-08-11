"""3-D volume PCA visualisation using the 3-D ViT encoder.

Two composite figures are produced:

  1. **PCA-RGB** — for each of up to nine depth slices of the 3-D patch
     grid, the patch tokens are reduced to three whitened PCA components
     and displayed as an RGB image.
  2. **CLS cosine** — for each depth slice, the cosine similarity of
     every patch token with the cls token is shown as a heatmap.

The 3-D model processes the whole volume at once (no depth-into-batch
folding), so the patch grid is genuinely 3-D.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CHECKPOINT_3D = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "small"
    / "neco_3d"
    / "encoder_teacher.ckpt"
)
DATASET_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

MAX_SLICES = 9


# ---------------------------------------------------------------------------
# Preprocessing (same as slice_wise_pca.py)
# ---------------------------------------------------------------------------


def preprocess_volume(volume: torch.Tensor) -> torch.Tensor:
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
    vol = vol.reshape(3, D, H, W).permute(0, 2, 3, 1)
    return vol


def _resample_channels(volume: torch.Tensor, target_c: int) -> torch.Tensor:
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
# Model loading
# ---------------------------------------------------------------------------


def load_3d_model(patch_size=14) -> torch.nn.Module:
    from med_adapt.models.base.vitv2_3d import (
        vitv2_3d_small,
        load_3d_checkpoint_with_anisotropic_patch,
    )

    model = vitv2_3d_small(patch_size=patch_size).to(DEVICE).eval()
    if isinstance(patch_size, int):
        state_dict = torch.load(CHECKPOINT_3D)
        missing, unexpected = model.load_state_dict(state_dict)
    else:
        missing, unexpected = load_3d_checkpoint_with_anisotropic_patch(
            str(CHECKPOINT_3D), model
        )
    if missing:
        print(
            f"Info: missing keys (expected for pos_embed with anisotropic ps): {missing}"
        )
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")
    return model


# ---------------------------------------------------------------------------
# PCA → RGB
# ---------------------------------------------------------------------------


def pca_to_rgb(
    patch_tokens: torch.Tensor, n_components: int = 3, whiten: bool = True
) -> np.ndarray:
    flat = patch_tokens.squeeze(0).cpu().float()
    mean = flat.mean(dim=0)
    centered = flat - mean
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    components = U[:, :n_components] * S[:n_components]
    if whiten:
        components = components / (S[:n_components] + 1e-8)
    for c in range(n_components):
        cmin = components[:, c].min()
        cmax = components[:, c].max()
        denom = cmax - cmin if (cmax - cmin) > 0 else 1.0
        components[:, c] = (components[:, c] - cmin) / denom
    return components.cpu().numpy()


# ---------------------------------------------------------------------------
# Pos-embed 3-D analysis
# ---------------------------------------------------------------------------


def analyse_3d_pos_embed(output_dir: Path) -> None:
    """Run the same spatial analyses on the 3-D pos_embed and save plots."""
    ckpt = torch.load(CHECKPOINT_3D, map_location="cpu")
    pos = ckpt["pos_embed"]
    cls_tok = pos[:, :1, :]
    patches = pos[:, 1:, :]
    H = int(round(patches.shape[1] ** (1 / 3)))
    assert H * H * H == patches.shape[1], f"Expected cubic grid, got {patches.shape[1]}"

    grid = patches.view(H, H, H, -1).cpu().numpy()
    norms = np.linalg.norm(grid, axis=-1)

    from torch.nn.functional import cosine_similarity

    t = torch.from_numpy(grid)

    # Neighbor similarities
    h_sim = cosine_similarity(t[1:], t[:-1]).mean().item()
    v_sim = cosine_similarity(t[:, 1:], t[:, :-1]).mean().item()
    d_sim = cosine_similarity(t[:, :, 1:], t[:, :, :-1]).mean().item()
    n = 5000
    rng = np.random.default_rng(42)
    idx1 = rng.integers(0, H, n)
    idx2 = rng.integers(0, H, n)
    idx3 = rng.integers(0, H, n)
    idx4 = rng.integers(0, H, n)
    idx5 = rng.integers(0, H, n)
    idx6 = rng.integers(0, H, n)
    r_sim = (
        cosine_similarity(
            t[idx1, idx2, idx3].reshape(n, -1),
            t[idx4, idx5, idx6].reshape(n, -1),
        )
        .mean()
        .item()
    )

    # Position correlations
    flat = grid.reshape(-1, grid.shape[-1])
    x = np.repeat(np.arange(H), H * H)
    y = np.tile(np.repeat(np.arange(H), H), H)
    z = np.tile(np.arange(H), H * H)
    flat_c = flat - flat.mean(axis=0)
    corr_stats = {}
    for name, coord in [("x", x), ("y", y), ("z", z)]:
        c = coord - coord.mean()
        corr = np.abs(flat_c.T @ c) / (
            np.linalg.norm(flat_c, axis=0) * np.linalg.norm(c)
        )
        corr_stats[f"{name}_mean"] = float(corr.mean())
        corr_stats[f"{name}_max"] = float(corr.max())

    # Radial profile from centre
    cy, cx, cz = H // 2, H // 2, H // 2
    yy, xx, zz = np.ogrid[:H, :H, :H]
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    unique_r = np.sort(np.unique(np.round(radii, 0)))
    rad_mean, rad_std = [], []
    for r in unique_r:
        mask = np.abs(radii - r) < 0.5
        if mask.sum() == 0:
            continue
        rad_mean.append(norms[mask].mean())
        rad_std.append(norms[mask].std())

    # --- Build figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.suptitle(
        "3-D ViT Position Embedding Analysis  —  "
        f"{H}×{H}×{H} patch grid  ·  embed_dim={grid.shape[-1]}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) Norm heatmap — central depth slice
    ax = axes[0, 0]
    mid = 0  # H // 2
    im = ax.imshow(norms[..., mid], cmap="viridis", aspect="equal")
    ax.set_title(f"Patch Norms  (central slice z={mid})", fontsize=11)
    ax.set_xlabel("patch x")
    ax.set_ylabel("patch y")
    plt.colorbar(im, ax=ax, label="‖eᵢ‖₂", shrink=0.8)

    # 2) Radial profile
    ax = axes[0, 1]
    ax.errorbar(
        unique_r,
        rad_mean,
        yerr=rad_std,
        fmt="o-",
        markersize=3,
        capsize=2,
        color="#2166ac",
    )
    ax.set_title("Radial Profile — Mean Norm vs. Distance from Centre", fontsize=11)
    ax.set_xlabel("radial distance (patches)")
    ax.set_ylabel("mean ‖eᵢ‖₂")
    ax.grid(True, alpha=0.3)
    ax.axhline(
        norms.mean(),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"overall mean = {norms.mean():.3f}",
    )
    ax.legend(fontsize=9)

    # 3) Neighbor vs. random cosine similarity
    ax = axes[0, 2]
    labels = ["horizontal", "vertical", "diagonal", "random baseline"]
    values = [h_sim, v_sim, d_sim, r_sim]
    colors = ["#2166ac"] * 3 + ["#d6604b"]
    bars = ax.barh(labels, values, color=colors)
    ax.set_title("Cosine Similarity: Neighbours vs. Random", fontsize=11)
    ax.set_xlabel("mean cosine similarity")
    ax.set_xlim(0, 1.0)
    for bar, v in zip(bars, values):
        ax.text(
            v + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=10,
        )
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    # 4) Position correlation bars
    ax = axes[1, 0]
    corr_labels = ["mean\n|x|", "mean\n|y|", "mean\n|z|", "max\n|x|", "max\n|y|"]
    corr_vals = [
        corr_stats["x_mean"],
        corr_stats["y_mean"],
        corr_stats["z_mean"],
        corr_stats["x_max"],
        corr_stats["y_max"],
    ]
    bars = ax.bar(corr_labels, corr_vals, color=["#729fcf"] * 3 + ["#8ae234"] * 2)
    ax.set_title("Embedding Dim ↔ Coordinate Correlation", fontsize=11)
    ax.set_ylabel("mean / max |Pearson r|")
    ax.set_ylim(0, 1.0)
    for bar, v in zip(bars, corr_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    # 5) First 8 embedding dimensions — central x-slice
    ax = axes[1, 1]
    n_dims = min(8, grid.shape[-1])
    cols = 4
    rows = int(np.ceil(n_dims / cols))
    # We'll just show a few slices
    sub_fig, sub_axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    sub_fig.suptitle("First 8 Embedding Dims  (slice x=18)", fontsize=11)
    for i in range(n_dims):
        sa = sub_axes[i // cols, i % cols]
        slice_2d = grid[18, :, :, i]
        im = sa.imshow(slice_2d, cmap="coolwarm", aspect="equal")
        sa.set_title(f"dim {i}", fontsize=9)
        sa.set_xticks([])
        sa.set_yticks([])
        plt.colorbar(im, ax=sa, shrink=0.7)
    for j in range(n_dims, len(sub_axes.flat)):
        sub_axes.flat[j].set_visible(False)
    plt.tight_layout()
    # Embed sub_fig into the main figure area — actually let's just skip this
    # and use the space for a 3D norm slice visualization instead
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        "See individual dim plots\nfor per-dimension layout",
        ha="center",
        va="center",
        fontsize=10,
        color="dimgray",
    )

    # 6) Token norm comparison
    ax = axes[1, 2]
    cls_norm = cls_tok.norm(dim=-1).item()
    patch_norms_flat = norms.flatten()
    categories = ["cls", "patch mean", "patch min", "patch max"]
    vals = [
        cls_norm,
        patch_norms_flat.mean(),
        patch_norms_flat.min(),
        patch_norms_flat.max(),
    ]
    bars = ax.bar(categories, vals, color=["#edae49", "#2166ac", "#2166ac", "#2166ac"])
    ax.set_title("Token Norm Comparison", fontsize=11)
    ax.set_ylabel("L2 norm")
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "pos_embed_3d_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Individual dimension plots
    n_dims = min(8, grid.shape[-1])
    cols = 4
    rows = int(np.ceil(n_dims / cols))
    fig, axes_d = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5), dpi=150)
    axes_d = np.asarray(axes_d).reshape(-1)
    fig.suptitle(
        "First 8 Embedding Dimensions — 3-D Grid Slices", fontsize=13, fontweight="bold"
    )
    for i in range(n_dims):
        ax = axes_d[i]
        # Show three orthogonal slices
        slice_yz = grid[:, 18, :, i]  # x=18
        slice_xz = grid[18, :, :, i]  # y=18
        slice_xy = grid[:, :, 18, i]  # z=18
        # Combine into a single image: [H, 3*H] with yz | xz | xy
        combined = np.hstack([slice_yz, slice_xz, slice_xy])
        im = ax.imshow(combined, cmap="coolwarm", aspect="equal")
        ax.set_title(
            f"dim [{i}]  (μ={grid[:, :, :, i].mean():.4f}, σ={grid[:, :, :, i].std():.4f})",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            -0.08,
            "yz | xz | xy",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="dimgray",
        )
        plt.colorbar(im, ax=ax, shrink=0.8)
    for j in range(n_dims, len(axes_d)):
        axes_d[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path2 = output_dir / "pos_embed_3d_first_dims.png"
    fig.savefig(out_path2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path2}")


# ---------------------------------------------------------------------------
# Volume PCA (3-D)
# ---------------------------------------------------------------------------


def plot_volume_pca_3d(
    dataset_name: str = "CLS002_FOMO26_Infarct",
    output_dir: Path | str = OUTPUT_DIR,
    max_depth_slices: int = MAX_SLICES,
    sample_index: int = 0,
    patch_size=14,
    resize_to=(384, 512, 384),
) -> Path:
    """Run the 3-D volume → patch-token PCA pipeline and save figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from med_adapt.registry import STORE

    dataset_cls = STORE.get("datasets", dataset_name)
    dataset = dataset_cls(
        root=DATASET_ROOT, fold=None, seed=None, n_splits=5, resize_to=resize_to
    )
    sample = dataset[sample_index]
    volume = sample["image"]
    print(
        f"Volume shape: {volume.shape}, label: {sample['label'].item()}, subject: {sample['subject']}"
    )

    vol = preprocess_volume(volume)
    print(f"Preprocessed volume: {vol.shape}")

    model = load_3d_model(patch_size=patch_size)
    print(f"Model loaded, patch_size={model.patch_size}")

    vol_tensor = vol.unsqueeze(0).to(DEVICE)  # [1, C, H, W, D]
    with torch.no_grad():
        out = model(vol_tensor)

    patch_tokens = out["patch_latent"]  # [1, N, E]
    cls_token = out["latent"].unsqueeze(1)  # [1, 1, E]
    # Use the ACTUAL patch grid dimensions for this input volume,
    # not the model's base (518×518×518) grid.
    B, _, H_in, W_in, D_in = vol_tensor.shape
    ps = model.patch_size
    if isinstance(ps, int):
        ps = (ps, ps, ps)
    ph = H_in // ps[0]
    pw = W_in // ps[1]
    pd_ = D_in // ps[2]
    print(f"Patch grid: {ph}×{pw}×{pd_} = {ph*pw*pd_} patches")

    # Select depth slices of the PATCH GRID (not the volume)
    if pd_ <= max_depth_slices:
        slice_indices = list(range(pd_))
    else:
        slice_indices = np.linspace(0, pd_ - 1, max_depth_slices, dtype=int).tolist()
    print(f"Selected patch-depth slices: {slice_indices}")

    pca_images: list[np.ndarray] = []
    cosine_images: list[np.ndarray] = []

    for d_idx in slice_indices:
        # Extract patches for this depth slice
        start = d_idx * ph * pw
        end = (d_idx + 1) * ph * pw
        slice_tokens = patch_tokens[:, start:end, :]  # [1, ph*pw, E]
        cls_t = cls_token  # [1, 1, E]

        # PCA-RGB
        pca_rgb = pca_to_rgb(slice_tokens, n_components=3, whiten=True)
        pca_rgb = pca_rgb.reshape(ph, pw, 3)

        # Cosine similarity
        cos_sim = F.cosine_similarity(slice_tokens, cls_t, dim=-1)  # [1, ph*pw]
        cos_sim = cos_sim.cpu().numpy().reshape(ph, pw)

        pca_images.append(pca_rgb)
        cosine_images.append(cos_sim)

    n = len(pca_images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # --- PCA-RGB panel ------------------------------------------------------
    fig_pca, axes_pca = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 4.5), dpi=150
    )
    axes_pca = np.asarray(axes_pca).reshape(-1)
    fig_pca.suptitle(
        "3-D ViT: Patch-Token PCA (whitened) per patch-depth slice",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    for i, (ax, pca_img, d_idx) in enumerate(zip(axes_pca, pca_images, slice_indices)):
        ax.imshow(pca_img, vmin=0, vmax=1)
        ax.set_title(f"patch-depth z={d_idx}  ({ph}×{pw} patches)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes_pca)):
        axes_pca[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pca_path = output_dir / "volume_pca_3d_rgb.png"
    fig_pca.savefig(pca_path, bbox_inches="tight")
    plt.close(fig_pca)
    print(f"Saved → {pca_path}")

    # --- Cosine panel -------------------------------------------------------
    fig_cos, axes_cos = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 4.5), dpi=150
    )
    axes_cos = np.asarray(axes_cos).reshape(-1)
    fig_cos.suptitle(
        "3-D ViT: Patch-token ↔ CLS cosine similarity per patch-depth slice",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    for i, (ax, cos_img, d_idx) in enumerate(
        zip(axes_cos, cosine_images, slice_indices)
    ):
        im = ax.imshow(cos_img, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"patch-depth z={d_idx}  ({ph}×{pw} patches)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.8, label="cosine sim")
    for j in range(n, len(axes_cos)):
        axes_cos[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    cos_path = output_dir / "volume_pca_3d_cosine.png"
    fig_cos.savefig(cos_path, bbox_inches="tight")
    plt.close(fig_cos)
    print(f"Saved → {cos_path}")

    return pca_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    analyse_3d_pos_embed(OUTPUT_DIR)
    plot_volume_pca_3d(patch_size=14, resize_to=(384, 512, 384))
    # plot_volume_pca_3d(patch_size=(14, 14, 1), resize_to=None)
