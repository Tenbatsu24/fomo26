"""Visualise the learned position embedding of a 2-D ViT checkpoint.

The position embedding (`pos_embed`) in a Vision Transformer is a
learned lookup table that assigns a dense vector to every patch (plus
the cls / register tokens).  This module loads a checkpoint, extracts
the embedding, and produces a set of plots that answer the question:
*what kind of signal does pos_embed carry?*

Typical hypotheses tested here:
  1. **Flat/uniform** — every token gets the same vector (null model).
  2. **Sinusoidal / explicit coordinate encoding** — each embedding
     dimension correlates cleanly with x / y grid position.
  3. **Radial / centre-weighted** — embeddings vary smoothly with
     distance from the image centre.
  4. **Learned spatial continuity** — neighbouring patches have
     similar embeddings, but the mapping is non-linear and not
     trivially invertible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from visualisation.utils import colorbar

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "understand" / "pos_embed_2d"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_pos_embed(
    checkpoint_path: Path | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pos_embed, cls_token and register_tokens from a ViT checkpoint.

    Returns
    -------
    cls_token : torch.Tensor  shape [1, 1, D]
    patch_pos : torch.Tensor  shape [1, H*W, D]
    reg_token : torch.Tensor  shape [1, R, D]
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cls_token = ckpt["cls_token"]
    patch_pos = ckpt["pos_embed"][:, 1:, :]  # drop cls from pos_embed
    reg_token = ckpt.get("register_tokens", torch.zeros(1, 0, cls_token.shape[-1]))
    return cls_token, patch_pos, reg_token


def to_grid(patch_pos: torch.Tensor, grid_size: int) -> np.ndarray:
    """Reshape [1, H*W, D] → [H, W, D] and bring to CPU for numpy ops."""
    return patch_pos.view(grid_size, grid_size, -1).cpu().numpy()


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------


def patch_norms(grid: np.ndarray) -> np.ndarray:
    """L2 norm of each patch embedding → [H, W]."""
    return np.linalg.norm(grid, axis=-1)


def radial_profile(
    norms: np.ndarray, center: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean norm as a function of radial distance from *center*."""
    if center is None:
        h, w = norms.shape
        center = (h // 2, w // 2)
    cy, cx = center
    yy, xx = np.ogrid[: norms.shape[0], : norms.shape[1]]
    radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    unique_r = np.sort(np.unique(np.round(radii, 0)))
    means, stds = [], []
    for r in unique_r:
        mask = np.abs(radii - r) < 0.5
        if mask.sum() == 0:
            continue
        means.append(norms[mask].mean())
        stds.append(norms[mask].std())
    return unique_r, np.array(means), np.array(stds)


def neighbor_cosine_similarity(grid: np.ndarray) -> dict[str, float]:
    """Cosine similarity between adjacent patch embeddings."""
    from torch.nn.functional import cosine_similarity

    t = torch.from_numpy(grid).float()
    h_sim = cosine_similarity(t[1:], t[:-1]).mean().item()
    v_sim = cosine_similarity(t[:, 1:], t[:, :-1]).mean().item()
    d_sim = cosine_similarity(t[1:, 1:], t[:-1, :-1]).mean().item()
    n = 5000
    idx1 = np.random.default_rng(42).integers(0, grid.shape[0], n)
    idx2 = np.random.default_rng(43).integers(0, grid.shape[1], n)
    rnd = np.random.default_rng(44).integers(0, grid.shape[0], n)
    rnd2 = np.random.default_rng(45).integers(0, grid.shape[1], n)
    r_sim = (
        cosine_similarity(
            torch.from_numpy(grid[idx1, idx2]),
            torch.from_numpy(grid[rnd, rnd2]),
        )
        .mean()
        .item()
    )
    return {
        "horizontal": h_sim,
        "vertical": v_sim,
        "diagonal": d_sim,
        "random_baseline": r_sim,
    }


def position_correlations(grid: np.ndarray) -> dict[str, float]:
    """Mean absolute Pearson correlation of each embedding dim with x/y/radius."""
    H, W, D = grid.shape
    x = np.repeat(np.arange(W), H)
    y = np.tile(np.arange(H), W)
    cy, cx = H // 2, W // 2
    dist = np.sqrt(
        (np.arange(H)[:, None] - cy) ** 2 + (np.arange(W)[None, :] - cx) ** 2
    )
    dist = dist.reshape(H * W).astype(np.float64)
    flat = grid.reshape(-1, D).astype(np.float64)
    flat_c = flat - flat.mean(axis=0)
    x_c = x - x.mean()
    y_c = y - y.mean()
    d_c = dist - dist.mean()
    denom_x = np.linalg.norm(x_c)
    denom_y = np.linalg.norm(y_c)
    denom_d = np.linalg.norm(d_c)
    corr_x = np.abs(flat_c.T @ x_c) / (np.linalg.norm(flat_c, axis=0) * denom_x)
    corr_y = np.abs(flat_c.T @ y_c) / (np.linalg.norm(flat_c, axis=0) * denom_y)
    corr_r = np.abs(flat_c.T @ d_c) / (np.linalg.norm(flat_c, axis=0) * denom_d)
    return {
        "x_position_mean_abs_corr": float(corr_x.mean()),
        "y_position_mean_abs_corr": float(corr_y.mean()),
        "radial_mean_abs_corr": float(corr_r.mean()),
        "x_position_max_abs_corr": float(corr_x.max()),
        "y_position_max_abs_corr": float(corr_y.max()),
    }


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def plot_pos_embed(
    checkpoint_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    grid_size: int = 37,
) -> Path:
    """Generate all position-embedding visualisations and save to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ----------------------------------------------------------------
    cls_token, patch_pos, reg_token = load_pos_embed(checkpoint_path)
    grid_np = to_grid(patch_pos, grid_size)  # [H, W, D] on CPU
    norms = patch_norms(grid_np)

    # --- Compute stats -------------------------------------------------------
    neighbor_sim = neighbor_cosine_similarity(grid_np)
    pos_corr = position_correlations(grid_np)
    radii, rad_mean, rad_std = radial_profile(norms)

    # --- PC1 / PC2 via sklearn PCA ------------------------------------------
    flat = torch.from_numpy(grid_np).float().reshape(-1, grid_np.shape[-1])
    pca = PCA(n_components=2, whiten=False)
    proj = pca.fit_transform(flat.numpy())  # [H*W, 2]
    pc1_grid = proj[:, 0].reshape(grid_size, grid_size)
    pc2_grid = proj[:, 1].reshape(grid_size, grid_size)
    explained_ratio = pca.explained_variance_ratio_

    # --- Build figure --------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150, constrained_layout=True)
    fig.suptitle(
        "ViT Position Embedding Analysis  —  "
        f"{grid_size}×{grid_size} patch grid  ·  embed_dim={grid_np.shape[-1]}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) Patch embedding norms (spatial map)
    ax = axes[0, 0]
    im = ax.imshow(norms, cmap="viridis", aspect="equal")
    ax.set_title("Patch Embedding Norms  (L2)", fontsize=11)
    ax.set_xlabel("patch x")
    ax.set_ylabel("patch y")
    colorbar(fig, im, ax=ax, label="‖eᵢ‖₂")
    h, w = norms.shape
    ax.text(
        0.5,
        -0.12,
        f"centre norm ≈ {norms[h//2-2:h//2+2, w//2-2:w//2+2].mean():.3f}    "
        f"edge norm ≈ {norms[:2, :].mean():.3f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    # 2) Radial profile
    ax = axes[0, 1]
    ax.errorbar(
        radii,
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
    labels = list(neighbor_sim.keys())
    values = list(neighbor_sim.values())
    colors = ["#2166ac" if "random" not in l else "#d6604b" for l in labels]
    bars = ax.barh(labels, values, color=colors)
    ax.set_title("Cosine Similarity: Neighbours vs. Random Pairs", fontsize=11)
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
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance (~0.5)")
    ax.legend(fontsize=9)

    # 4) PC1 of Patch Embeddings (spatial layout)
    ax = axes[1, 0]
    im = ax.imshow(pc1_grid, cmap="coolwarm", aspect="equal")
    ax.set_title(
        f"PC1 of Patch Embeddings  (spatial layout)  "
        f"(explains {explained_ratio[0]:.1%} variance)",
        fontsize=11,
    )
    ax.set_xlabel("patch x")
    ax.set_ylabel("patch y")
    colorbar(fig, im, ax=ax, label="PC1 score")

    # 5) Position-correlation bar chart
    ax = axes[1, 1]
    corr_labels = [
        "mean\n|x-cor|",
        "mean\n|y-cor|",
        "mean\n|radial|",
        "max\n|x-cor|",
        "max\n|y-cor|",
    ]
    corr_vals = [
        pos_corr["x_position_mean_abs_corr"],
        pos_corr["y_position_mean_abs_corr"],
        pos_corr["radial_mean_abs_corr"],
        pos_corr["x_position_max_abs_corr"],
        pos_corr["y_position_max_abs_corr"],
    ]
    bars = ax.bar(
        corr_labels,
        corr_vals,
        color=["#729fcf", "#729fcf", "#ef2929", "#8ae234", "#8ae234"],
    )
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
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="0.3 threshold")
    ax.legend(fontsize=9)

    # 6) cls and register token norms
    ax = axes[1, 2]
    cls_norm = cls_token.norm(dim=-1).item()
    reg_norms = (
        reg_token.norm(dim=-1).cpu().numpy() if reg_token.shape[1] > 0 else np.array([])
    )
    patch_norm_arr = norms.flatten()
    categories = [
        "cls",
        "register (mean)",
        "patch (mean)",
        "patch (min)",
        "patch (max)",
    ]
    vals = [
        cls_norm,
        reg_norms.mean() if len(reg_norms) else 0,
        patch_norm_arr.mean(),
        patch_norm_arr.min(),
        patch_norm_arr.max(),
    ]
    colors_tok = ["#edae49", "#edae49", "#2166ac", "#2166ac", "#2166ac"]
    bars = ax.bar(categories, vals, color=colors_tok)
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

    out_path = output_dir / "pos_embed_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # --- Also save individual diagnostic plots --------------------------------
    _save_dim_grid(patch_pos, grid_size, output_dir)
    _save_distance_heatmap(patch_pos, grid_size, output_dir)

    return out_path


# ---------------------------------------------------------------------------
# Supplementary plots
# ---------------------------------------------------------------------------


def _save_dim_grid(patch_pos: torch.Tensor, grid_size: int, out: Path) -> None:
    """First 8 embedding dimensions as a 2×4 grid of heatmaps with one colour bar."""
    grid = patch_pos.view(grid_size, grid_size, -1).to("cpu")
    n_dims = min(8, grid.shape[-1])
    cols = 4
    rows = int(np.ceil(n_dims / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5 + 1, rows * 4), dpi=150, constrained_layout=True
    )
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(
        "First 8 Embedding Dimensions — Spatial Layout",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    # Determine global vmin/vmax across all dimensions for a single colour bar
    all_vals = grid[:, :, :n_dims].reshape(-1, n_dims)
    global_min = all_vals.min()
    global_max = all_vals.max()

    for i in range(n_dims):
        ax = axes[i]
        im = ax.imshow(
            grid[:, :, i].numpy(),
            cmap="coolwarm",
            aspect="equal",
            vmin=global_min,
            vmax=global_max,
        )
        ax.set_title(
            f"dim [{i}]  (μ={grid[:, :, i].mean():.4f}, σ={grid[:, :, i].std():.4f})"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    # Single shared colour bar for the whole figure
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="value")
    for j in range(n_dims, len(axes)):
        axes[j].set_visible(False)
    out_path = out / "pos_embed_first_dims.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _save_distance_heatmap(patch_pos: torch.Tensor, grid_size: int, out: Path) -> None:
    """Pairwise cosine-distance heatmap for a central crop of the patch grid."""
    grid = patch_pos.view(grid_size, grid_size, -1).to("cpu").numpy()
    crop = 13
    offset = (grid_size - crop) // 2
    sub = grid[offset : offset + crop, offset : offset + crop]
    flat = sub.reshape(-1, sub.shape[-1])
    from torch.nn.functional import cosine_similarity

    t = torch.from_numpy(flat).float()
    sim = cosine_similarity(t.unsqueeze(1), t.unsqueeze(0))
    dist = 1 - sim.numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, constrained_layout=True)
    im = ax.imshow(dist, cmap="coolwarm", vmin=0, vmax=0.5, aspect="equal")
    ax.set_title(f"Pairwise Cosine Distance  ({crop}×{crop} central crop)", fontsize=12)
    ax.set_xlabel("patch index (flattened)")
    ax.set_ylabel("patch index (flattened)")
    fig.colorbar(im, ax=ax, label="1 − cos_sim", shrink=0.8)
    out_path = out / "pos_embed_distance_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise the learned position embedding of a 2-D ViT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the ViT checkpoint file.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=37,
        help="Side length of the square patch grid (default: 37).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    plot_pos_embed(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
    )


if __name__ == "__main__":
    main()
