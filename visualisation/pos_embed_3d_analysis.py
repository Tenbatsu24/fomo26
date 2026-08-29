"""Visualise the learned position embedding of a 3-D ViT checkpoint.

The position embedding (`pos_embed`) in a 3-D Vision Transformer is a
learned lookup table that assigns a dense vector to every volumetric
patch (plus the cls token).  This module loads a checkpoint, extracts
the embedding, and produces a set of plots that answer the question:
*what kind of 3-D spatial signal does pos_embed carry?*
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import cosine_similarity

from visualisation.utils import colorbar

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "understand" / "pos_embed_3d"


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def analyse_3d_pos_embed(
    checkpoint_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run the spatial analyses on the 3-D pos_embed and save plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_sd = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    pos = raw_sd["model.pos_embed"]
    cls_tok = pos[:, :1, :]
    patches = pos[:, 1:, :]
    H = int(round(patches.shape[1] ** (1 / 3)))
    assert H * H * H == patches.shape[1], f"Expected cubic grid, got {patches.shape[1]}"

    grid = patches.view(H, H, H, -1).cpu().numpy()
    norms = np.linalg.norm(grid, axis=-1)

    # Neighbor similarities
    t = torch.from_numpy(grid)
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150, constrained_layout=True)
    fig.suptitle(
        "3-D ViT Position Embedding Analysis  —  "
        f"{H}×{H}×{H} patch grid  ·  embed_dim={grid.shape[-1]}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) Norm heatmap — central depth slice
    ax = axes[0, 0]
    mid = H // 2
    im = ax.imshow(norms[..., mid], cmap="viridis", aspect="equal")
    ax.set_title(f"Patch Norms  (central slice z={mid})", fontsize=11)
    ax.set_xlabel("patch x")
    ax.set_ylabel("patch y")
    colorbar(fig, im, ax=ax, label="‖eᵢ‖₂")

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

    # 5) Placeholder — per-dimension plots saved separately
    ax = axes[1, 1]
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

    out_path = output_dir / "pos_embed_3d_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # --- Individual dimension plots (single colour bar) ----------------------
    n_dims = min(8, grid.shape[-1])
    cols = 4
    rows = int(np.ceil(n_dims / cols))
    fig, axes_d = plt.subplots(
        rows, cols, figsize=(cols * 5 + 1, rows * 4), dpi=150, constrained_layout=True
    )
    axes_d = np.asarray(axes_d).reshape(-1)
    fig.suptitle(
        "First 8 Embedding Dimensions — 3-D Grid Slices",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    mid = H // 2
    # Global vmin/vmax across all displayed dimensions
    all_vals = grid[:, mid, :, :n_dims].reshape(-1, n_dims)
    global_min = all_vals.min()
    global_max = all_vals.max()

    for i in range(n_dims):
        ax = axes_d[i]
        slice_yz = grid[:, mid, :, i]  # x=mid
        slice_xz = grid[mid, :, :, i]  # y=mid
        slice_xy = grid[:, :, mid, i]  # z=mid
        combined = np.hstack([slice_yz, slice_xz, slice_xy])
        im = ax.imshow(
            combined, cmap="coolwarm", aspect="equal", vmin=global_min, vmax=global_max
        )
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
    fig.colorbar(im, ax=axes_d.tolist(), shrink=0.8, label="value")
    for j in range(n_dims, len(axes_d)):
        axes_d[j].set_visible(False)
    out_path2 = output_dir / "pos_embed_3d_first_dims.png"
    fig.savefig(out_path2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path2}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise the learned position embedding of a 3-D ViT checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the 3-D ViT checkpoint file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save output figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    analyse_3d_pos_embed(checkpoint_path=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
