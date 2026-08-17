"""Visualise the learned 3-D patch-embedding convolution weights and bias.

The 3-D patch embedding (`patch_embed.proj`) is a 3-D convolution that
maps an input patch ``(C, ps_z, ps_y, ps_x)`` to an embedding vector
of dimension ``embed_dim``.  This module loads a checkpoint, extracts
the convolution kernel and bias, and produces a set of diagnostic
plots analogous to the 2-D patch-embedding analysis.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from visualisation.utils import colorbar, save_figure

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "understand" / "patch_embed_3d"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_patch_embed(
    checkpoint_path: Path | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load 3-D patch_embed weights and bias from a ViT checkpoint.

    Returns
    -------
    weight : torch.Tensor  shape [embed_dim, in_chans, ps_z, ps_y, ps_x]
    bias   : torch.Tensor  shape [embed_dim]
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        raw_sd = ckpt
    weight = raw_sd["model.patch_embed.proj.weight"]  # [E, C, pz, py, px]
    bias = raw_sd["model.patch_embed.proj.bias"]  # [E]
    return weight, bias


def to_numpy(weight: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return weight.cpu().numpy(), bias.cpu().numpy()


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------


def kernel_l2_norms(weight: np.ndarray) -> np.ndarray:
    """L2 norm of each output-channel kernel → [embed_dim]."""
    flat = weight.reshape(weight.shape[0], -1)
    return np.linalg.norm(flat, axis=1)


def kernel_l1_norms(weight: np.ndarray) -> np.ndarray:
    """L1 norm of each output-channel kernel → [embed_dim]."""
    return np.abs(weight).reshape(weight.shape[0], -1).sum(axis=1)


def kernel_sparsity(weight: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Fraction of near-zero entries per kernel → [embed_dim]."""
    flat = weight.reshape(weight.shape[0], -1)
    return (np.abs(flat) < threshold).mean(axis=1)


def per_channel_norms(weight: np.ndarray) -> np.ndarray:
    """Mean kernel norm contributed by each input channel → [in_chans]."""
    E, C = weight.shape[0], weight.shape[1]
    flat = weight.transpose(1, 0, 2, 3, 4).reshape(C, -1)
    channel_norms = np.linalg.norm(flat, axis=1)
    return channel_norms / E


def bias_statistics(bias: np.ndarray) -> dict[str, float]:
    """Descriptive statistics for the bias vector."""
    return {
        "mean": float(bias.mean()),
        "std": float(bias.std()),
        "min": float(bias.min()),
        "max": float(bias.max()),
        "median": float(np.median(bias)),
    }


def layer_spectral_norm(weight: np.ndarray) -> float:
    """Largest singular value of the full [embed_dim, C·pz·py·px] weight matrix."""
    flat = weight.reshape(weight.shape[0], -1)
    s = np.linalg.svd(flat, compute_uv=False, full_matrices=False)
    return float(s[0])


def singular_value_spectrum(weight: np.ndarray) -> np.ndarray:
    """All singular values of the flattened weight matrix, sorted descending."""
    flat = weight.reshape(weight.shape[0], -1)
    return np.linalg.svd(flat, compute_uv=False, full_matrices=False)


def spectral_condition_number(weight: np.ndarray) -> float:
    """Ratio of largest to smallest singular value."""
    sv = singular_value_spectrum(weight)
    return float(sv[0] / (sv[-1] + 1e-12))


def spectral_energy_fraction(
    weight: np.ndarray, cumsum_thresh: float = 0.95
) -> tuple[int, float]:
    """Number of singular values needed to capture *cumsum_thresh* of total energy."""
    sv = singular_value_spectrum(weight)
    sv2 = sv**2
    cumsum = np.cumsum(sv2) / sv2.sum()
    n_eff = int(np.searchsorted(cumsum, cumsum_thresh) + 1)
    return n_eff, float(cumsum[n_eff - 1])


def neighbor_kernel_similarity(weight: np.ndarray) -> dict[str, float]:
    """Mean cosine similarity between adjacent output-channel kernels."""
    flat = weight.reshape(weight.shape[0], -1).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1)
    cos_sim = flat @ flat.T / (norms[:, None] * norms[None, :] + 1e-8)
    np.fill_diagonal(cos_sim, 0)
    return {
        "mean_pairwise_cos_sim": float(cos_sim.mean()),
        "median_pairwise_cos_sim": float(np.median(cos_sim)),
        "max_pairwise_cos_sim": float(cos_sim.max()),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _show_kernel_slices(
    weight: np.ndarray, start: int, n: int, title: str
) -> plt.Figure:
    """Display *n* kernels as a grid of central slices.

    For each output channel, shows the central ``(ps_y, ps_x)`` slice
    summed across the depth dimension, plus the central ``(ps_z, ps_x)``
    and ``(ps_z, ps_y)`` slices.
    """
    E, C, pz, py, px = weight.shape
    rows = math.ceil(n**0.5)
    cols = 3 * math.ceil(n / rows)  # 3 views per kernel
    sub = weight[start : start + n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for i in range(min(n, rows * cols // 3)):
        k = sub[i]  # [C, pz, py, px]
        # Sum across input channels for visibility
        summed = k.sum(axis=0)  # [pz, py, px]

        views = [
            (summed[:, :, px // 2], f"mid px"),
            (summed[:, py // 2, :], f"mid py"),
            (summed[pz // 2, :, :], f"mid pz"),
        ]
        base_idx = i * 3
        for j, (view, label) in enumerate(views):
            ax = axes[base_idx + j]
            im = ax.imshow(view, cmap="coolwarm", aspect="equal")
            ax.set_title(f"ch {start + i}  {label}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            colorbar(fig, im, ax=ax, label="" if j > 0 else "value")

    for j in range(min(n, rows * cols // 3) * 3, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def plot_patch_embed_3d(
    checkpoint_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Generate 3-D patch-embedding visualisations and save to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ----------------------------------------------------------------
    weight, bias = load_patch_embed(checkpoint_path)
    weight_np, bias_np = to_numpy(weight, bias)
    E, C, pz, py, px = weight_np.shape

    # --- Compute stats -------------------------------------------------------
    l2_norms = kernel_l2_norms(weight_np)
    l1_norms = kernel_l1_norms(weight_np)
    sparsity = kernel_sparsity(weight_np)
    per_ch = per_channel_norms(weight_np)
    bias_stats = bias_statistics(bias_np)
    neighbor_sim = neighbor_kernel_similarity(weight_np)
    layer_spec = layer_spectral_norm(weight_np)
    sv_spectrum = singular_value_spectrum(weight_np)
    cond_num = spectral_condition_number(weight_np)
    n_eff_95, frac_95 = spectral_energy_fraction(weight_np, cumsum_thresh=0.95)
    n_eff_90, frac_90 = spectral_energy_fraction(weight_np, cumsum_thresh=0.90)

    # --- Build composite figure ----------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    fig.suptitle(
        "3-D ViT Patch Embedding Analysis  —  "
        f"Conv3d[{C}→{E}, ks={pz}×{py}×{px}]  ·  σ₁ = {layer_spec:.3f}  ·  κ = {cond_num:.0f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) First 12 kernels as central slices
    _fig_kernels = _show_kernel_slices(
        weight_np, start=0, n=12, title="First 12 Kernels  (central slices)"
    )
    out_path_kernels = output_dir / "patch_embed_3d_first_kernels.png"
    _fig_kernels.savefig(out_path_kernels, bbox_inches="tight")
    plt.close(_fig_kernels)
    print(f"Saved → {out_path_kernels}")
    axes[0, 0].set_visible(False)

    # 2) Singular value spectrum
    ax = axes[0, 1]
    sv_idx = np.arange(1, len(sv_spectrum) + 1)
    ax.semilogy(
        sv_idx,
        sv_spectrum,
        marker="o",
        markersize=2,
        color="#2166ac",
        linewidth=1.2,
        alpha=0.8,
    )
    ax.axvline(n_eff_90, color="orange", linestyle="--", label=f"90% at n={n_eff_90}")
    ax.axvline(n_eff_95, color="red", linestyle="--", label=f"95% at n={n_eff_95}")
    ax.set_title("Singular Value Spectrum", fontsize=11)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σᵢ (log scale)")
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"σ₁ = {layer_spec:.3f}\nκ = {cond_num:.0f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="dimgray",
    )

    # 3) Kernel L2-norm distribution
    ax = axes[0, 2]
    ax.hist(l2_norms, bins=60, color="#2166ac", edgecolor="white", alpha=0.8)
    ax.set_title("Kernel L2 Norm Distribution", fontsize=11)
    ax.set_xlabel("‖wₑ‖₂")
    ax.set_ylabel("count")
    ax.axvline(
        l2_norms.mean(),
        color="red",
        linestyle="--",
        label=f"mean = {l2_norms.mean():.3f}",
    )
    ax.axvline(
        np.median(l2_norms),
        color="orange",
        linestyle="--",
        label=f"median = {np.median(l2_norms):.3f}",
    )
    ax.legend(fontsize=9)

    # 4) Kernel L1 vs L2 norm scatter
    ax = axes[1, 0]
    ax.scatter(l2_norms, l1_norms, s=4, color="#2166ac", alpha=0.6)
    ax.set_title("L1 Norm vs L2 Norm  (per kernel)", fontsize=11)
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("L1 norm")
    max_ratio = (pz * py * px * C) ** 0.5
    ax.axline(
        (0, 0),
        slope=max_ratio,
        color="red",
        linestyle=":",
        label=f"max ratio = {max_ratio:.2f}",
    )
    ax.legend(fontsize=9)

    # 5) Per-input-channel norm contribution
    ax = axes[1, 1]
    ch_labels = [f"in_ch {i}" for i in range(C)]
    bar_colors = plt.cm.tab10(np.linspace(0, 1, C))
    bars = ax.bar(ch_labels, per_ch, color=bar_colors)
    ax.set_title("Mean Kernel Norm per Input Channel", fontsize=11)
    ax.set_ylabel("avg ‖wₑ[c, :, :, :]‖₂")
    for bar, v in zip(bars, per_ch):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    # 6) Bias + spectral summary
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"Weight shape : {weight_np.shape}\n"
        f"Bias shape   : {bias_np.shape}\n"
        f"Layer spectral norm : {layer_spec:.4f}\n"
        f"Condition number   : {cond_num:.1f}\n"
        f"Effective rank (90%): {n_eff_90} / {len(sv_spectrum)} ({frac_90:.0%})\n"
        f"Effective rank (95%): {n_eff_95} / {len(sv_spectrum)} ({frac_95:.0%})\n"
        f"Kernel L2 — mean: {l2_norms.mean():.4f}, std: {l2_norms.std():.4f}\n"
        f"Sparsity    — mean: {sparsity.mean():.4f}\n"
        f"Bias — μ: {bias_stats['mean']:.4f}, σ: {bias_stats['std']:.4f}\n"
        f"Neighbor cos sim — mean: {neighbor_sim['mean_pairwise_cos_sim']:.4f}"
    )
    ax.text(
        0.5,
        0.5,
        summary_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        family="monospace",
        color="dimgray",
    )
    ax.set_title("Summary Statistics", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "patch_embed_3d_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")

    # --- Singular value plot -------------------------------------------------
    _save_sv_plot(
        sv_spectrum,
        layer_spec,
        cond_num,
        n_eff_90,
        frac_90,
        n_eff_95,
        frac_95,
        output_dir,
    )

    # --- Print summary -------------------------------------------------------
    print("\n=== 3-D Patch Embed Summary ===")
    print(f"  Weight shape : {weight_np.shape}")
    print(f"  Bias shape   : {bias_np.shape}")
    print(f"  Layer spectral norm : {layer_spec:.4f}")
    print(f"  Singular values    : {len(sv_spectrum)} total, κ = {cond_num:.1f}")
    print(
        f"  Effective rank (90% energy) : {n_eff_90} / {len(sv_spectrum)}  ({frac_90:.1%})"
    )
    print(
        f"  Effective rank (95% energy) : {n_eff_95} / {len(sv_spectrum)}  ({frac_95:.1%})"
    )
    print(
        f"  Kernel L2  — mean: {l2_norms.mean():.4f},  std: {l2_norms.std():.4f},  "
        f"min: {l2_norms.min():.4f},  max: {l2_norms.max():.4f}"
    )
    print(f"  Sparsity   — mean: {sparsity.mean():.4f}")
    print(f"  Bias       — μ: {bias_stats['mean']:.4f},  σ: {bias_stats['std']:.4f}")
    print(f"  Per-input-channel avg norm: {per_ch}")
    print(
        f"  Neighbor cos sim — mean: {neighbor_sim['mean_pairwise_cos_sim']:.4f},  "
        f"max: {neighbor_sim['max_pairwise_cos_sim']:.4f}"
    )

    return out_path


def _save_sv_plot(
    sv_spectrum: np.ndarray,
    layer_spec: float,
    cond_num: float,
    n_eff_90: int,
    frac_90: float,
    n_eff_95: int,
    frac_95: float,
    out: Path,
) -> None:
    """Spectral analysis: scree plot and cumulative energy."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.suptitle(
        f"3-D Patch Embed Spectral Analysis  —  σ₁ = {layer_spec:.3f}  ·  κ = {cond_num:.0f}",
        fontsize=13,
        fontweight="bold",
    )

    ax = axes[0]
    sv_idx = np.arange(1, len(sv_spectrum) + 1)
    ax.semilogy(sv_idx, sv_spectrum, color="#2166ac", linewidth=1.2, alpha=0.8)
    ax.axvline(n_eff_90, color="orange", linestyle="--", label=f"90% at n={n_eff_90}")
    ax.axvline(n_eff_95, color="red", linestyle="--", label=f"95% at n={n_eff_95}")
    ax.set_title("Singular Value Spectrum", fontsize=11)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σᵢ (log scale)")
    ax.legend(fontsize=9)

    ax = axes[1]
    sv2 = sv_spectrum**2
    cumsum_frac = np.cumsum(sv2) / sv2.sum()
    ax.plot(cumsum_frac, color="#2166ac", linewidth=1.5)
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.7)
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.7)
    ax.set_title("Cumulative Energy Fraction", fontsize=11)
    ax.set_xlabel("number of singular values")
    ax.set_ylabel("Σσᵢ² / Σσⱼ²")
    ax.set_ylim(0.85, 1.01)

    plt.tight_layout()
    out_path = out / "patch_embed_3d_spectral.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise the learned 3-D patch embedding of a ViT checkpoint."
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

    plot_patch_embed_3d(checkpoint_path=args.checkpoint, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
