"""Visualise the learned patch-embedding Conv2d weights and bias.

The patch embedding (`patch_embed.proj`) in a ViT is a convolution that
maps an input patch `(C, ps, ps)` to an embedding vector of dimension
`embed_dim`.  This module loads a checkpoint, extracts the convolution
kernel and bias, and produces a set of plots that answer the question:
*what kind of spatial filters has patch_embed learnt?*

Typical hypotheses tested here:
  1. **Gabor-like / edge detectors** — kernels have oriented bar or
     edge patterns, common in early vision layers.
  2. **Color-contrast kernels** — kernels show sign differences across
     input channels (e.g. R−G, B−(R+G)).
  3. **Blob / center-surround** — radially symmetric excitation /
     inhibition patterns.
  4. **Uniform / low-rank** — many kernels are nearly identical or
     close to zero (under-utilised output channels).

The checkpoint used here is a 2-D ViT-S (patch_size=14, img_size=518)
so the kernel shape is `[embed_dim, in_chans, ps, ps] = [384, 3, 14, 14]`.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "small"
    / "neco"
    / "encoder_teacher.ckpt"
)
OUTPUT_DIR = Path(__file__).resolve().parent
PATCH_SIZE = 14
IN_CHANS = 3
EMBED_DIM = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PALETTE_SEQUENTIAL = "viridis"
PALETTE_DIVERGING = "coolwarm"
PALETTE_GRAY = "gray_r"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_patch_embed(
    checkpoint_path: Path | str = CHECKPOINT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load patch_embed weights and bias from a ViT checkpoint.

    Returns
    -------
    weight : torch.Tensor  shape [embed_dim, in_chans, ps, ps]
    bias   : torch.Tensor  shape [embed_dim]
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    weight = ckpt["patch_embed.proj.weight"]  # [E, C, ps, ps]
    bias = ckpt["patch_embed.proj.bias"]  # [E]
    return weight, bias


def to_numpy(weight: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return weight.cpu().numpy(), bias.cpu().numpy()


# ---------------------------------------------------------------------------
# Helper statistics
# ---------------------------------------------------------------------------


def kernel_l2_norms(weight: np.ndarray) -> np.ndarray:
    """L2 norm of each output-channel kernel → [embed_dim].

    Each kernel has shape (in_chans, ps, ps); we flatten and take the
    Euclidean norm.
    """
    flat = weight.reshape(weight.shape[0], -1)
    return np.linalg.norm(flat, axis=1)


def kernel_spectral_norms(weight: np.ndarray) -> np.ndarray:
    """Spectral norm of each per-output-channel convolution operator.

    For a single output channel the conv is a rank-1 map from
    R^(C·ps·ps) → R, so the spectral norm equals the L2 norm of the
    flattened kernel.  We return the same values for consistency with
    the naming convention; for multi-output spectral norm see
    :func:`layer_spectral_norm`.
    """
    return kernel_l2_norms(weight)


def layer_spectral_norm(weight: np.ndarray) -> float:
    """Largest singular value of the full [embed_dim, C·ps·ps] weight matrix."""
    flat = weight.reshape(weight.shape[0], -1)
    # Use a partial SVD for efficiency when embed_dim is large.
    s = np.linalg.svd(flat, compute_uv=False, full_matrices=False)
    return float(s[0])


def singular_value_spectrum(weight: np.ndarray) -> np.ndarray:
    """All singular values of the flattened weight matrix, sorted descending.

    Returns
    -------
    sv : np.ndarray  shape [min(embed_dim, C·ps·ps)]
    """
    flat = weight.reshape(weight.shape[0], -1)
    return np.linalg.svd(flat, compute_uv=False, full_matrices=False)


def spectral_condition_number(weight: np.ndarray) -> float:
    """Ratio of largest to smallest singular value."""
    sv = singular_value_spectrum(weight)
    return float(sv[0] / (sv[-1] + 1e-12))


def spectral_energy_fraction(
    weight: np.ndarray, cumsum_thresh: float = 0.95
) -> tuple[int, float]:
    """Number of singular values needed to capture *cumsum_thresh* of total energy.

    Returns
    -------
    n_eff : int
        Effective rank — number of singular values whose squared sum
        reaches *cumsum_thresh* of the total squared singular values.
    fraction : float
        The fraction captured by the top *n_eff* singular values.
    """
    sv = singular_value_spectrum(weight)
    sv2 = sv**2
    cumsum = np.cumsum(sv2) / sv2.sum()
    n_eff = int(np.searchsorted(cumsum, cumsum_thresh) + 1)
    return n_eff, float(cumsum[n_eff - 1])


def kernel_l1_norms(weight: np.ndarray) -> np.ndarray:
    """L1 norm (sum of absolute values) of each kernel → [embed_dim]."""
    return np.abs(weight).reshape(weight.shape[0], -1).sum(axis=1)


def kernel_frobenius_norms(weight: np.ndarray) -> np.ndarray:
    """Frobenius norm of each kernel — same as L2 for a vector."""
    return np.linalg.norm(weight.reshape(weight.shape[0], -1), axis=1)


def per_channel_norms(weight: np.ndarray) -> np.ndarray:
    """Mean kernel norm contributed by each input channel → [in_chans].

    For each input channel c, averages the L2 norm of the (ps, ps) slice
    across all output channels.
    """
    E, C, H, W = weight.shape
    # Reshape to [C, E*H*W] so we can norm over the flattened spatial+output axes.
    channel_norms = np.linalg.norm(weight.transpose(1, 0, 2, 3).reshape(C, -1), axis=1)
    return channel_norms / E


def kernel_orientation(weight: np.ndarray) -> np.ndarray:
    """Dominant orientation angle (in degrees) of each kernel.

    Computed from the eigenvectors of the spatial covariance matrix of
    each flattened kernel.  Returns angles in [0, 180).
    """
    E, C, H, W = weight.shape
    angles = np.zeros(E)
    for e in range(E):
        # Stack H/W slices for each channel and compute covariance.
        k = weight[e].reshape(C, -1)  # [C, H*W]
        cov = k.T @ k / k.shape[1]  # [H*W, H*W]
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Largest eigenvector → dominant orientation.
        v = eigvecs[:, -1][:H]  # take the H-component
        angle = math.degrees(math.atan2(v.sum(), (np.abs(v) + 1e-8).sum()))
        angle = angle % 180
        angles[e] = angle
    return angles


def kernel_sparsity(weight: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """Fraction of near-zero entries per kernel → [embed_dim]."""
    flat = weight.reshape(weight.shape[0], -1)
    return (np.abs(flat) < threshold).mean(axis=1)


def bias_statistics(bias: np.ndarray) -> dict[str, float]:
    """Descriptive statistics for the bias vector."""
    return {
        "mean": float(bias.mean()),
        "std": float(bias.std()),
        "min": float(bias.min()),
        "max": float(bias.max()),
        "median": float(np.median(bias)),
        "skewness": float(
            (((bias - bias.mean()) ** 3).mean()) / (bias.std() ** 3 + 1e-8)
        ),
        "kurtosis": float(
            ((bias - bias.mean()) ** 4).mean() / (bias.std() ** 4 + 1e-8) - 3
        ),
    }


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


def _fig_kw(**override) -> dict:
    base = {"figsize": (8, 6), "dpi": 150}
    base.update(override)
    return base


def _colorbar(ax, mappable, label: str = "") -> None:
    plt.colorbar(mappable, ax=ax, label=label, shrink=0.8)


def _show_kernel_grid(
    weight: np.ndarray, start: int, n: int, title: str, cmap: str = PALETTE_DIVERGING
) -> plt.Figure:
    """Return a figure displaying *n* kernels starting at *start* in a grid.

    Each kernel is (in_chans, ps, ps).  If in_chans == 3 we render as an
    RGB-like image (clipped to [−1, 1] after normalising each channel
    independently).  Otherwise we show the sum across input channels.
    """
    rows = math.ceil(n**0.5)
    cols = math.ceil(n / rows)
    sub = weight[start : start + n]  # [n, C, ps, ps]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    if sub.shape[1] == 3:
        for i in range(min(n, rows * cols)):
            ax = axes[i]
            ch = sub[i]  # [3, ps, ps]
            ch_normed = np.zeros_like(ch)
            for c in range(3):
                cmin, cmax = ch[c].min(), ch[c].max()
                if cmax > cmin:
                    ch_normed[c] = 2.0 * (ch[c] - cmin) / (cmax - cmin) - 1.0
                else:
                    ch_normed[c] = 0.0
            ax.imshow(ch_normed.transpose(1, 2, 0), vmin=-1, vmax=1)
            ax.set_title(f"out_ch {start + i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        summed = sub.sum(axis=1)
        norm = mcolors.Normalize(vmin=summed.min(), vmax=summed.max())
        for i in range(min(n, rows * cols)):
            ax = axes[i]
            rgb = plt.cm.get_cmap(cmap)(norm(summed[i]))
            ax.imshow(rgb)
            ax.set_title(f"out_ch {start + i}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    for j in range(min(n, rows * cols), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def plot_patch_embed(
    checkpoint_path: Path | str = CHECKPOINT,
    output_dir: Path | str = OUTPUT_DIR,
) -> Path:
    """Generate all patch-embedding visualisations and save to *output_dir*.

    Returns the path to the saved composite figure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ----------------------------------------------------------------
    weight, bias = load_patch_embed(checkpoint_path)
    weight_np, bias_np = to_numpy(weight, bias)
    E, C, ps, _ = weight_np.shape

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
    orientations = kernel_orientation(weight_np)

    # --- Build composite figure ----------------------------------------------
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*imshow.*RGB.*")
        _build_composite(
            weight_np,
            bias_np,
            l2_norms,
            l1_norms,
            sparsity,
            per_ch,
            bias_stats,
            neighbor_sim,
            layer_spec,
            sv_spectrum,
            cond_num,
            n_eff_90,
            frac_90,
            n_eff_95,
            frac_95,
            output_dir,
            E,
            C,
            ps,
        )

    # --- Also save individual diagnostic plots --------------------------------
    _save_kernel_slices(weight_np, output_dir)
    _save_orientation_histogram(orientations, output_dir)
    _save_bias_detail(bias_np, output_dir)
    _save_singular_value_plot(
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
    print("\n=== Patch Embed Summary ===")
    print(f"  Weight shape : {weight_np.shape}")
    print(f"  Bias shape   : {bias_np.shape}")
    print(f"  Layer spectral norm : {layer_spec:.4f}")
    print(f"  Singular values    : {len(sv_spectrum)} total, " f"κ = {cond_num:.1f}")
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
    print(f"  Kernel L1  — mean: {l1_norms.mean():.4f}")
    print(f"  Sparsity   — mean: {sparsity.mean():.4f}")
    print(
        f"  Bias       — μ: {bias_stats['mean']:.4f},  σ: {bias_stats['std']:.4f},  "
        f"skew: {bias_stats['skewness']:.3f},  kurt: {bias_stats['kurtosis']:.3f}"
    )
    print(f"  Per-input-channel avg norm: {per_ch}")
    print(
        f"  Neighbor cos sim — mean: {neighbor_sim['mean_pairwise_cos_sim']:.4f},  "
        f"max: {neighbor_sim['max_pairwise_cos_sim']:.4f}"
    )

    return output_dir / "patch_embed_analysis.png"


def _build_composite(
    weight: np.ndarray,
    bias: np.ndarray,
    l2_norms: np.ndarray,
    l1_norms: np.ndarray,
    sparsity: np.ndarray,
    per_ch: np.ndarray,
    bias_stats: dict,
    neighbor_sim: dict,
    layer_spec: float,
    sv_spectrum: np.ndarray,
    cond_num: float,
    n_eff_90: int,
    frac_90: float,
    n_eff_95: int,
    frac_95: float,
    output_dir: Path,
    E: int,
    C: int,
    ps: int,
) -> None:
    """Build and save the 3×3 composite analysis figure."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=150)
    fig.suptitle(
        "ViT Patch Embedding Analysis  —  "
        f"Conv[{C}→{E}, ks={ps}×{ps}]  ·  σ₁ = {layer_spec:.3f}  ·  κ = {cond_num:.0f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # 1) First 24 kernels as RGB-like visualisation
    _fig_kernels = _show_kernel_grid(
        weight, start=0, n=24, title="First 24 Kernels  (RGB-normalised)"
    )
    out_path_kernels = output_dir / "patch_embed_first_kernels.png"
    _fig_kernels.savefig(out_path_kernels, bbox_inches="tight")
    plt.close(_fig_kernels)
    print(f"Saved → {out_path_kernels}")
    axes[0, 0].set_visible(False)

    # 2) Singular value spectrum (scree plot)
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
    ax.axvline(
        n_eff_90, color="orange", linestyle="--", label=f"90% energy at n={n_eff_90}"
    )
    ax.axvline(
        n_eff_95, color="red", linestyle="--", label=f"95% energy at n={n_eff_95}"
    )
    ax.set_title("Singular Value Spectrum  (scree plot)", fontsize=11)
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

    # 3) Singular value density (histogram + KDE)
    ax = axes[0, 2]
    ax.hist(
        sv_spectrum,
        bins=50,
        density=True,
        color="#729fcf",
        edgecolor="white",
        alpha=0.7,
        label="histogram",
    )
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(sv_spectrum)
    x_range = np.linspace(sv_spectrum.min(), sv_spectrum.max(), 200)
    ax.plot(x_range, kde(x_range), color="red", linewidth=2, label="KDE")
    ax.set_title("Singular Value Distribution", fontsize=11)
    ax.set_xlabel("σᵢ")
    ax.set_ylabel("density")
    ax.axvline(
        sv_spectrum.mean(),
        color="orange",
        linestyle="--",
        label=f"mean = {sv_spectrum.mean():.3f}",
    )
    ax.legend(fontsize=9)

    # 4) Kernel L1 norm vs L2 norm (scatter)
    ax = axes[1, 0]
    ax.scatter(l2_norms, l1_norms, s=4, color="#2166ac", alpha=0.6)
    ax.set_title("L1 Norm vs L2 Norm  (per kernel)", fontsize=11)
    ax.set_xlabel("L2 norm")
    ax.set_ylabel("L1 norm")
    max_ratio = (ps * ps * C) ** 0.5
    ax.axline(
        (0, 0),
        slope=max_ratio,
        color="red",
        linestyle=":",
        label=f"max ratio = {max_ratio:.2f}",
    )
    ax.legend(fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"mean L1/L2 = {(l1_norms / (l2_norms + 1e-8)).mean():.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    # 5) Kernel L2-norm distribution
    ax = axes[1, 1]
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

    # 6) Bias trend — line plot over embedding dimension index
    ax = axes[1, 2]
    ax.plot(bias, color="#2166ac", linewidth=0.8)
    ax.set_title("Bias Trend  (per embedding dimension)", fontsize=11)
    ax.set_xlabel("embedding dimension index")
    ax.set_ylabel("bias value")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.axhline(
        bias_stats["mean"],
        color="red",
        linestyle="--",
        label=f"mean = {bias_stats['mean']:.4f}",
    )
    ax.legend(fontsize=9)

    # 7) Bias histogram
    ax = axes[2, 0]
    ax.hist(bias, bins=40, color="#ef2929", edgecolor="white", alpha=0.8)
    ax.set_title("Bias Distribution", fontsize=11)
    ax.set_xlabel("bias value")
    ax.set_ylabel("count")
    ax.axvline(
        bias_stats["mean"],
        color="blue",
        linestyle="--",
        label=f"μ = {bias_stats['mean']:.4f}",
    )
    ax.axvline(
        bias_stats["median"],
        color="orange",
        linestyle="--",
        label=f"med = {bias_stats['median']:.4f}",
    )
    ax.legend(fontsize=9)

    # 8) Per-input-channel norm contribution
    ax = axes[2, 1]
    ch_labels = [f"in_ch {i}" for i in range(C)]
    bars = ax.bar(ch_labels, per_ch, color=["#d6604b", "#edae49", "#729fcf"])
    ax.set_title("Mean Kernel Norm per Input Channel", fontsize=11)
    ax.set_ylabel("avg ‖wₑ[c, :, :]‖₂")
    for bar, v in zip(bars, per_ch):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    # 9) Neighbor cosine similarity + kernel norm box plot
    ax = axes[2, 2]
    q_bins = np.percentile(l2_norms, [0, 25, 50, 75, 100])
    q = np.digitize(l2_norms, q_bins[1:-1], right=True)  # 0..3
    box_data = [l2_norms[q == i] for i in range(4)]
    bp = ax.boxplot(box_data, tick_labels=["Q1", "Q2", "Q3", "Q4"], patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#2166ac", "#729fcf", "#ef2929", "#d6604b"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Kernel Norm by Quartile", fontsize=11)
    ax.set_ylabel("L2 norm")
    ax2 = ax.twinx()
    ax2.text(
        0.02,
        0.98,
        f"mean pairwise cos_sim: {neighbor_sim['mean_pairwise_cos_sim']:.4f}\n"
        f"median pairwise cos_sim: {neighbor_sim['median_pairwise_cos_sim']:.4f}\n"
        f"max pairwise cos_sim: {neighbor_sim['max_pairwise_cos_sim']:.4f}",
        transform=ax2.transAxes,
        va="top",
        fontsize=9,
        color="dimgray",
    )
    ax2.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = output_dir / "patch_embed_analysis.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Supplementary plots
# ---------------------------------------------------------------------------


def _save_kernel_slices(weight: np.ndarray, out: Path) -> None:
    """Per-input-channel kernel visualisation: average kernel per input channel.

    For each input channel c, computes the mean kernel across all output
    channels and displays it alongside the per-channel kernel std.
    """
    C, ps = weight.shape[1], weight.shape[2]
    mean_kernels = weight.mean(axis=0)  # [C, ps, ps]
    std_kernels = weight.std(axis=0)  # [C, ps, ps]

    n = C
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 4, rows * 4), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    fig.suptitle(
        "Per-Input-Channel Kernel Statistics  (mean & std across output channels)",
        fontsize=13,
        fontweight="bold",
    )
    for c in range(C):
        ax_mean = axes[c * 2]
        im_mean = ax_mean.imshow(
            mean_kernels[c], cmap=PALETTE_DIVERGING, aspect="equal"
        )
        ax_mean.set_title(
            f"Input ch {c}  —  mean kernel  "
            f"(μ={mean_kernels[c].mean():.4f}, σ={mean_kernels[c].std():.4f})"
        )
        ax_mean.set_xticks([])
        ax_mean.set_yticks([])
        _colorbar(ax_mean, im_mean, label="mean value")

        ax_std = axes[c * 2 + 1]
        im_std = ax_std.imshow(std_kernels[c], cmap=PALETTE_SEQUENTIAL, aspect="equal")
        ax_std.set_title(f"Input ch {c}  —  std across output ch")
        ax_std.set_xticks([])
        ax_std.set_yticks([])
        _colorbar(ax_std, im_std, label="std")

    for j in range(C * 2, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    out_path = out / "patch_embed_per_channel_slices.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _save_orientation_histogram(orientations: np.ndarray, out: Path) -> None:
    """Histogram of dominant kernel orientations."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(
        orientations,
        bins=36,
        range=(0, 180),
        color="#2166ac",
        edgecolor="white",
        alpha=0.8,
    )
    ax.set_title(
        "Dominant Kernel Orientation Distribution", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("count")
    ax.axvline(
        orientations.mean(),
        color="red",
        linestyle="--",
        label=f"mean = {orientations.mean():.1f}°",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = out / "patch_embed_orientation_hist.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _save_bias_detail(bias: np.ndarray, out: Path) -> None:
    """Detailed bias analysis: rolling mean, rolling std, and rank plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=150)
    fig.suptitle("Bias Detail Analysis", fontsize=13, fontweight="bold")

    # Rolling statistics
    window = 32
    rolling_mean = np.convolve(bias, np.ones(window) / window, mode="valid")
    rolling_std = np.array(
        [bias[max(0, i) : i + window].std() for i in range(len(bias) - window + 1)]
    )
    idx = np.arange(window // 2, window // 2 + len(rolling_mean))

    ax = axes[0, 0]
    ax.plot(idx, bias[idx], alpha=0.4, color="#2166ac", linewidth=0.5)
    ax.plot(
        idx, rolling_mean, color="red", linewidth=1.5, label=f"rolling μ (w={window})"
    )
    ax.fill_between(
        idx,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="red",
        alpha=0.2,
    )
    ax.set_title("Bias — Raw, Rolling Mean & ±1σ Band", fontsize=11)
    ax.set_xlabel("embedding dimension index")
    ax.set_ylabel("bias value")
    ax.legend(fontsize=9)

    # Rank plot (sorted values)
    ax = axes[0, 1]
    sorted_bias = np.sort(bias)
    ax.plot(sorted_bias, color="#2166ac", linewidth=1)
    ax.set_title("Bias — Rank Plot (sorted)", fontsize=11)
    ax.set_xlabel("rank")
    ax.set_ylabel("bias value")

    # Autocorrelation
    ax = axes[1, 0]
    bias_c = bias - bias.mean()
    autocorr = np.correlate(bias_c, bias_c, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr /= autocorr[0] + 1e-8
    ax.plot(autocorr[:100], color="#2166ac", linewidth=1)
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.set_title("Bias Autocorrelation (first 100 lags)", fontsize=11)
    ax.set_xlabel("lag")
    ax.set_ylabel("correlation")

    # Histogram with KDE-like density
    ax = axes[1, 1]
    ax.hist(bias, bins=40, density=True, color="#ef2929", edgecolor="white", alpha=0.7)
    # Simple Gaussian KDE overlay
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(bias)
    x_range = np.linspace(bias.min(), bias.max(), 200)
    ax.plot(x_range, kde(x_range), color="blue", linewidth=2)
    ax.set_title("Bias Density (histogram + KDE)", fontsize=11)
    ax.set_xlabel("bias value")
    ax.set_ylabel("density")

    plt.tight_layout()
    out_path = out / "patch_embed_bias_detail.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def _save_singular_value_plot(
    sv_spectrum: np.ndarray,
    layer_spec: float,
    cond_num: float,
    n_eff_90: int,
    frac_90: float,
    n_eff_95: int,
    frac_95: float,
    out: Path,
) -> None:
    """Spectral analysis: scree plot, cumulative energy, and density."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle(
        f"Patch Embed Spectral Analysis  —  σ₁ = {layer_spec:.3f}  ·  κ = {cond_num:.0f}  ·  "
        f"n={len(sv_spectrum)} singular values",
        fontsize=13,
        fontweight="bold",
    )

    # 1) Scree plot (log scale)
    ax = axes[0]
    sv_idx = np.arange(1, len(sv_spectrum) + 1)
    ax.semilogy(sv_idx, sv_spectrum, color="#2166ac", linewidth=1.2, alpha=0.8)
    ax.axvline(
        n_eff_90,
        color="orange",
        linestyle="--",
        label=f"90% at n={n_eff_90} ({frac_90:.0%})",
    )
    ax.axvline(
        n_eff_95,
        color="red",
        linestyle="--",
        label=f"95% at n={n_eff_95} ({frac_95:.0%})",
    )
    ax.set_title("Singular Value Spectrum", fontsize=11)
    ax.set_xlabel("singular value index")
    ax.set_ylabel("σᵢ (log scale)")
    ax.legend(fontsize=9)

    # 2) Cumulative energy fraction
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
    ax.annotate(
        f"90% → {n_eff_90}",
        xy=(n_eff_90, 0.90),
        xytext=(n_eff_90 + 20, 0.92),
        arrowprops=dict(arrowstyle="->", color="orange"),
        fontsize=9,
        color="orange",
    )
    ax.annotate(
        f"95% → {n_eff_95}",
        xy=(n_eff_95, 0.95),
        xytext=(n_eff_95 + 20, 0.97),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )

    # 3) Singular value density
    ax = axes[2]
    ax.hist(
        sv_spectrum,
        bins=50,
        density=True,
        color="#729fcf",
        edgecolor="white",
        alpha=0.7,
    )
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(sv_spectrum)
    x_range = np.linspace(sv_spectrum.min(), sv_spectrum.max(), 200)
    ax.plot(x_range, kde(x_range), color="red", linewidth=2)
    ax.axvline(
        sv_spectrum.mean(),
        color="orange",
        linestyle="--",
        label=f"mean = {sv_spectrum.mean():.3f}",
    )
    ax.set_title("Singular Value Density", fontsize=11)
    ax.set_xlabel("σᵢ")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = out / "patch_embed_spectral.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_patch_embed()
