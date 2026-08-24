from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from visualisation.utils import sigmoid

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "understand" / "pca_2d"
DEFAULT_DATASET = "CLS002_FOMO26_Infarct"


def preprocess_volume(volume: torch.Tensor) -> torch.Tensor:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    volume = volume.to(DEVICE).float()
    C, H, W, D = volume.shape

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1, 1)

    if C == 3:
        vol = volume
    elif C == 1:
        vol = volume.expand(3, H, W, D)
    else:
        raise ValueError(f"Expected 1 or 3 input channels, got {C}")

    ch_min = vol.amin(dim=(0, 1, 2), keepdim=True)
    ch_max = vol.amax(dim=(0, 1, 2), keepdim=True)

    denom = ch_max - ch_min
    denom = torch.where(denom == 0, 1.0, denom)

    vol = (vol - ch_min) / denom

    vol = (vol - mean) / std
    return vol


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path | str) -> torch.nn.Module:
    """Build and load the 2-D ViT-S encoder from the checkpoint."""
    from med_adapt.models.base.vit2d import vitv2_small

    model = vitv2_small(img_size=518, patch_size=14)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    filtered = {k: v for k, v in ckpt.items() if not k.startswith("projection_head")}
    model.load_state_dict(filtered, strict=False)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE).eval()
    return model


def get_patch_tokens(
    model: torch.nn.Module, slice_2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one 2-D slice through the model and return patch + cls tokens.

    Returns
    -------
    patch_tokens : torch.Tensor  shape [1, D, h_p, w_p]
    cls_token    : torch.Tensor  shape [1, D]
    """
    slice_2d = slice_2d.unsqueeze(0)  # [1, 3, H, W]
    with torch.no_grad():
        out = model(slice_2d, return_dict=True)
    patch_tokens = out["patch_latent"]  # [1, D, h_p, w_p]
    cls_token = out["latent"]  # [1, D]
    return patch_tokens, cls_token


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def plot_volume_pca(
    checkpoint_path: Path | str,
    crop_size: tuple[int, int, int],
    resize_size: Optional[tuple[int, int, int]] = None,
    dataset_name: str = DEFAULT_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_slices: int = 9,
    sample_index: int = 0,
    whiten: bool = True,
    colormap: str = "coolwarm",
    seed: int | None = None,
    fold: int | None = None,
) -> Path:
    from torchvision.transforms import Compose

    from med_adapt.registry import STORE
    from med_adapt.augs import PadToShape3D, CenterCrop3D

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trnsfrms = Compose(
        [
            PadToShape3D(size=crop_size, label_key=None),
            CenterCrop3D(size=crop_size, label_key=None),
        ]
    )

    # --- Load dataset --------------------------------------------------------
    dataset_cls = STORE.get("datasets", dataset_name)
    dataset = dataset_cls(
        root=Path(__file__).resolve().parents[1] / "data",
        fold=fold,
        seed=seed,
        n_splits=5,
        resample_spacing="median",
        resize_to=resize_size,
        transform=trnsfrms,
    )
    sample = dataset[sample_index]
    volume = sample["image"]  # [C, H, W, D]
    print(
        f"Volume shape: {volume.shape}, label: {sample['label'].item()}, "
        f"subject: {sample['subject']}"
    )

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
    model = load_model(checkpoint_path)
    print(f"Model loaded, patch_size={model.patch_size}")

    # --- Process each slice, collect patch tokens ----------------------------
    all_patch_tokens: list[torch.Tensor] = []
    all_cls_tokens: list[torch.Tensor] = []

    for d in slice_indices:
        slice_2d = vol[:, :, :, d]  # [3, H, W]
        pt, ct = get_patch_tokens(model, slice_2d)
        all_patch_tokens.append(pt)  # [1, E, h_p, w_p]
        all_cls_tokens.append(ct)  # [1, E]

    # Stack across depth: [1, d_total, h_p, w_p]
    all_patch_tokens = torch.stack(all_patch_tokens, dim=-1).squeeze(
        0
    )  # [E, h_p, w_p, d]
    all_cls_tokens = torch.stack(all_cls_tokens, dim=-1).squeeze(0)  # [E, d]
    E, h_p, w_p, d_total = all_patch_tokens.shape
    print(f"Patch grid per slice: {h_p}×{w_p}, total slices: {d_total}")

    # --- Fit PCA on the ENTIRE volume ---------------------------------------
    # Flatten spatial dims: [h_p * w_p * d_total, E]
    flat = all_patch_tokens.permute(3, 1, 2, 0).reshape(-1, E)

    # --- Extract per-slice projections --------------------------------------
    pca_images: list[np.ndarray] = []
    cosine_images: list[np.ndarray] = []

    pca = PCA(n_components=3, whiten=whiten)
    slice_projs = pca.fit_transform(flat.cpu().numpy())  # [N_total, 3]
    print(
        f"PCA fitted on {slice_projs.shape[0]} patches, explained variance: {pca.explained_variance_ratio_}"
    )
    for slice_idx, d in enumerate(slice_indices):
        start = slice_idx * h_p * w_p
        end = start + h_p * w_p
        slice_proj = slice_projs[start:end]

        # Min-max scale each component to [0, 1]
        pca_images.append(sigmoid(2 * slice_proj.reshape(h_p, w_p, 3)))

        # Cosine similarity to the corresponding cls token
        cls_t = all_cls_tokens[:, slice_idx]  # [E]
        slice_flat = all_patch_tokens[..., slice_idx].flatten(1)  # [E, h_p, w_p]
        cos_sim = F.cosine_similarity(slice_flat, cls_t.unsqueeze(-1), dim=0)
        cosine_images.append(cos_sim.cpu().numpy().reshape(h_p, w_p))

    # --- Build composite figure (PCA-RGB panel) -----------------------------
    n = len(pca_images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig_pca, axes_pca = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=150)
    axes_pca = np.asarray(axes_pca).reshape(-1)
    for i, (ax, pca_img, depth) in enumerate(zip(axes_pca, pca_images, slice_indices)):
        ax.imshow(pca_img, vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, len(axes_pca)):
        axes_pca[j].set_visible(False)
    pca_path = output_dir / "volume_pca_rgb.png"
    fig_pca.savefig(pca_path)
    plt.close(fig_pca)
    print(f"Saved → {pca_path}")

    # --- Build composite figure (cosine-sim panel, single colour bar) --------
    fig_cos, axes_cos = plt.subplots(
        rows, cols, figsize=(cols * 5 + 1, rows * 4), dpi=150, constrained_layout=True
    )
    axes_cos = np.asarray(axes_cos).reshape(-1)
    # Compute global vmin/vmax for a single shared colour bar
    all_cos = np.concatenate(cosine_images)
    global_cmin, global_cmax = all_cos.min(), all_cos.max()

    for i, (ax, cos_img, depth) in enumerate(
        zip(axes_cos, cosine_images, slice_indices)
    ):
        im = ax.imshow(
            cos_img,
            cmap=colormap,
            vmin=global_cmin,
            vmax=global_cmax,
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    # One colour bar for the entire figure
    fig_cos.colorbar(im, ax=axes_cos.tolist(), shrink=0.8, label="cosine sim")
    for j in range(n, len(axes_cos)):
        axes_cos[j].set_visible(False)
    cos_path = output_dir / "volume_pca_cosine.png"
    fig_cos.savefig(cos_path)
    plt.close(fig_cos)
    print(f"Saved → {cos_path}")

    return pca_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2-D ViT full-volume PCA visualisation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the 2-D ViT checkpoint.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=3,
        required=True,
        help="Target crop size as three ints: H W D.",
    )
    parser.add_argument(
        "--resize-size",
        type=int,
        nargs=3,
        required=False,
        default=None,
        help="Resize images to as three ints: H W D.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Dataset seed (default: None).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Dataset fold (default: None).",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="coolwarm",
        help="Colormap for cosine-sim panel (default: coolwarm).",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        default=True,
        help="Whiten PCA components (default: True).",
    )
    parser.add_argument(
        "--no-whiten",
        action="store_false",
        dest="whiten",
        help="Do not whiten PCA components.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the sample to visualise (default: 0).",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=9,
        help="Maximum number of depth slices to show (default: 9).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    plot_volume_pca(
        checkpoint_path=args.checkpoint,
        crop_size=tuple(args.crop_size),
        resize_size=tuple(args.resize_size) if args.resize_size is not None else None,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_slices=args.max_slices,
        sample_index=args.sample_index,
        whiten=args.whiten,
        colormap=args.colormap,
        seed=args.seed,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()
