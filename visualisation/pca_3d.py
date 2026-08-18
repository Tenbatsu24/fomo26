from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
from matplotlib.colors import Normalize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from visualisation.utils import sigmoid

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "understand" / "pca_3d"
DEFAULT_DATASET = "CLS002_FOMO26_Infarct"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess_volume_channels(volume: torch.Tensor) -> torch.Tensor:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    volume = volume.to(DEVICE).float()
    C, H, W, D = volume.shape
    if C == 3:
        return volume
    elif C == 1:
        return volume.expand(3, H, W, D)
    else:
        raise ValueError(f"Expected 1 or 3 input channels, got {C}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: Path | str,
    volume_size: tuple[int, int, int] = (224, 224, 128),
    patch_size: tuple[int, int, int] = (14, 14, 8),
    med_in_channels: int = 1,
) -> torch.nn.Module:
    """Build and load the 3-D ViT-S encoder from the checkpoint."""
    from med_adapt.models.base.vit3d import vitv2_3d_small

    model = vitv2_3d_small(
        volume_size=volume_size,
        volume_patch_size=patch_size,
        med_in_channels=med_in_channels,
        use_patch_decode=False,
        use_mask=False,
    ).eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_sd = ckpt["state_dict"]
    else:
        raw_sd = ckpt

    # Only keep keys prefixed with 'model.' and strip that prefix.
    model_sd = {
        k[len("model.") :]: v for k, v in raw_sd.items() if k.startswith("model.")
    }

    if not model_sd:
        model_sd = {**raw_sd}

    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    if missing:
        print(
            f"Info: missing keys (expected for pos_embed with anisotropic ps): {missing}"
        )
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(DEVICE)


def _model_forward_avg_channels(
    model: torch.nn.Module, volume: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    C = volume.shape[0]
    cls_acc: list[torch.Tensor] = []
    patch_acc: list[torch.Tensor] = []
    recon_list: list[torch.Tensor] = []

    for c in range(C):
        vol_c = volume[c : c + 1].unsqueeze(0)  # [1, 1, H, W, D]
        with torch.no_grad():
            out = model(vol_c, return_dict=True)
        cls_acc.append(out["latent"])  # [1, E]
        patch_acc.append(out["patch_latent"])  # [1, E, Ph, Pw, Pd]
        if out["recon"] is not None:
            recon_list.append(out["recon"])  # [1, 1, H, W, D]

    cls_token = torch.stack(cls_acc, dim=0).mean(dim=0)  # [1, E]
    patch_tokens = torch.stack(patch_acc, dim=0).mean(dim=0)  # [1, E, Ph, Pw, Pd]
    recon = torch.cat(recon_list, dim=0) if recon_list else None  # [C, 1, H, W, D]
    return cls_token, patch_tokens, recon


def _model_forward_all_channels(
    model: torch.nn.Module, volume: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

    vol_b = volume.unsqueeze(0)  # [1, 1, H, W, D]
    with torch.no_grad():
        out = model(vol_b, return_dict=True)
    cls_token = out["latent"]  # [1, E]
    patch_tokens = out["patch_latent"]  # [1, E, Ph, Pw, Pd]

    if out["recon"] is not None:
        recon = out["recon"].transpose(0, 1)  # [1, 3, H, W, D]

    return cls_token, patch_tokens, recon


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------


def plot_volume_pca_3d(
    checkpoint_path: Path | str,
    crop_size: tuple[int, int, int],
    resize_size: Optional[tuple[int, int, int]] = None,
    model_volume_size: Optional[tuple[int, int, int]] = None,
    dataset_name: str = DEFAULT_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_depth_slices: int = 9,
    sample_index: int = 0,
    patch_size: tuple[int, int, int] = (14, 14, 8),
    med_in_channels: int = 1,
    whiten: bool = True,
    colormap: str = "coolwarm",
    seed: int | None = None,
    fold: int | None = None,
) -> Path:
    """Run the 3-D volume → patch-token PCA pipeline and save figures."""
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
    volume = sample["image"]
    print(
        f"Volume shape: {volume.shape}, label: {sample['label'].item()}, "
        f"subject: {sample['subject']}"
    )

    vol = _preprocess_volume_channels(volume)
    print(f"Preprocessed volume: {vol.shape}")

    if model_volume_size is None:
        model_volume_size = crop_size

    model = load_model(
        checkpoint_path,
        volume_size=model_volume_size,
        patch_size=patch_size,
        med_in_channels=med_in_channels,
    )
    print(f"Model loaded, patch_size={model.patch_size}")

    # Run once per channel and average representations; keep per-channel recons.
    if med_in_channels == dataset_cls.NUM_MODALITIES:
        cls_token, patch_tokens, recon = _model_forward_all_channels(model, vol)
    elif med_in_channels == 1:
        cls_token, patch_tokens, recon = _model_forward_avg_channels(model, vol)
    else:
        raise NotImplementedError(
            "can not run when model input and dataset modalities are not divisible"
        )

    # cls_token: [1, E], patch_tokens: [1, E, Ph, Pw, Pd]

    B, E, ph, pw, pd_ = patch_tokens.shape
    print(f"Patch grid: {ph}×{pw}×{pd_} = {ph * pw * pd_} patches")

    # Select depth slices of the PATCH GRID (not the volume)
    if pd_ <= max_depth_slices:
        slice_indices = list(range(pd_))
    else:
        slice_indices = np.linspace(0, pd_ - 1, max_depth_slices, dtype=int).tolist()
    print(f"Selected patch-depth slices: {slice_indices}")

    # --- Fit PCA on the ENTIRE volume ---------------------------------------
    # patch_tokens shape: [1, E, Ph, Pw, Pd] → [Ph*Pw*Pd, E]
    flat = patch_tokens.squeeze(0).permute(3, 1, 2, 0).reshape(-1, E)
    pca = PCA(n_components=3, whiten=whiten)
    projected = pca.fit_transform(flat.cpu().numpy())  # [N_total, 3]
    print(
        f"PCA fitted on {projected.shape[0]} patches, explained variance: {pca.explained_variance_ratio_}"
    )

    # --- Extract per-slice projections --------------------------------------
    pca_images: list[np.ndarray] = []
    cosine_images: list[np.ndarray] = []

    for d_idx in slice_indices:
        start = d_idx * ph * pw
        end = start + ph * pw
        slice_proj = projected[start:end]  # [ph * pw, 3]

        # Min-max scale each component to [0, 1]
        pca_images.append(sigmoid(2 * slice_proj.reshape(ph, pw, 3)))

        # Cosine similarity to cls token
        cls_t = cls_token[0, ...]  # [1, E]
        slice_flat = patch_tokens[0, ..., d_idx]  # [E, ph, pw]
        cos_sim = F.cosine_similarity(
            slice_flat.reshape(E, ph * pw), cls_t.unsqueeze(-1), dim=0
        )
        cosine_images.append(cos_sim.cpu().numpy().reshape(ph, pw))

    n = len(pca_images)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # --- PCA-RGB panel ------------------------------------------------------
    fig_pca, axes_pca = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), dpi=150)
    axes_pca = np.asarray(axes_pca).reshape(-1)
    fig_pca.suptitle(
        "3-D ViT: Full-Volume Patch-Token PCA (whitened) per patch-depth slice",
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
    pca_path = output_dir / "volume_pca_3d_rgb.png"
    fig_pca.savefig(pca_path)
    plt.close(fig_pca)
    print(f"Saved → {pca_path}")

    # --- Cosine panel (single colour bar) -----------------------------------
    fig_cos, axes_cos = plt.subplots(
        rows, cols, figsize=(cols * 5 + 1, rows * 4), dpi=150, constrained_layout=True
    )
    axes_cos = np.asarray(axes_cos).reshape(-1)
    fig_cos.suptitle(
        "3-D ViT: Patch-token ↔ CLS cosine similarity per patch-depth slice",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    all_cos = np.concatenate(cosine_images)
    global_cmin, global_cmax = all_cos.min(), all_cos.max()

    for i, (ax, cos_img, d_idx) in enumerate(
        zip(axes_cos, cosine_images, slice_indices)
    ):
        im = ax.imshow(
            cos_img,
            cmap=colormap,
            vmin=global_cmin,
            vmax=global_cmax,
            aspect="equal",
        )
        ax.set_title(f"patch-depth z={d_idx}  ({ph}×{pw} patches)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig_cos.colorbar(im, ax=axes_cos.tolist(), shrink=0.8, label="cosine sim")
    for j in range(n, len(axes_cos)):
        axes_cos[j].set_visible(False)
    cos_path = output_dir / "volume_pca_3d_cosine.png"
    fig_cos.savefig(cos_path)
    plt.close(fig_cos)
    print(f"Saved → {cos_path}")

    # --- Reconstruction panel (only when model produces recon; no normalisation) ---
    if recon is not None:
        n_channels = recon.shape[0]  # [C, 1, H, W, D]
        ps = model.patch_size
        if isinstance(ps, int):
            ps = (ps, ps, ps)

        rec_rows = 3
        rec_cols = 3 * n_channels

        fig_rec, axes_rec = plt.subplots(
            rec_rows,
            rec_cols,
            figsize=(rec_cols * 5 + 1, rec_rows * 4),
            dpi=150,
            constrained_layout=True,
        )
        axes_rec = np.asarray(axes_rec).reshape(rec_rows, rec_cols)
        fig_rec.suptitle(
            f"3-D ViT: Reconstructions per channel  (C={n_channels})",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        all_recon_slices: list[np.ndarray] = []
        for c in range(n_channels):
            for r in range(3):
                for s in range(3):
                    d_idx = slice_indices[r * 3 + s]
                    d_start = d_idx * ps[2]
                    d_end = d_start + ps[2]
                    # Average over the depth range to get a 2-D slice
                    recon_slice = (
                        recon[c, 0, :, :, d_start:d_end].mean(dim=-1).cpu().numpy()
                    )
                    all_recon_slices.append(recon_slice)
                    # NO normalisation — reconstruction is output, not input

        global_rmin = min(s.min() for s in all_recon_slices)
        global_rmax = max(s.max() for s in all_recon_slices)

        norm = Normalize(vmin=global_rmin, vmax=global_rmax)

        for i, ax in enumerate(axes_rec.reshape(-1)):
            ax.imshow(
                all_recon_slices[i],
                cmap="gray",
                norm=norm,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # Shared colorbar
        sm = plt.cm.ScalarMappable(cmap="gray", norm=norm)
        sm.set_array([])  # for older matplotlib versions

        fig_rec.colorbar(
            sm,
            ax=axes_rec,
            shrink=0.8,
            label="recon intensity",
        )

        recon_path = output_dir / "volume_pca_3d_recon.png"
        fig_rec.savefig(recon_path)
        plt.close(fig_rec)
        print(f"Saved → {recon_path}")

    return pca_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-D ViT full-volume PCA visualisation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the 3-D ViT checkpoint.",
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
        "--volume-size",
        type=int,
        nargs=3,
        required=False,
        default=None,
        help="Volume size model was trained with as three ints: H W D.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(14, 14, 8),
        help="Volume patch size as three ints (default: 14 14 8).",
    )
    parser.add_argument(
        "--med-in-channels",
        type=int,
        default=1,
        help="Number of input channels for the 3-D model (default: 1).",
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
        "--max-depth-slices",
        type=int,
        default=9,
        help="Maximum patch-depth slices to show (default: 9).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    plot_volume_pca_3d(
        checkpoint_path=args.checkpoint,
        crop_size=tuple(args.crop_size),
        model_volume_size=(
            tuple(args.volume_size) if args.volume_size is not None else None
        ),
        resize_size=tuple(args.resize_size) if args.resize_size is not None else None,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_depth_slices=args.max_depth_slices,
        sample_index=args.sample_index,
        patch_size=tuple(args.patch_size),
        med_in_channels=args.med_in_channels,
        whiten=args.whiten,
        colormap=args.colormap,
        seed=args.seed,
        fold=args.fold,
    )


if __name__ == "__main__":
    main()
