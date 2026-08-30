import os
import re
import logging

from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from torch import nn
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger(__name__)


RESIZE: tuple[int, int, int] = (256, 256, 256)


MODEL_DIR = Path(os.environ.get("MED_ADAPT_MODEL_DIR", "/app/models"))

FOLD_RE = re.compile(r"^fold_(\d+)$")


# -------------------------------------------------------------------------
# Model construction
# -------------------------------------------------------------------------


def build_model() -> nn.Module:
    from med_adapt.models import vitv2_3d_small

    return vitv2_3d_small()


# -------------------------------------------------------------------------
# Image loading / preprocessing
# -------------------------------------------------------------------------


def load_nifti(path: Path) -> np.ndarray:
    """Load a 3D NIfTI image as float32."""
    if not path.exists():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    image = nib.load(str(path))
    array = np.asarray(image.get_fdata(dtype=np.float32))

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D image at {path}, got shape {array.shape}")

    return array


def make_foreground_mask(
    image: np.ndarray,
    percentile_threshold: float = 5.0,
) -> np.ndarray:
    """
    Construct the foreground mask from the original image intensities.

    If the minimum finite intensity is exactly zero:

        foreground = image != 0

    Otherwise there is no explicit zero-valued background, so the bottom
    5 percent of the intensity distribution is treated as background:

        foreground = image > P5

    Non-finite voxels are always considered background.
    """
    image = np.asarray(image, dtype=np.float32)

    finite_mask = np.isfinite(image)

    if not finite_mask.any():
        raise ValueError("Image contains no finite voxels.")

    finite_values = image[finite_mask]
    minimum = float(finite_values.min())

    if minimum == 0.0:
        foreground = image != 0.0
    else:
        threshold = float(
            np.percentile(
                finite_values,
                percentile_threshold,
            )
        )
        foreground = image > threshold

    foreground &= finite_mask

    return foreground


def resize_volume(
    image: np.ndarray,
    size: tuple[int, int, int],
) -> np.ndarray:
    """
    Resize a 3D image.

    Downsampling is performed using area interpolation.

    If one or more dimensions need upsampling, resizing is done in two
    stages:

      1. shrink dimensions using area interpolation;
      2. enlarge remaining dimensions using trilinear interpolation.

    This means dimensions that genuinely require downsampling still use
    area interpolation even for mixed up/down resize operations.
    """
    source_size = tuple(int(v) for v in image.shape)
    target_size = tuple(int(v) for v in size)

    if source_size == target_size:
        return image.astype(np.float32, copy=False)

    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]

    # First perform any required downsampling.
    intermediate_size = tuple(
        min(src, dst) for src, dst in zip(source_size, target_size)
    )

    if intermediate_size != source_size:
        tensor = F.interpolate(
            tensor,
            size=intermediate_size,
            mode="area",
        )

    # Then perform any required upsampling.
    if intermediate_size != target_size:
        tensor = F.interpolate(
            tensor,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )

    return tensor[0, 0].numpy()


def resize_mask(
    mask: np.ndarray,
    size: tuple[int, int, int],
) -> np.ndarray:
    """
    Resize a binary foreground mask with nearest-neighbor interpolation.
    """
    source_size = tuple(int(v) for v in mask.shape)
    target_size = tuple(int(v) for v in size)

    if source_size == target_size:
        return mask.astype(bool, copy=False)

    tensor = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))[None, None]

    tensor = F.interpolate(
        tensor,
        size=target_size,
        mode="nearest",
    )

    return tensor[0, 0].numpy() > 0.5


def zscore_normalize(
    image: np.ndarray,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Clip intensities to the 0.5th / 99.5th percentiles and z-score.

    Processing:

        clipped = clip(image, P0.5, P99.5)
        normalized = (clipped - mean) / std

    Non-finite voxels are excluded from percentile/statistic calculation and
    set to zero afterward.
    """
    image = np.asarray(image, dtype=np.float32)

    finite_mask = np.isfinite(image)

    if not finite_mask.any():
        raise ValueError("Image contains no finite voxels.")

    finite_values = image[finite_mask]

    lower, upper = np.percentile(
        finite_values,
        [lower_percentile, upper_percentile],
    )

    clipped = np.clip(image, lower, upper)

    values = clipped[finite_mask]

    mean = float(values.mean())
    std = float(values.std())

    if std < eps:
        logger.warning(
            "Image standard deviation is approximately zero; "
            "returning zero-normalized image."
        )
        normalized = np.zeros_like(clipped, dtype=np.float32)
    else:
        normalized = (clipped - mean) / std

    normalized[~finite_mask] = 0.0

    return normalized.astype(np.float32, copy=False)


def preprocess_volume(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    image = load_nifti(path)

    print(
        f"[preprocess] Loaded {path.name}: "
        f"shape={image.shape}, "
        f"min={np.nanmin(image):.4f}, "
        f"max={np.nanmax(image):.4f}"
    )

    foreground_mask = make_foreground_mask(image)

    foreground_mask = binary_fill_holes(
        foreground_mask,
    ).astype(bool)

    foreground_voxels = int(foreground_mask.sum())
    total_voxels = int(foreground_mask.size)

    print(
        f"[preprocess] Original foreground: "
        f"{foreground_voxels}/{total_voxels} voxels "
        f"({100.0 * foreground_voxels / total_voxels:.2f}%)"
    )

    image = resize_volume(
        image,
        size=RESIZE,
    )

    foreground_mask = resize_mask(
        foreground_mask,
        size=RESIZE,
    )

    resized_foreground_voxels = int(foreground_mask.sum())
    resized_total_voxels = int(foreground_mask.size)

    print(
        f"[preprocess] Resized to {RESIZE}; foreground: "
        f"{resized_foreground_voxels}/{resized_total_voxels} voxels "
        f"({100.0 * resized_foreground_voxels / resized_total_voxels:.2f}%)"
    )

    image = zscore_normalize(
        image,
        lower_percentile=0.5,
        upper_percentile=99.5,
    )

    print(
        f"[preprocess] Normalized image: "
        f"mean={image.mean():.4f}, "
        f"std={image.std():.4f}, "
        f"min={image.min():.4f}, "
        f"max={image.max():.4f}"
    )

    return image, foreground_mask


def prepare_input(
    nifti: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare the model image and foreground mask.

    Returns:
        image:
            Tensor with shape:

                [1, 1, *RESIZE]

            dtype=torch.float32

        image_mask:
            Boolean tensor with shape:

                [1, 1, *RESIZE]

            True denotes foreground voxels that the final CLS pooling is
            allowed to use.
    """
    image, foreground_mask = preprocess_volume(nifti)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

    mask_tensor = (
        torch.from_numpy(np.ascontiguousarray(foreground_mask))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return (
        image_tensor.contiguous(),
        mask_tensor.bool().contiguous(),
    )


# -------------------------------------------------------------------------
# Checkpoint loading
# -------------------------------------------------------------------------


def find_checkpoint(model_dir: Path) -> Path:
    """
    Return the single checkpoint in a fold.

    The deployment layout is expected to contain exactly one .ckpt per fold.
    """
    checkpoints = sorted(model_dir.glob("*.ckpt"))

    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No .ckpt file found in {model_dir}")

    if len(checkpoints) > 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {model_dir}, "
            f"found {len(checkpoints)}: "
            f"{[p.name for p in checkpoints]}"
        )

    return checkpoints[0]


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """
    Extract the model state dict from either:

      * a Lightning-style checkpoint containing ``state_dict``;
      * a raw PyTorch state dict.

    Lightning checkpoints commonly prefix model parameters with ``model.``.
    That prefix is stripped when present.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(checkpoint)!r}")

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict is not a dictionary.")

    if any(key.startswith("model.") for key in state_dict):
        state_dict = {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

    return state_dict


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Construct and load one fold model.
    """
    # Load weights on CPU first so loading several ensemble members
    # sequentially does not unnecessarily consume GPU memory.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = extract_state_dict(checkpoint)

    model = build_model()

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=True,
    )
    print(f"missing: {len(missing)} - [{missing}]")
    print(f"unexpected: {len(unexpected)} - [{unexpected}]")

    model.requires_grad_(False)
    model.eval()
    model.to(device)

    return model


def get_device() -> torch.device:
    """
    This deployment is intended for CUDA inference.
    """
    if not torch.cuda.is_available():
        print(
            "CUDA is not available. Run the Apptainer image with --nv "
            "and ensure a compatible NVIDIA driver is installed on the host."
        )
        return torch.device("cpu")

    return torch.device("cuda")


# -------------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------------


@torch.inference_mode()
def run_ensemble_inference(
    image: torch.Tensor,
    image_mask: torch.Tensor,
    models_dir: Path,
    device: torch.device,
) -> np.ndarray:
    """
    Run the model and return its final masked CLS representation.
    """
    image = image.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    image_mask = image_mask.to(
        device=device,
        dtype=torch.bool,
        non_blocking=True,
    )

    checkpoint_path = find_checkpoint(models_dir)

    model = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    output = model(
        image,
        image_mask=image_mask,
        return_dict=True,
    ).squeeze(0)

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output.detach().cpu().numpy()


def predict_case(
    nifti: Path,
    models_dir: Path = MODEL_DIR,
) -> np.ndarray:
    """
    Complete inference pipeline for one subject.
    """
    image, image_mask = prepare_input(nifti)

    device = get_device()

    return run_ensemble_inference(
        image=image,
        image_mask=image_mask,
        models_dir=models_dir,
        device=device,
    )
