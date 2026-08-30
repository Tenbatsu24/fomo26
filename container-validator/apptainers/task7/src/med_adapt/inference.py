import os
import re
import logging

from pathlib import Path

import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

from torch import nn

logger = logging.getLogger(__name__)


RESIZE: tuple[int, int, int] = (264, 320, 24)


MODEL_DIR = Path(os.environ.get("MED_ADAPT_MODEL_DIR", "/app/models"))

FOLD_RE = re.compile(r"^fold_(\d+)$")


# -------------------------------------------------------------------------
# Model construction
# -------------------------------------------------------------------------


def build_model(device) -> nn.Module:
    from med_adapt.models import vitv2_a_3d_small

    return vitv2_a_3d_small(
        mea=not (device == torch.device("cpu")),
        n_modalities=3,
        classes=2,
        task="classification",
    )


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


def preprocess_volume(path: Path) -> np.ndarray:
    """Load, resize, percentile-clip, and z-score one modality."""
    image = load_nifti(path)

    image = resize_volume(
        image,
        size=RESIZE,
    )

    image = zscore_normalize(
        image,
        lower_percentile=0.5,
        upper_percentile=99.5,
    )

    return image


def prepare_input(
    flair: Path,
    adc: Path,
    dwi: Path,
) -> torch.Tensor:
    """
    Prepare the model input.

    Channel order is fixed as:

        1: ADC
        2: DWI
        3: FLAIR

    Returns:
        Tensor of shape:

            [1, 3, *RESIZE]
    """
    modalities = [
        preprocess_volume(adc),
        preprocess_volume(dwi),
        preprocess_volume(flair),
    ]

    tensor = torch.stack(
        [torch.from_numpy(x) for x in modalities],
        dim=0,
    )

    # Add batch dimension.
    tensor = tensor.unsqueeze(0)

    return tensor.contiguous()


# -------------------------------------------------------------------------
# Fold / checkpoint discovery
# -------------------------------------------------------------------------


def find_fold_directories(models_dir: Path) -> list[Path]:
    """
    Find fold_N directories and sort them numerically.

    Missing fold numbers are fine. For example:

        fold_0
        fold_1
        fold_3
        fold_4

    will result in four ensemble members.
    """
    if not models_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {models_dir}")

    folds: list[tuple[int, Path]] = []

    for path in models_dir.iterdir():
        if not path.is_dir():
            continue

        match = FOLD_RE.match(path.name)
        if match is None:
            continue

        fold_number = int(match.group(1))
        folds.append((fold_number, path))

    if not folds:
        raise RuntimeError(f"No fold_N directories found under {models_dir}")

    folds.sort(key=lambda item: item[0])

    return [path for _, path in folds]


def find_checkpoint(fold_dir: Path) -> Path:
    """
    Return the single checkpoint in a fold.

    The deployment layout is expected to contain exactly one .ckpt per fold.
    """
    checkpoints = sorted(fold_dir.glob("*.ckpt"))

    if len(checkpoints) == 0:
        raise FileNotFoundError(f"No .ckpt file found in {fold_dir}")

    if len(checkpoints) > 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint in {fold_dir}, "
            f"found {len(checkpoints)}: "
            f"{[p.name for p in checkpoints]}"
        )

    return checkpoints[0]


# -------------------------------------------------------------------------
# Checkpoint loading
# -------------------------------------------------------------------------


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

    model = build_model(device)

    # For inference deployment I prefer strict=True. A partially loaded
    # checkpoint should fail rather than silently produce incorrect results.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.requires_grad_(False)
    model.eval()
    model.to(device)

    return model


# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------


def output_to_probability(output: object) -> float:
    """
    Convert classifier output into positive-class probability.

    Supported output shapes:

        [1, 2]  -> softmax, class index 1
        [2]     -> softmax, class index 1
        [1, 1]  -> sigmoid
        [1]     -> sigmoid

    A list/tuple output uses the final element, matching the behavior of the
    previous inference implementation.
    """
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Model returned an empty output list.")
        output = output[-1]

    if isinstance(output, dict):
        if "logits" in output:
            output = output["logits"]
        elif "output" in output:
            output = output["output"]
        else:
            raise TypeError(
                f"Cannot determine logits from model output keys: " f"{list(output)}"
            )

    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected model output to be a Tensor, got {type(output)!r}")

    # Remove batch dimension for one-sample inference.
    if output.ndim == 2:
        if output.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got output shape {tuple(output.shape)}"
            )
        output = output[0]

    if output.ndim == 0:
        probability = torch.sigmoid(output)

    elif output.ndim == 1 and output.numel() == 1:
        probability = torch.sigmoid(output[0])

    elif output.ndim == 1 and output.numel() == 2:
        probability = torch.softmax(output, dim=0)[1]

    else:
        raise ValueError(
            "Unsupported classifier output shape: " f"{tuple(output.shape)}"
        )

    return float(probability.detach().cpu().item())


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


@torch.inference_mode()
def run_ensemble_inference(
    image: torch.Tensor,
    models_dir: Path,
    device: torch.device,
) -> tuple[float, list[float]]:
    """
    Run every available fold sequentially and average probabilities.
    """
    fold_dirs = find_fold_directories(models_dir)

    image = image.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    probabilities: list[float] = []

    for fold_dir in fold_dirs:
        checkpoint_path = find_checkpoint(fold_dir)

        logger.info(
            "Running %s using %s",
            fold_dir.name,
            checkpoint_path.name,
        )

        model = load_model(
            checkpoint_path=checkpoint_path,
            device=device,
        )

        output = model(image)

        probability = output_to_probability(output)
        probabilities.append(probability)

        # Only one fold needs to reside on the GPU at a time.
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not probabilities:
        raise RuntimeError("No fold predictions were produced.")

    ensemble_probability = float(np.mean(probabilities))

    return ensemble_probability, probabilities


def predict_case(
    flair: Path,
    adc: Path,
    dwi: Path,
    models_dir: Path = MODEL_DIR,
) -> tuple[float, list[float]]:
    """
    Complete inference pipeline for one subject.
    """
    image = prepare_input(
        flair=flair,
        adc=adc,
        dwi=dwi,
    )

    device = get_device()

    return run_ensemble_inference(
        image=image,
        models_dir=models_dir,
        device=device,
    )
