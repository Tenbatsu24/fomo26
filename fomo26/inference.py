"""Sliding-window inference utilities for large-volume segmentation.

Provides ``sliding_window_predict`` that tiles a volume into overlapping
patches, runs each through the model, and recombines with cosine overlap
weighting to avoid boundary artifacts.
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def _cosine_overlap_weight(
    patch_shape: Tuple[int, ...], overlap: Tuple[int, ...]
) -> torch.Tensor:
    """Return a cosine-weighted mask for a patch of the given shape.

    The weight is 1.0 in the centre and tapers to 0.0 at the edges according
    to a half cosine ramp over the ``overlap`` region on each side.
    """
    ndim = len(patch_shape)
    ramp = torch.ones(1, *patch_shape)

    for dim, o in enumerate(overlap):
        if o <= 0:
            continue
        half = o // 2
        if half == 0:
            continue
        ramp_1d = torch.ones(1, *patch_shape, device=ramp.device, dtype=ramp.dtype)
        # Left edge
        slices: list = [slice(None)] * ndim
        slices[dim] = slice(0, half)
        idx = torch.arange(half, device=ramp.device)
        ramp_1d[tuple(slices)] = 0.5 * (1.0 - torch.cos(torch.pi * (idx + 1) / (2 * half)))
        # Right edge
        slices[dim] = slice(-half, None)
        idx = torch.arange(half, device=ramp.device)
        ramp_1d[tuple(slices)] = 0.5 * (1.0 - torch.cos(torch.pi * (idx + 1) / (2 * half)))
        ramp = ramp * ramp_1d

    return ramp


def _generate_patches(
    volume_shape: Tuple[int, ...],
    patch_shape: Tuple[int, ...],
    overlap: Tuple[int, ...],
) -> Tuple[list, list]:
    """Return (starts, ends) tuples for a sliding-window grid."""
    starts, ends = [], []
    ndim = len(patch_shape)
    strides = tuple(max(1, ps - ol) for ps, ol in zip(patch_shape, overlap))

    # Iterative Cartesian product over valid start positions per dimension
    ranges = [range(0, max(1, volume_shape[d] - patch_shape[d] + 1), strides[d])
              for d in range(ndim)]

    if ndim == 2:
        for i0 in ranges[0]:
            for i1 in ranges[1]:
                idx = [i0, i1]
                end = tuple(s + p for s, p in zip(idx, patch_shape))
                end_clamped = tuple(min(e, v) for e, v in zip(end, volume_shape))
                if all(e > s for s, e in zip(idx, end_clamped)):
                    starts.append(tuple(idx))
                    ends.append(end_clamped)
    elif ndim == 3:
        for i0 in ranges[0]:
            for i1 in ranges[1]:
                for i2 in ranges[2]:
                    idx = [i0, i1, i2]
                    end = tuple(s + p for s, p in zip(idx, patch_shape))
                    end_clamped = tuple(min(e, v) for e, v in zip(end, volume_shape))
                    if all(e > s for s, e in zip(idx, end_clamped)):
                        starts.append(tuple(idx))
                        ends.append(end_clamped)
    else:
        def _recurse(d, current):
            if d == ndim:
                end = tuple(s + p for s, p in zip(current, patch_shape))
                end_clamped = tuple(min(e, v) for e, v in zip(end, volume_shape))
                if all(e > s for s, e in zip(current, end_clamped)):
                    starts.append(tuple(current))
                    ends.append(end_clamped)
                return
            for v in ranges[d]:
                _recurse(d + 1, current + [v])
        _recurse(0, [])

    return starts, ends


def sliding_window_predict(
    model: torch.nn.Module,
    volume: torch.Tensor,
    patch_size: Tuple[int, ...],
    overlap: Optional[Tuple[int, ...]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    amp: bool = False,
) -> torch.Tensor:
    """Run *model* on *volume* using a sliding-window strategy.

    The input volume is tiled into overlapping patches. Each patch is passed
    through the model and the results are averaged using cosine overlap
    weighting. This avoids boundary discontinuities between patches.

    Args:
        model: a PyTorch module whose forward returns logits of shape
               ``(B, C, *)`` where ``*`` matches the patch spatial size.
        volume: input volume of shape ``(B, C, D, H, W)`` (5-D).
        patch_size: spatial patch size. For 3-D models this is
            ``(ph, pw, pd)``. For 2-D models (that fold depth into batch)
            this is ``(ph, pw)`` — depth is processed slice-by-slice.
        overlap: overlap in voxels on each spatial dimension. Defaults to
                 one quarter of the patch size per dimension.
        device: target device. Defaults to ``volume.device``.
        batch_size: process this many patches at once.
        amp: enable automatic mixed precision.

    Returns:
        Full-volume logits of shape ``(B, C_out, D, H, W)``.
    """
    device = device or volume.device
    model = model.to(device).eval()
    B, C_in, D, H, W = volume.shape

    if overlap is None:
        overlap = tuple(max(1, ps // 4) for ps in patch_size)

    # Determine whether this is a 2-D or 3-D model
    is_2d = hasattr(model, "patch_size") and not hasattr(model, "volume_patch_size")

    if is_2d:
        # 2-D model: slide over H, W; process each depth slice separately
        ph, pw = patch_size
        spatial_vol = (H, W)
        spatial_patch = (ph, pw)
        spatial_overlap = overlap[:2] if len(overlap) >= 2 else (overlap[0], overlap[0])
    else:
        # 3-D model
        ph, pw, pd = patch_size[0], patch_size[1], patch_size[2]
        spatial_vol = (D, H, W)
        spatial_patch = (ph, pw, pd)
        spatial_overlap = overlap[:3] if len(overlap) >= 3 else (overlap[0], overlap[1], overlap[2])

    starts, ends = _generate_patches(spatial_vol, spatial_patch, spatial_overlap)
    LOGGER.info("Sliding window: %d patches over %s (2d=%s)", len(starts), spatial_vol, is_2d)

    if len(starts) == 0:
        LOGGER.warning("No patches generated; falling back to full-volume forward.")
        with torch.no_grad(), torch.amp.autocast(
            "cuda" if device.type == "cuda" else "cpu", enabled=amp
        ):
            return model(volume.to(device))

    # Pre-compute cosine weight mask
    weight_mask = _cosine_overlap_weight(spatial_patch, spatial_overlap).to(device)

    output_acc = torch.zeros(B, C_in, *spatial_vol, device=device, dtype=torch.float32)
    weight_acc = torch.zeros(B, 1, *spatial_vol, device=device, dtype=torch.float32)

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        batch_ends = ends[batch_start : batch_start + batch_size]
        cur_batch_size = len(batch_indices)

        patch_tensors = []
        for si, ei in zip(batch_indices, batch_ends):
            if is_2d:
                # 2-D: extract (B, C, D, h, w) slice region
                s0, s1 = si
                e0, e1 = ei
                patch = volume[:, :, :, s0:e0, s1:e1]  # (B, C, D, h, w)
            else:
                # 3-D: extract (B, C, d, h, w) volume region
                s0, s1, s2 = si
                e0, e1, e2 = ei
                patch = volume[:, :, s0:e0, s1:e1, s2:e2]  # (B, C, d, h, w)

            # Pad to full patch size if at boundary
            ph_p, pw_p = spatial_patch[0], spatial_patch[1]
            pd_p = spatial_patch[2] if not is_2d else 1
            ph_act = e0 - s0
            pw_act = e1 - s1
            pd_act = e2 - s2 if not is_2d else 1

            padded = torch.zeros(
                cur_batch_size, C_in, pd_p, ph_p, pw_p,
                device=device, dtype=volume.dtype,
            )
            padded[:, :, :pd_act, :ph_act, :pw_act] = patch
            patch_tensors.append(padded)

        patch_tensor = torch.cat(patch_tensors, dim=0)  # (cur_batch_size, C, pd, ph, pw)

        with torch.no_grad(), torch.amp.autocast(
            "cuda" if device.type == "cuda" else "cpu", enabled=amp
        ):
            logits = model(patch_tensor)

        # logits: (cur_batch_size, C_out, pd', ph', pw')
        for i, (si, ei) in enumerate(zip(batch_indices, batch_ends)):
            logit_patch = logits[i : i + 1]

            if is_2d:
                # Weight mask is 2-D; expand to 3-D by replicating over depth
                wm = weight_mask.unsqueeze(2)  # (1, ph, pw, 1) -> (1, ph, pw, 1) wait, wrong shape
                # Actually weight_mask is (1, ph, pw) for 2D
                wm_3d = weight_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, ph, pw)
                # Crop to actual patch size
                wm_crop = wm_3d[:, :, : logit_patch.shape[3], : logit_patch.shape[4]]
                out_slices = [
                    slice(None), slice(None),
                    slice(si[0], ei[0]),   # D
                    slice(si[1], ei[1]),   # H
                    slice(si[2], ei[2]),   # W  -- but wait, for 2D si has 2 elements
                ]
                # Fix: for 2D, si=(sH, sW), ei=(eH, eW)
                out_slices = [
                    slice(None), slice(None),
                    slice(None),  # full depth
                    slice(si[0], ei[0]),
                    slice(si[1], ei[1]),
                ]
            else:
                wm_crop = weight_mask[
                    :, : logit_patch.shape[2], : logit_patch.shape[3], : logit_patch.shape[4]
                ]
                out_slices = [
                    slice(None), slice(None),
                    slice(si[0], ei[0]),
                    slice(si[1], ei[1]),
                    slice(si[2], ei[2]),
                ]

            # Crop logits to actual valid region
            if is_2d:
                logit_crop = logit_patch[:, :, :, : wm_crop.shape[3], : wm_crop.shape[4]]
            else:
                logit_crop = logit_patch[:, :, : wm_crop.shape[2], : wm_crop.shape[3], : wm_crop.shape[4]]

            output_acc[tuple(out_slices)] += logit_crop
            weight_acc[tuple(out_slices)] += wm_crop.unsqueeze(0).unsqueeze(0)

    safe_weight = weight_acc.clamp(min=1e-8)
    predictions = output_acc / safe_weight
    return predictions
