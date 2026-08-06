"""Sliding-window inference utilities for large-volume segmentation.

Provides ``sliding_window_predict`` that tiles a volume into overlapping
patches, runs each through the model, and recombines with cosine overlap
weighting to avoid boundary artifacts.
"""

from typing import Optional, Tuple

import torch

from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


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
        ramp_1d[tuple(slices)] = 0.5 * (
            1.0 - torch.cos(torch.pi * (idx + 1) / (2 * half))
        )
        # Right edge
        slices[dim] = slice(-half, None)
        idx = torch.arange(half, device=ramp.device)
        ramp_1d[tuple(slices)] = 0.5 * (
            1.0 - torch.cos(torch.pi * (idx + 1) / (2 * half))
        )
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

    ranges = [
        range(0, max(1, volume_shape[d] - patch_shape[d] + 1), strides[d])
        for d in range(ndim)
    ]

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

    For 2-D models (that fold depth into the batch internally), the window
    slides over H and W only; each patch keeps the full depth dimension.
    For 3-D models, the window slides over D, H, and W.

    Args:
        model: a PyTorch module whose forward returns logits of shape
               ``(B, C_out, H, W, D)``.
        volume: input volume of shape ``(B, C, H, W, D)`` (5-D).
        patch_size: spatial patch size. For 3-D models this is
            ``(ph, pw, pd)``. For 2-D models this is ``(ph, pw)``.
        overlap: overlap in voxels on each spatial dimension. Defaults to
                 one quarter of the patch size per dimension.
        device: target device. Defaults to ``volume.device``.
        batch_size: process this many patches at once.
        amp: enable automatic mixed precision.

    Returns:
        Full-volume logits of shape ``(B, C_out, H, W, D)``.
    """
    device = device or volume.device
    model = model.to(device).eval()
    B, C_in, H, W, D = volume.shape

    if overlap is None:
        overlap = tuple(max(1, int(ps) // 4) for ps in patch_size)

    # Determine whether this is a 2-D or 3-D model
    is_2d = hasattr(model, "patch_size") and not hasattr(model, "volume_patch_size")

    if is_2d:
        # 2-D model: slide over H, W; each patch keeps full depth D
        ph, pw = patch_size[0], patch_size[1]
        spatial_vol = (H, W)
        spatial_patch = (ph, pw)
        spatial_overlap = (
            (overlap[0], overlap[1]) if len(overlap) >= 2 else (overlap[0], overlap[0])
        )
    else:
        # 3-D model: slide over D, H, W
        ph, pw, pd = patch_size[0], patch_size[1], patch_size[2]
        spatial_vol = (D, H, W)
        spatial_patch = (ph, pw, pd)
        spatial_overlap = (
            (overlap[0], overlap[1], overlap[2])
            if len(overlap) >= 3
            else (overlap[0], overlap[1], overlap[2])
        )

    starts, ends = _generate_patches(spatial_vol, spatial_patch, spatial_overlap)
    logger.info(
        "Sliding window: %d patches over %s (2d=%s)", len(starts), spatial_vol, is_2d
    )

    if len(starts) == 0:
        logger.warning("No patches generated; falling back to full-volume forward.")
        with (
            torch.no_grad(),
            torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=amp),
        ):
            return model(volume.to(device))

    # Pre-compute cosine weight mask
    weight_mask = _cosine_overlap_weight(spatial_patch, spatial_overlap).to(device)

    # Accumulators: output and weight over the full spatial volume.
    # Model outputs (B, C_out, H, W, D) for both 2D and 3D variants.
    # Use C_in as a placeholder; we'll resize after the first forward pass.
    output_acc = torch.zeros(B, C_in, H, W, D, device=device, dtype=torch.float32)
    weight_acc = torch.zeros(B, 1, H, W, D, device=device, dtype=torch.float32)
    output_channels = C_in  # will be updated after first forward pass

    for batch_start in range(0, len(starts), batch_size):
        batch_indices = starts[batch_start : batch_start + batch_size]
        batch_ends = ends[batch_start : batch_start + batch_size]
        cur_batch_size = len(batch_indices)

        patch_tensors = []
        for si, ei in zip(batch_indices, batch_ends):
            if is_2d:
                # 2-D: extract (B, C, h, w, D) — full depth, partial H/W
                s_h, s_w = si
                e_h, e_w = ei
                patch = volume[:, :, s_h:e_h, s_w:e_w, :]  # (B, C, h, w, D)
            else:
                # 3-D: extract (B, C, d, h, w) — partial D/H/W
                s_d, s_h, s_w = si
                e_d, e_h, e_w = ei
                patch = volume[:, :, s_d:e_d, s_h:e_h, s_w:e_w]  # (B, C, d, h, w)

            # Pad to full patch size if at boundary
            ph_act = ei[0] - si[0]
            pw_act = ei[1] - si[1]
            if is_2d:
                pd_act = D  # full depth
                padded = torch.zeros(
                    cur_batch_size,
                    C_in,
                    ph,
                    pw,
                    pd_act,
                    device=device,
                    dtype=volume.dtype,
                )
                padded[:, :, :ph_act, :pw_act, :] = patch
            else:
                pd_act = ei[2] - si[2]
                padded = torch.zeros(
                    cur_batch_size,
                    C_in,
                    pd_act,
                    ph,
                    pw,
                    device=device,
                    dtype=volume.dtype,
                )
                padded[:, :, :pd_act, :ph_act, :pw_act] = patch
            patch_tensors.append(padded)

        patch_tensor = torch.cat(patch_tensors, dim=0)

        with (
            torch.no_grad(),
            torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=amp),
        ):
            logits = model(patch_tensor)

        # Update accumulator size from first batch's output
        C_out = logits.shape[1]
        if C_out != output_channels:
            output_acc = torch.zeros(
                B, C_out, H, W, D, device=device, dtype=torch.float32
            )
            weight_acc = torch.zeros(B, 1, H, W, D, device=device, dtype=torch.float32)
            output_channels = C_out

        # logits shape: (cur_batch_size, C_out, H', W', D')
        for i, (si, ei) in enumerate(zip(batch_indices, batch_ends)):
            logit_patch = logits[i : i + 1]

            if is_2d:
                # logit_patch: (1, C_out, h, w, D)
                # Weight mask: (1, ph, pw) -> expand to (1, 1, h, w, 1)
                wm_crop = weight_mask[
                    :, : logit_patch.shape[2], : logit_patch.shape[3]
                ]  # (1, h, w)
                wm_full = wm_crop.unsqueeze(0).unsqueeze(-1)  # (1, 1, h, w, 1)

                out_slices = (
                    slice(None),
                    slice(None),
                    slice(si[0], ei[0]),
                    slice(si[1], ei[1]),
                    slice(None),  # full depth
                )
                output_acc[out_slices] += logit_patch
                weight_acc[out_slices] += wm_full
            else:
                # logit_patch: (1, C_out, d, h, w)
                wm_crop = weight_mask[
                    :,
                    : logit_patch.shape[2],
                    : logit_patch.shape[3],
                    : logit_patch.shape[4],
                ]  # (1, d, h, w)
                wm_full = wm_crop.unsqueeze(0)  # (1, 1, d, h, w)

                out_slices = (
                    slice(None),
                    slice(None),
                    slice(si[0], ei[0]),
                    slice(si[1], ei[1]),
                    slice(si[2], ei[2]),
                )
                output_acc[out_slices] += logit_patch
                weight_acc[out_slices] += wm_full

    safe_weight = weight_acc.clamp(min=1e-8)
    predictions = output_acc / safe_weight
    return predictions
