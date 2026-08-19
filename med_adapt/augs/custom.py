import random

import torch
import torch.nn.functional as F


class Resize3D:
    """Resize the entire 3D volume to a fixed target shape on the CPU."""

    def __init__(
        self,
        data_key="image",
        label_key="label",
        target_size: tuple | list = None,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.target_size = tuple(target_size) if target_size else None

    def __call__(self, data_dict):
        image = data_dict[self.data_key]
        if self.target_size is not None:
            # image shape: (C, H, W, D) -> interpolate expects (N, C, D, H, W)
            resized = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            # resized: (C, target_D, target_H, target_W) -> back to (C, target_H, target_W, target_D)
            data_dict[self.data_key] = resized

        if data_dict.get(self.label_key) is not None and self.target_size is not None:
            label = data_dict[self.label_key]
            # label shape: (H, W, D) -> (N, C, D, H, W) for interpolate
            # Cast to float for interpolate, then back to original dtype.
            label_float = label.to(dtype=torch.float32)
            resized_label = torch.nn.functional.interpolate(
                label_float.unsqueeze(0),
                size=self.target_size,
                mode="nearest",
            ).squeeze(0)
            # resized_label: (target_D, target_H, target_W) -> (target_H, target_W, target_D)
            data_dict[self.label_key] = resized_label.to(label.dtype)

        return data_dict


class PadToShape3D:

    def __init__(self, size, label_key=None):
        self.size = tuple(size)  # (H, W, D)
        self.label_key = label_key

    def __call__(self, sample):
        out = dict(sample)

        image = sample["image"]

        H, W, D = image.shape[-3:]
        target_H, target_W, target_D = self.size

        pad_h = max(0, target_H - H)
        pad_w = max(0, target_W - W)
        pad_d = max(0, target_D - D)

        if pad_h == 0 and pad_w == 0 and pad_d == 0:
            return out

        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before

        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before

        pad_d_before = pad_d // 2
        pad_d_after = pad_d - pad_d_before

        # F.pad order: (..., D, W, H)
        pad = (
            pad_d_before,
            pad_d_after,
            pad_w_before,
            pad_w_after,
            pad_h_before,
            pad_h_after,
        )

        out["image"] = F.pad(
            image,
            pad,
            mode="constant",
            value=float(image.min()),
        )

        if self.label_key is not None and self.label_key in sample:
            label = sample[self.label_key]

            if isinstance(label, torch.Tensor) and label.shape[-3:] == (H, W, D):
                out[self.label_key] = F.pad(
                    label,
                    pad,
                    mode="constant",
                    value=0,
                )

        return out


class RandomResizedCrop3D:
    def __init__(
        self,
        size,
        scale=(0.5, 1.0),
        ratio=(0.9, 1.1),
        label_key=None,
    ):
        self.size = tuple(size)
        self.scale = scale
        self.ratio = ratio
        self.label_key = label_key

    def _sample_crop(self, H, W, D):
        volume = H * W * D

        for _ in range(10):
            target_volume = random.uniform(*self.scale) * volume

            r_hw = random.uniform(*self.ratio)
            r_hd = random.uniform(*self.ratio)

            h = round((target_volume * r_hw * r_hd) ** (1.0 / 3.0))
            w = round(h / r_hw)
            d = round(h / r_hd)

            if 0 < h <= H and 0 < w <= W and 0 < d <= D:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                depth = random.randint(0, D - d)

                return top, left, depth, h, w, d

        # Fallback: use the whole volume
        return 0, 0, 0, H, W, D

    def _crop_resize_image(self, x, params):
        top, left, depth, h, w, d = params

        cropped = x[
            ...,
            :,
            top : top + h,
            left : left + w,
            depth : depth + d,
        ]

        had_batch = cropped.ndim == 5

        if not had_batch:
            cropped = cropped.unsqueeze(0)

        resized = F.interpolate(
            cropped,
            size=self.size,
            mode="trilinear",
            align_corners=False,
        )

        if not had_batch:
            resized = resized.squeeze(0)

        return resized

    def _crop_resize_label_3d(self, x, params):
        top, left, depth, h, w, d = params

        cropped = x[
            ...,
            :,
            top : top + h,
            left : left + w,
            depth : depth + d,
        ]

        had_batch = cropped.ndim == 5

        if not had_batch:
            cropped = cropped.unsqueeze(0)

        resized = F.interpolate(
            cropped.float(),
            size=self.size,
            mode="nearest",
        )

        if not had_batch:
            resized = resized.squeeze(0)

        return resized.to(x.dtype)

    def __call__(self, sample):
        image = sample["image"]

        H, W, D = image.shape[-3:]

        params = self._sample_crop(H, W, D)

        out = dict(sample)

        out["image"] = self._crop_resize_image(
            image,
            params,
        )

        if self.label_key is not None and self.label_key in sample:
            label = sample[self.label_key]

            if isinstance(label, torch.Tensor) and label.shape[-3:] == (H, W, D):
                out[self.label_key] = self._crop_resize_label_3d(
                    label,
                    params,
                )

        return out


class CenterCrop3D:
    """
    Center crop for samples of the form:

        {
            "image": [..., C, H, W, D],
            "label": optional
        }

    Crops image and any spatially-matching label to the
    requested output size (H, W, D).

    Assumes the input is already large enough (e.g. after
    PadToShape3D).
    """

    def __init__(self, size, label_key=None):
        self.size = tuple(size)  # (H, W, D)
        self.label_key = label_key

    def __call__(self, sample):
        out = dict(sample)

        image = sample["image"]

        H, W, D = image.shape[-3:]

        crop_H, crop_W, crop_D = self.size

        top = (H - crop_H) // 2
        left = (W - crop_W) // 2
        depth = (D - crop_D) // 2

        out["image"] = image[
            ...,
            :,
            top : top + crop_H,
            left : left + crop_W,
            depth : depth + crop_D,
        ]

        if self.label_key is not None and self.label_key in sample:
            label = sample[self.label_key]

            if isinstance(label, torch.Tensor) and label.shape[-3:] == (H, W, D):
                out[self.label_key] = label[
                    ...,
                    :,
                    top : top + crop_H,
                    left : left + crop_W,
                    depth : depth + crop_D,
                ]

        return out


class RandomSwapSpatialDims3D:

    def __init__(self, p=0.5, label_key=None):
        self.p = p
        self.label_key = label_key

        # indices of H,W,D relative to the last 3 dims
        self._perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]

    def _apply(self, x, perm):
        ndim = x.ndim

        spatial = [ndim - 3 + p for p in perm]

        order = list(range(ndim - 3)) + spatial

        return x.permute(*order)

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample

        image = sample["image"]

        H, W, D = image.shape[-3:]

        perm = random.choice(self._perms)

        out = dict(sample)

        out["image"] = self._apply(image, perm)

        if self.label_key is not None and self.label_key in sample:
            label = sample[self.label_key]

            if isinstance(label, torch.Tensor) and label.shape[-3:] == (H, W, D):
                out[self.label_key] = self._apply(label, perm)

        return out


class RandomFlipSpatialDims3D:

    def __init__(self, p=0.5, label_key=None):
        self.p = p
        self.label_key = label_key

    def _apply(self, x, dims):
        return torch.flip(x, dims=dims)

    def __call__(self, sample):
        image = sample["image"]

        # spatial dims are always the last 3 dims
        spatial_dims = [-3, -2, -1]

        flip_dims = [d for d in spatial_dims if random.random() < self.p]

        if not flip_dims:
            return sample

        H, W, D = image.shape[-3:]

        out = dict(sample)

        out["image"] = self._apply(image, flip_dims)

        if self.label_key is not None and self.label_key in sample:
            label = sample[self.label_key]

            if isinstance(label, torch.Tensor) and label.shape[-3:] == (H, W, D):
                out[self.label_key] = self._apply(label, flip_dims)

        return out
