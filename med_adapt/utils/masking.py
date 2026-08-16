import random

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def complete_mask_randomly_np(mask, num_masking_voxels, rng):
    """
    Internal convention:
        True  = masked (drop)
        False = keep
    """
    flat = mask.reshape(-1)

    missing = num_masking_voxels - flat.sum()
    if missing <= 0:
        return mask

    available = np.flatnonzero(~flat)

    chosen = rng.choice(
        available,
        size=missing,
        replace=False,
    )

    flat[chosen] = True
    return mask


class MaskGenerator3D:
    """
    Input convention:
        (H, W, D)

    Internal mask convention:
        True  = masked (drop)
        False = keep
    """

    def __init__(
        self,
        input_size,
        min_num_voxels=8,
        min_aspect=0.3,
        max_aspect=3.33,
        max_tries=10,
    ):
        if isinstance(input_size, int):
            input_size = (input_size, input_size, input_size)

        self.h, self.w, self.d = input_size

        self.num_patches = self.h * self.w * self.d

        self.min_num_voxels = min_num_voxels

        self.log_min_aspect = np.log(min_aspect)
        self.log_max_aspect = np.log(
            max_aspect if max_aspect is not None else 1.0 / min_aspect
        )

        self.max_tries = max_tries

    def __call__(
        self,
        num_masking_voxels,
        starting_mask=None,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        if starting_mask is None:
            mask = np.zeros(
                (self.h, self.w, self.d),
                dtype=np.bool_,
            )
        else:
            mask = starting_mask.copy()

        mask_count = mask.sum()

        while mask_count < num_masking_voxels:
            max_mask = num_masking_voxels - mask_count

            if max_mask < self.min_num_voxels:
                break

            delta = self._mask(mask, max_mask, rng)

            if delta == 0:
                break

            mask_count += delta

        mask = complete_mask_randomly_np(
            mask,
            num_masking_voxels,
            rng,
        )

        return mask

    def _mask(
        self,
        mask,
        max_mask_voxels,
        rng,
    ):
        """
        Sample a random cuboid.
        """

        for _ in range(self.max_tries):
            target = rng.uniform(
                self.min_num_voxels,
                max_mask_voxels,
            )

            aspect1 = np.exp(
                rng.uniform(
                    self.log_min_aspect,
                    self.log_max_aspect,
                )
            )

            aspect2 = np.exp(
                rng.uniform(
                    self.log_min_aspect,
                    self.log_max_aspect,
                )
            )

            h = int(round((target * aspect1) ** (1 / 3)))
            w = int(round((target * aspect2 / aspect1) ** (1 / 3)))
            d = int(round((target / (aspect1 * aspect2)) ** (1 / 3)))

            if h <= 0 or w <= 0 or d <= 0 or h >= self.h or w >= self.w or d >= self.d:
                continue

            top = rng.integers(0, self.h - h + 1)
            left = rng.integers(0, self.w - w + 1)
            depth = rng.integers(0, self.d - d + 1)

            region = mask[
                top : top + h,
                left : left + w,
                depth : depth + d,
            ]

            newly = (~region).sum()

            if 0 < newly <= max_mask_voxels:
                region[:] = True
                return newly

        return 0


def generate_masks(
    patch_resolution,
    number_of_samples,
    mask_prob=0.1,
    per_sample_range=(0.1, 0.2),
):
    mask_generator = MaskGenerator3D(
        input_size=patch_resolution,
    )
    num_masks = int(number_of_samples * mask_prob)

    num_tokens = mask_generator.num_patches

    prob_per_sample = np.linspace(
        *per_sample_range,
        num=num_masks,
    )

    masks = []

    for i in range(number_of_samples):
        if i < num_masks:
            masked = mask_generator(
                num_masking_voxels=int(prob_per_sample[i] * num_tokens)
            )
        else:
            masked = mask_generator(num_masking_voxels=0)

        masks.append(masked)

    random.shuffle(masks)

    masks = np.stack(
        masks,
        dtype=np.bool_,
    )

    masks = torch.from_numpy(masks)

    masks = masks.flatten(1)

    return masks


def visualize_volume_mask(mask_keep):

    d = mask_keep.shape[2]

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(10, 10),
    )

    cmap = ListedColormap(
        [
            "white",  # kept
            "red",  # dropped
        ]
    )

    for z, ax in enumerate(axes.flat):
        if z >= d:
            ax.axis("off")
            continue

        ax.imshow(
            mask_keep[:, :, z],
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )

        # draw voxel grid
        ax.set_xticks(np.arange(-0.5, mask_keep.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mask_keep.shape[0], 1), minor=True)

        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)

        ax.tick_params(
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

        ax.set_title(f"z={z}")

    plt.tight_layout()
    plt.show()


def main():
    h, w, d = 14, 14, 16

    masks = generate_masks(
        (h, w, d),
        number_of_samples=4,
        mask_prob=1.0,
        per_sample_range=(0.10, 0.20),
    )

    for mask in masks:

        mask = mask.reshape(h, w, d).numpy()

        print(f"drop ratio = {mask.mean():.4f}, keep ratio = {(mask == 0).mean():.4f}")

        visualize_volume_mask(mask)


if __name__ == "__main__":
    main()
