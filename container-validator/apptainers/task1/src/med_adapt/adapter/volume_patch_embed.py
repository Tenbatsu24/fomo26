import math

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn


def make_3tuple(x) -> tuple[int, int, int]:
    if isinstance(x, tuple) or isinstance(x, list):
        assert len(x) == 3
        return tuple(x)

    assert isinstance(x, int)
    return x, x, x


class PatchEmbed3D(nn.Module):
    """
    3D image to patch embedding with HWD layout: (B, C, H, W, D) -> (B, N, E)

    Args:
        img_size: Input volume size (Height, Width, Depth).
        patch_size: Patch token size (height, width, depth).
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
        flatten_embedding: Whether to flatten the spatial/depth dimensions.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int, int]] = (224, 224, 32),
        patch_size: Union[int, Tuple[int, int, int]] = (14, 14, 2),
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HWD = make_3tuple(img_size)
        patch_HWD = make_3tuple(patch_size)
        patch_grid_size = (
            image_HWD[0] // patch_HWD[0],
            image_HWD[1] // patch_HWD[1],
            image_HWD[2] // patch_HWD[2],
        )

        self.img_size = image_HWD
        self.patch_size = patch_HWD
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        # PyTorch Conv3D internally requires (depth, height, width) ordering.
        # So we map HWD -> DHW for Conv3d kernel and stride configurations.
        conv_kernel_size = (patch_HWD[2], patch_HWD[0], patch_HWD[1])
        conv_stride_size = (patch_HWD[2], patch_HWD[0], patch_HWD[1])

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=conv_kernel_size, stride=conv_stride_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: B, C, H, W, D
        _, _, H, W, D = x.shape

        # Transpose HWD to PyTorch's native DHW conv format: (B, C, D, H, W)
        x = x.permute(0, 1, 4, 2, 3)

        x = self.proj(x)  # Shape: B, C, D_out, H_out, W_out
        D_out, H_out, W_out = x.size(2), x.size(3), x.size(4)

        # Flatten and transpose: (B, D_out * H_out * W_out, C)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        if not self.flatten_embedding:
            # Reshape back using HWD ordering: (B, H_out, W_out, D_out, C)
            # 1. Reshape to DHW order: (B, D_out, H_out, W_out, C)
            x = x.reshape(-1, D_out, H_out, W_out, self.embed_dim)
            # 2. Permute back to HWD order: (B, H_out, W_out, D_out, C)
            x = x.permute(0, 2, 3, 1, 4)

        return x

    def flops(self) -> float:
        Ho, Wo, Do = self.patches_resolution
        flops = (
            Ho
            * Wo
            * Do
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        )
        if not isinstance(self.norm, nn.Identity):
            flops += Ho * Wo * Do * self.embed_dim
        return flops

    def reset_parameters(self):
        patch_volume = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        k = 1 / (self.in_chans * patch_volume)
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
