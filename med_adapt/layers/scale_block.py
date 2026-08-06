# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
from torch import nn
from einops import rearrange


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv_type: str = "2d"):
        super().__init__()

        if conv_type == "2d":
            conv1 = nn.ConvTranspose2d
            conv2 = nn.Conv2d
        elif conv_type == "3d":
            conv1 = nn.ConvTranspose3d
            conv2 = nn.Conv3d
        else:
            raise ValueError(f"Unknown conv type: {conv_type}")

        self.conv1 = conv1(
            embed_dim,
            embed_dim // 2,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = conv2(
            embed_dim // 2,
            embed_dim // 2,
            kernel_size=3,
            padding=1,
            groups=embed_dim // 2,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = rearrange(x, "b c ... -> b ... c")
        x = self.norm(x)
        x = rearrange(x, "b ... c -> b c ...")

        return x
