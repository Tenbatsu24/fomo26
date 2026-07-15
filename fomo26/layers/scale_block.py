# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
from einops import rearrange
from torch import nn


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim // 2,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
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
