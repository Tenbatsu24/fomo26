import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor):

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        idx = (None, slice(None), *([None] * (x.ndim - 2)))

        return self.weight[idx] * x + self.bias[idx]


def build_stride_schedule(patch_size):

    remaining = list(patch_size)
    schedule = []

    while max(remaining) > 1:
        stride = []
        for i in range(len(remaining)):
            if remaining[i] > 1:
                stride.append(2)
                remaining[i] = math.ceil(remaining[i] / 2)
            else:
                stride.append(1)
        schedule.append(tuple(stride))
    return schedule


class ScaleBlock(nn.Module):
    """
    Upsample + local refinement.

    ConvTranspose -> DepthwiseConv -> Norm -> GELU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
        )

        self.refine = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )

        self.norm = norm(out_channels)
        self.act = activation()

    def forward(self, x):
        x = self.up(x)
        x = self.refine(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class ScaleDecode(nn.Module):

    def __init__(
        self,
        patch_size,
        embed_dim,
        out_channels,
        min_channels=8,
        norm=LayerNormNd,
        activation=nn.GELU,
        interpolation_mode="trilinear",
    ):
        super().__init__()

        self.patch_size = tuple(patch_size)
        self.interpolation_mode = interpolation_mode

        schedule = build_stride_schedule(self.patch_size)
        num_stages = len(schedule)

        if num_stages == 0:
            channels = [embed_dim]
        else:
            channels = np.geomspace(
                embed_dim,
                max(min_channels, out_channels),
                num_stages + 1,
            )

            channels = np.round(channels / 8) * 8
            channels = channels.astype(int).tolist()

            channels[0] = embed_dim
            channels[-1] = max(min_channels, out_channels)

        blocks = []

        for i, stride in enumerate(schedule):

            blocks.append(
                ScaleBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    stride=stride,
                    norm=norm,
                    activation=activation,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Conv3d(
            channels[-1],
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        spatial_in = x.shape[2:]

        target_size = tuple(
            spatial_in[d] * self.patch_size[d] for d in range(len(self.patch_size))
        )

        for block in self.blocks:
            x = block(x)

        if x.shape[2:] != target_size:
            x = F.interpolate(
                x,
                size=target_size,
                mode=self.interpolation_mode,
                align_corners=self.interpolation_mode
                not in {"linear", "bilinear", "bicubic", "trilinear"},
            )
        x = self.head(x)

        return x


# If you want to handle multiple patch sizes and compare:
if __name__ == "__main__":
    import thop

    x = torch.randn(1, 384, 37, 37, 37)
    patch_sizes = [
        (8, 8, 8),
    ]  # Add more for comparison

    results = []
    for patch_size in patch_sizes:
        decoder = ScaleDecode(
            patch_size=patch_size,
            embed_dim=384,
            out_channels=1,
        )
        decoder.eval()

        flops, params = thop.profile(decoder, inputs=(x,), verbose=False)
        flops_f, params_f = thop.clever_format([flops, params], "%.3f")

        results.append(
            {
                "patch_size": patch_size,
                "params": params,
                "params_f": params_f,
                "flops": flops,
                "flops_f": flops_f,
            }
        )

        y = decoder(x)
        print(
            f"Patch {patch_size}: Params={params_f}, FLOPs={flops_f}, Output={tuple(y.shape)}"
        )
