import torch
import torch.nn as nn


class InputChannelAdapter(nn.Module):
    """
    Adapts a 3D volume with an arbitrary number of input channels into a
    fixed-channel output (e.g. mapping N modality/grayscale-like channels
    into an RGB-like triplet for a downstream img2img-style pipeline).

    Input:  x of shape (B, C_in, D, H, W)
    Output: (B, out_channels, D, H, W)

    Pure conv stack (Conv3d), with spatial mixing on by default -- each
    layer uses a 3x3x3 kernel, and dilation grows across layers to expand
    the receptive field without downsampling. Good for larger volumes
    (e.g. ~256^3-ish spatial extent) where you want the adapter to blend
    in real neighborhood context, not just remix channels per-voxel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden_dim: int = 8,
        kernel_size: int = 3,
        num_layers: int = 2,
        growing_dilation: bool = True,  # dilation = 1, 2, 4, ... per layer for larger receptive field
        activation: nn.Module = nn.GELU,
        norm: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels

        layers = []
        c_in = in_channels
        for i in range(num_layers - 1):
            dilation = (2**i) if growing_dilation else 1
            padding = dilation * (kernel_size - 1) // 2  # keeps spatial size fixed
            layers.append(
                nn.Conv3d(
                    c_in,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                )
            )
            if norm:
                layers.append(
                    nn.BatchNorm3d(
                        hidden_dim, affine=False, track_running_stats=False, eps=1e-3
                    )
                )
            layers.append(activation())
            c_in = hidden_dim

        # final projection layer, standard dilation=1, kernel as given
        padding = (kernel_size - 1) // 2
        layers.append(
            nn.Conv3d(c_in, out_channels, kernel_size=kernel_size, padding=padding)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, D, H, W)
        returns: (B, out_channels, D, H, W), same spatial size as input
        """
        assert (
            x.shape[1] == self.in_channels
        ), f"expected {self.in_channels} input channels, got {x.shape[1]}"
        return self.net(x)


if __name__ == "__main__":
    adapter = InputChannelAdapter(in_channels=1, out_channels=3)
    vol = torch.randn(2, 1, 128, 256, 256)  # (B, C_in, D, H, W)
    out = adapter(vol)  # (2, 3, 128, 256, 256)
    print(f"Input shape: {vol.shape}, Output shape: {out.shape}")
