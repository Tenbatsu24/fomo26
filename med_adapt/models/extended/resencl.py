from typing import Any, Mapping

import torch

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from einops import rearrange

from med_adapt.adapter import AttentionPooling
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


@register_model("resencl")
class ResEncLAdaption(torch.nn.Module):

    def __init__(
        self,
        n_modalities: int,
        classes: int,
        patch_size: tuple[int, int, int] = (128, 128, 128),
    ):
        super().__init__()

        self.embed_dim = 320
        self.n_modalities = n_modalities
        self.patch_size = patch_size  # for e.g. (128, 128, 128)

        self.encoder = make_residual_encoder_l(n_modalities)

        self.latent_norm = torch.nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn_head = torch.nn.Sequential(
            AttentionPooling(self.embed_dim, num_classes=1, num_heads=4),
            torch.nn.Linear(self.embed_dim, classes),
        )

    def forward(self, x, max_batch_size: int = 2):
        tiles = self._extract_tiles(
            x
        )  # Shape: (num_tiles, C, patch_D, patch_H, patch_W)
        # print(tiles.shape)

        num_tiles = tiles.shape[0]

        # Process tiles in chunks
        all_latents = []
        for i in range(0, num_tiles, max_batch_size):
            chunk = tiles[
                i : i + max_batch_size
            ]  # Shape: (chunk_size, C, patch_D, patch_H, patch_W)

            z = self.encoder(chunk)  # Shape: (chunk_size, embed_dim, spatial_dims...)
            z = z.flatten(2).permute(0, 2, 1)  # Shape: (chunk_size, N, C)
            all_latents.append(z)

        # Concatenate all latents from chunks
        z = torch.cat(all_latents, dim=0)  # Shape: (num_tiles, N, C)

        z_norm = self.latent_norm(z)  # Shape: (num_tiles, N, C)
        logits = self.attn_head(rearrange(z_norm, "b n c -> 1 (b n) c")).squeeze(
            1
        )  # Shape: (num_tiles, classes)

        return logits

    def _compute_tile_starts(
        self, volume_size: tuple[int, int, int]
    ) -> list[tuple[int, int, int]]:
        """Compute starting positions for tiles with minimum 50% overlap."""
        patch_d, patch_h, patch_w = self.patch_size
        vol_d, vol_h, vol_w = volume_size

        # Minimum stride is 50% of patch size (i.e., overlap at least 50%)
        stride_d = max(1, 3 * patch_d // 4)
        stride_h = max(1, 3 * patch_h // 4)
        stride_w = max(1, 3 * patch_w // 4)

        starts_d = list(range(0, vol_d - patch_d + 1, stride_d))
        starts_h = list(range(0, vol_h - patch_h + 1, stride_h))
        starts_w = list(range(0, vol_w - patch_w + 1, stride_w))

        # Ensure the last tile covers the end of the volume
        if starts_d[-1] + patch_d < vol_d:
            starts_d.append(vol_d - patch_d)
        if starts_h[-1] + patch_h < vol_h:
            starts_h.append(vol_h - patch_h)
        if starts_w[-1] + patch_w < vol_w:
            starts_w.append(vol_w - patch_w)

        # Generate all combinations of start positions
        tile_starts = []
        for d in starts_d:
            for h in starts_h:
                for w in starts_w:
                    tile_starts.append((d, h, w))

        return tile_starts

    def _extract_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """Extract all tiles from the input volume."""
        tile_starts = self._compute_tile_starts(x.shape[2:])  # D, H, W

        tiles = []
        for d, h, w in tile_starts:
            tile = x[
                :,
                :,
                d : d + self.patch_size[0],
                h : h + self.patch_size[1],
                w : w + self.patch_size[2],
            ]
            tiles.append(tile)

        # Concatenate all tiles at the batch dimension
        return torch.cat(tiles, dim=0)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        keys_to_fix = [
            "encoder.stem.convs.0.conv.weight",
            "encoder.stem.convs.0.all_modules.0.weight",
        ]
        state_dict = {**state_dict}

        for key in keys_to_fix:
            pretrained_weight = state_dict[key]
            current_weight = self.state_dict()[key]

            pretrained_in_channels = pretrained_weight.shape[1]
            target_in_channels = current_weight.shape[1]

            if pretrained_in_channels != target_in_channels:
                logger.info(
                    f"Trying to repeat {pretrained_in_channels=} to {target_in_channels=}"
                )
                if pretrained_in_channels != 1:
                    raise ValueError(
                        f"Expected pretrained '{key}' to have 1 input channel to duplicate "
                        f"across modalities, but got {pretrained_in_channels}."
                    )
                if target_in_channels != self.n_modalities:
                    raise ValueError(
                        f"Model's '{key}' in_channels ({target_in_channels}) does not match "
                        f"self.n_modalities ({self.n_modalities})."
                    )

                # Duplicate the single-modality stem across the channel dim (dim=1).
                repeat_shape = [1] * pretrained_weight.dim()
                repeat_shape[1] = target_in_channels
                duplicated_weight = pretrained_weight.repeat(*repeat_shape)

                duplicated_weight = duplicated_weight / target_in_channels

                state_dict[key] = duplicated_weight
                logger.success(
                    f"Repeated {pretrained_in_channels=} to {target_in_channels=}"
                )

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def additional_trainable(self):
        return [
            "latent_norm",
            "attn_head",
            "encoder.stem.convs.0.conv.weight",
            "encoder.stem.convs.0.all_modules.0.weight",
        ]

    def do_not_load(self):
        return None


def make_residual_encoder_l(
    input_channels,
):
    network = ResidualEncoder(
        input_channels=input_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=tuple((3, 3, 3) for _ in range(6)),
        strides=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        block=BasicBlockD,
        bottleneck_channels=None,
        return_skips=False,
        disable_default_stem=False,
        stem_channels=None,
    )
    return network


if __name__ == "__main__":
    _n_modalities = 3

    _randn = torch.randn(
        1,
        _n_modalities,
        160,
        160,
        160,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    _encl = ResEncLAdaption(_n_modalities, 2).to("cuda").eval()

    _missing, _unexpected = _encl.load_state_dict(
        torch.load("../../../checkpoints/resencl/128/encoder_only.ckpt"),
        strict=False,
    )
    _encl.to(torch.float16)
    print(
        f"[missing keys={len(_missing)}]\n\t{_missing},\n[unexpected_keys={len(_unexpected)}]\n\t{_unexpected}"
    )

    with torch.no_grad():
        print(_encl(_randn).shape)
