from typing import Any, Mapping

import torch

from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)

from med_adapt.adapter import AttentionPooling
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


@register_model("resencl")
class ResEncLAdaption(torch.nn.Module):

    def __init__(self, n_modalities, _, classes, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embed_dim = 320
        self.n_modalities = n_modalities

        self.encoder = make_residual_encoder_l(n_modalities)

        self.latent_norm = torch.nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn_head = torch.nn.Sequential(
            AttentionPooling(self.embed_dim, num_classes=1, num_heads=4),
            torch.nn.Linear(self.embed_dim, classes),
        )

    def forward(self, x):
        z = self.encoder(x).flatten(2).permute(0, 2, 1)
        logits = self.attn_head(self.latent_norm(z)).squeeze(1)
        return logits

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

    _randn = torch.randn(1, _n_modalities, 160, 160, 160, device="cuda")
    _encl = ResEncLAdaption(_n_modalities, None, 2).to("cuda")

    _missing, _unexpected = _encl.load_state_dict(
        torch.load("../../../checkpoints/resencl/128/encoder_only.ckpt"),
        strict=False,
    )
    print(
        f"[missing keys={len(_missing)}]\n\t{_missing},\n[unexpected_keys={len(_unexpected)}]\n\t{_unexpected}"
    )

    print(_encl(_randn).shape)
