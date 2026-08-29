from functools import partial
from typing import Literal, Mapping, Any, Optional

import torch
import torch.nn as nn

from med_adapt.adapter import AttentionPooling
from med_adapt.models.base import ViT3D
from med_adapt.registry import register_model
from med_adapt.utils.config import get_logger
from med_adapt.layers import (
    Block,
    ScaleDecode,
    Attention,
    MemEffAttention,
    LoRAAttention,
    LoRAMemEffAttention,
)

logger = get_logger(__name__)


def possibly_clean_lightning_sd(state_dict, prefix="model"):
    if "state_dict" in state_dict:
        possibly_clean_sd = state_dict["state_dict"]
    else:
        possibly_clean_sd = dict(state_dict)

    sd = {
        k.replace(f"{prefix}.", ""): v
        for k, v in possibly_clean_sd.items()
        if k.startswith(prefix)
    }

    if not sd:
        return possibly_clean_sd
    else:
        return sd


class ViT3DAdaption(ViT3D):

    def __init__(
        self,
        *,
        n_modalities,
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        query_from: Optional[int] = None,
        num_q_tokens: Optional[int] = None,
        **kwargs,
    ):
        super(ViT3DAdaption, self).__init__(med_in_channels=n_modalities, **kwargs)

        self.task = task
        self.classes = classes
        self.n_modalities = n_modalities

        if query_from is None:
            query_from = -3

        self.query_from = (
            len(self.blocks) + query_from if query_from < 0 else query_from
        )

        if task in ["classification", "regression"]:
            self.num_q_tokens = num_q_tokens if num_q_tokens is not None else 16
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_q_tokens, self.embed_dim)
            )
            nn.init.normal_(self.query_tokens, std=1e-6)
            self.query_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.attn_head = nn.Sequential(
                AttentionPooling(self.embed_dim, num_classes=1, num_heads=4),
                nn.Linear(self.embed_dim, classes),
            )
        else:  # task == "segmentation":
            self.num_q_tokens = 0
            self.query_tokens = None
            self.head = ScaleDecode(self.patch_size, self.embed_dim, classes)

        self.init_weights()

    def forward(self, x, **kwargs):
        b, c, h, w, d = x.shape
        lp = tuple(l // p for l, p in zip([h, w, d], self.patch_size))

        preds = []

        x = self.prepare_tokens(x)

        for i, blk in enumerate(self.blocks):
            if (self.query_tokens is not None) and (i == self.query_from):
                x = torch.cat((self.query_tokens.repeat(b, 1, 1), x), dim=1)
            x = blk(x)

            if self.task != "segmentation":
                query_latent = self.query_norm(
                    x[:, : self.num_q_tokens, :]
                )  # [B, q, d]
                pred = self.attn_head(query_latent).squeeze(1)
                if self.task == "regression":
                    pred = 100 * torch.sigmoid(pred)

                preds.append(pred)

        if self.task == "segmentation":
            patch_tokens = self.norm(
                x[:, self.num_q_tokens + self.num_register_tokens + 1 :, :]
            )
            spatial = patch_tokens.unflatten(1, lp).permute(0, -1, 1, 2, 3)
            out = self.head(spatial)
        else:
            out = preds

        return out

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = possibly_clean_lightning_sd(state_dict)

        key = "patch_embed.proj.weight"
        if key in state_dict:
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
        default_ = [
            "attn_head",
            "query_tokens",
            "query_norm",
            "head",
            "channel_adapter",
        ]
        if self.n_modalities > 1:
            default_ += ["patch_embed"]
        return default_


@register_model("vitv2_a_3d_tiny")
def vitv2_a_3d_tiny(
    lora=False,
    mea=True,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViT3DAdaption(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=0,
        use_mask=False,
        use_patch_decode=False,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_small")
def vitv2_a_3d_small(
    lora=False,
    mea=True,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViT3DAdaption(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=0,
        use_mask=False,
        use_patch_decode=False,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_base")
def vitv2_a_3d_base(
    lora=False,
    mea=True,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViT3DAdaption(
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=0,
        use_mask=False,
        use_patch_decode=False,
        **kwargs,
    )
    return model


@register_model("vitv2_a_3d_large")
def vitv2_a_3d_large(
    lora=False,
    mea=True,
    **kwargs,
):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ViT3DAdaption(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(
            Block,
            attn_class=(
                (LoRAMemEffAttention if mea else LoRAAttention)
                if lora
                else (MemEffAttention if mea else Attention)
            ),
        ),
        num_register_tokens=0,
        use_mask=False,
        use_patch_decode=False,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    _m = vitv2_a_3d_small(
        n_modalities=1,
        task="regression",
        classes=2,
        lora=False,
        mea=True,
    ).to("cuda")

    _missing, _unexpected = _m.load_state_dict(
        torch.load("../../../checkpoints/small/296_518/last.ckpt"),
        strict=False,
    )
    print(
        f"[missing keys={len(_missing)}]\n\t{_missing},\n[unexpected_keys={len(_unexpected)}]\n\t{_unexpected}"
    )
    print(
        [
            (
                _out.shape
                if isinstance(_out, torch.Tensor)
                else [_out_i.shape for _out_i in _out]
            )
            for _out in _m(
                torch.randn(1, 1, 196, 196, 24, device="cuda", dtype=torch.float32)
            )
        ]
    )
