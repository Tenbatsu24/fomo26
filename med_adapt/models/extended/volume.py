from functools import partial
from typing import Literal, Mapping, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        task: Literal["regression", "classification", "segmentation", "none"],
        classes: int,
        pred_from: Optional[int] = None,
        **kwargs,
    ):
        super(ViT3DAdaption, self).__init__(**kwargs)

        self.task = task
        self.classes = classes

        if pred_from is None:
            pred_from = -6

        self.pred_from = len(self.blocks) + pred_from if pred_from < 0 else pred_from

        if task not in ["segmentation", "none"]:
            self.num_q_tokens = 0 if task != "classification" else classes
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_q_tokens, self.embed_dim), requires_grad=True
            )
            nn.init.normal_(self.query_tokens, std=1e-6)
        else:
            self.query_tokens = None
            self.query_mlp = None
            self.num_q_tokens = 0

        if task == "segmentation":
            self.patch_decode = ScaleDecode(self.patch_size, self.embed_dim, classes)
        elif task == "classification":
            self.query_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 4, classes, bias=False),
            )
        else:
            self.query_mlp = nn.ModuleDict(
                {
                    f"reg_{i}": nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
                        nn.GELU(),
                        nn.Linear(self.embed_dim // 4, 1, bias=True),
                    )
                    for i in range(self.num_q_tokens)
                }
            )

    def forward(self, x, **kwargs):
        b, c, h, w, d = x.shape
        lp = tuple(l // p for l, p in zip([h, w, d], self.patch_size))

        x = self.prepare_tokens(x)

        preds = []

        for i, blk in enumerate(self.blocks):
            if (self.query_tokens is not None) and (i == self.pred_from):
                x = torch.cat((self.query_tokens.repeat(b, 1, 1), x), dim=1)

            # logger.debug(f"Depth: {i=}, {x.shape}")
            x = blk(x)

            if i >= self.pred_from:
                if self.task == "segmentation":
                    patch_tokens = self.norm(
                        x[:, self.num_q_tokens + self.num_register_tokens + 1 :, :]
                    )

                    spatial = patch_tokens.unflatten(1, lp).permute(0, -1, 1, 2, 3)
                    preds.append(self.patch_decode(spatial))
                else:
                    if self.task == "none":
                        preds.append(
                            self.norm(x[:, : self.num_register_tokens + 1, :])[:, 0]
                        )
                    else:
                        query_logits = self.norm(
                            x[:, : self.num_q_tokens, :]
                        )  # [B, q, d]
                        if self.task == "classification":
                            cls_pred = self.query_mlp(query_logits[:, 0])
                            preds.append(cls_pred)
                        else:  # self.task == "regression":
                            reg_pred = torch.stack(
                                [
                                    self.query_mlp[f"class_{i}"](query_logits[:, i, :])[
                                        :, 0
                                    ]
                                    for i in range(self.num_q_tokens)
                                ],
                                dim=-1,
                            )
                            preds.append(reg_pred)
        return preds

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = possibly_clean_lightning_sd(state_dict)
        model_ps = self.patch_size

        new_state_dict = {**state_dict}

        ckpt_ps = state_dict["patch_embed.proj.weight"].shape[2:]
        if model_ps != ckpt_ps:
            w_ckpt = state_dict["patch_embed.proj.weight"]
            kd_model = model_ps[2]
            kd_ckpt = ckpt_ps[2]
            logger.info(f"Resampling patch_embed depth: {kd_ckpt} -> {kd_model}")
            w_resampled = F.interpolate(
                w_ckpt,
                size=model_ps,
                mode="trilinear",
                align_corners=False,
            )
            new_state_dict["patch_embed.proj.weight"] = w_resampled
            logger.info(f"  weight: {w_ckpt.shape} -> {w_resampled.shape}")
        else:
            logger.info(f"Keeping trained patch size: {ckpt_ps}")

        ckpt_inch = new_state_dict["patch_embed.proj.weight"].shape[1]
        if self.in_channels != ckpt_inch:
            logger.info(f" in_ch: {ckpt_inch} -> {self.in_channels}")
            new_proj = (
                new_state_dict["patch_embed.proj.weight"]
                .mean(dim=1, keepdim=True)
                .repeat(1, self.in_channels, 1, 1, 1)
            )
            new_state_dict["patch_embed.proj.weight"] = new_proj
        else:
            logger.info(f"Keeping original trained patch_embed: {ckpt_inch}")

        if self.task == "segmentation" and any(
            ["patch_decode" in k for k in new_state_dict.keys()]
        ):
            ckpt_outch = new_state_dict["patch_decode.head.weight"].shape[0]
            if self.classes != ckpt_outch:
                logger.info(f"out_ch: {ckpt_outch} -> {self.classes}")
                new_proj = (
                    new_state_dict["patch_decode.head.weight"]
                    .mean(dim=0, keepdim=True)
                    .repeat(self.classes, 1, 1, 1, 1)
                )
                new_state_dict["patch_decode.head.weight"] = new_proj
                new_bias = (
                    new_state_dict["patch_decode.head.bias"]
                    .mean(dim=0, keepdim=True)
                    .repeat(self.classes)
                )
                new_state_dict["patch_decode.head.bias"] = new_bias

        del state_dict

        return super().load_state_dict(new_state_dict, strict=strict, assign=assign)

    def additional_trainable(self):
        return [
            "query_mlp",
            "query_tokens",
            "patch_decode",
        ]


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
        **kwargs,
    )
    return model


if __name__ == "__main__":
    _m = vitv2_a_3d_small(
        med_in_channels=1,
        task="segmentation",
        classes=2,
        lora=False,
    ).to("cuda")

    _missing, _unexpected = _m.load_state_dict(
        torch.load("../../../checkpoints/small/neco_3d/last.ckpt"),
        strict=False,
    )
    print(
        f"[missing keys={len(_missing)}]\n\t{_missing},\n[unexpected_keys={len(_unexpected)}]\n\t{_unexpected}"
    )
    print(
        [
            _out.shape
            for _out in _m(
                torch.randn(1, 1, 196, 196, 24, device="cuda", dtype=torch.float32)
            )
        ]
    )
