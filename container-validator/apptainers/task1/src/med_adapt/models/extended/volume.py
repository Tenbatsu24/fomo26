from functools import partial
from typing import Literal, Mapping, Any, Optional

import torch
import torch.nn as nn

from med_adapt.adapter import AttentionPooling, ConvexModalityAdapter
from med_adapt.layers import (
    Block,
    ScaleDecode,
    Attention,
    MemEffAttention,
)
from med_adapt.models.base.vit3d import ViT3D


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
        pred_from: Optional[int] = None,
        num_q_tokens: Optional[int] = None,
        **kwargs,
    ):
        super(ViT3DAdaption, self).__init__(**kwargs)

        self.task = task
        self.classes = classes
        self.n_modalities = n_modalities

        if pred_from is None:
            pred_from = -3

        self.pred_from = len(self.blocks) + pred_from if pred_from < 0 else pred_from

        if n_modalities != 1:
            self.channel_adapter = ConvexModalityAdapter(self.n_modalities)
        else:
            self.channel_adapter = nn.Identity()

        if task in ["classification", "regression"]:
            self.num_q_tokens = num_q_tokens if num_q_tokens is not None else 4
            self.query_tokens = nn.Parameter(
                torch.zeros(1, self.num_q_tokens, self.embed_dim)
            )
            nn.init.normal_(self.query_tokens, std=1e-6)
            self.query_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            self.attn_head = nn.Sequential(
                AttentionPooling(self.embed_dim, num_classes=1, num_heads=1),
                nn.Linear(self.embed_dim, classes),
            )
            if task == "regression":
                nn.init.constant_(self.attn_head[-1].bias, 50.0)
        elif task == "segmentation":
            self.num_q_tokens = 0
            self.query_tokens = None
            self.head = ScaleDecode(self.patch_size, self.embed_dim, classes)
        else:
            self.num_q_tokens = 1
            self.query_tokens = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.query_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x, **kwargs):
        b, c, h, w, d = x.shape
        lp = tuple(l // p for l, p in zip([h, w, d], self.patch_size))

        x = self.prepare_tokens(self.channel_adapter(x))

        preds = []

        for i, blk in enumerate(self.blocks):

            if (self.query_tokens is not None) and (i == self.pred_from):
                x = torch.cat((self.query_tokens.repeat(b, 1, 1), x), dim=1)

            x = blk(x)

            if i >= self.pred_from:
                if self.task in ["segmentation", "none"]:
                    patch_tokens = self.norm(
                        x[:, self.num_q_tokens + self.num_register_tokens + 1 :, :]
                    )
                    spatial = patch_tokens.unflatten(1, lp).permute(0, -1, 1, 2, 3)
                    if self.task == "segmentation":
                        preds.append(self.head(spatial))
                    else:
                        query_latent = self.query_norm(x[:, : self.num_q_tokens, :])[
                            :, 0, :
                        ]  # [B, q, d]
                        preds.append((query_latent, spatial))
                else:
                    query_latent = self.query_norm(
                        x[:, : self.num_q_tokens, :]
                    )  # [B, q, d]
                    res = self.attn_head(query_latent).squeeze(1)
                    preds.append(res)
        return preds

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        state_dict = possibly_clean_lightning_sd(state_dict)

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def additional_trainable(self):
        return ["attn_head", "query_tokens", "query_norm", "head", "channel_adapter"]


def vitv2_a_3d_small(
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
            attn_class=(MemEffAttention if mea else Attention),
        ),
        num_register_tokens=0,
        med_in_channels=1,
        use_mask=False,
        use_patch_decode=False,
        **kwargs,
    )
    return model
