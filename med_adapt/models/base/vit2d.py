# Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math

from functools import partial
from typing import Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import Parameter

from torch.nn.init import trunc_normal_
from torch.nn.functional import interpolate

from med_adapt.layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    Block,
    LayerScale,
    RMSNorm,
)
from med_adapt.utils.config import get_logger
from med_adapt.registry import register_model

logger = get_logger(__name__)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, attn_bias=None):
        # Adaptation for returing attentions
        for i, b in enumerate(self):
            x = b(x, attn_bias=attn_bias)
        return x


class ViTv2(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.1,
        drop_path_uniform=True,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.img_size = img_size
        self.in_channels = in_chans
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim

        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens

        self.interpolate_offset = interpolate_offset
        self.interpolate_antialias = interpolate_antialias

        self.patch_embed = self.intialise_patch_embed(embed_layer)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = self.intialise_pos_embed()

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.mask_token = None
        self.norm = norm_layer(embed_dim)

        # Initialize the model's weights
        self.init_weights()

    def intialise_patch_embed(self, embed_layer) -> Any:
        patch_embed = embed_layer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=self.embed_dim,
        )
        return patch_embed

    def intialise_pos_embed(self) -> Parameter:
        pos_embed = nn.Parameter(
            torch.zeros(
                1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim
            )
        )
        return pos_embed

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        if self.mask_token is not None:
            nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(
                0, 3, 1, 2
            ),
            scale_factor=(sx, sy),
            mode="bicubic",
            # antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
            previous_dtype
        )

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward(self, x, distill_from=-1, return_dict=False, **kwargs):
        *_, h, w = x.shape
        h_p, w_p = h // self.patch_size, w // self.patch_size

        x = self.prepare_tokens(x)

        resolved_idx = (
            self.n_blocks + distill_from if distill_from < 0 else distill_from
        )

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i >= resolved_idx:
                cls_token = self.norm(x[:, : self.num_register_tokens + 1])[:, 0]
                patch_tokens = (
                    self.norm(x[:, self.num_register_tokens + 1 :])
                    .unflatten(1, (h_p, w_p))
                    .permute(0, -1, 1, 2)
                )

                outs.append((cls_token, patch_tokens))

        if return_dict:
            return {"latent": outs[-1][0], "patch_latent": outs[-1][1]}

        return outs

    def additional_trainable(self):
        return None

    def do_not_load(self):
        return None


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


@register_model("vitv2_tiny")
def vitv2_tiny(patch_size=16, num_register_tokens=4, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


@register_model("vitv2_small")
def vitv2_small(patch_size=16, num_register_tokens=4, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


@register_model("vitv2_base")
def vitv2_base(patch_size=16, num_register_tokens=4, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 0.1
    model = ViTv2(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


@register_model("vitv2_large")
def vitv2_large(patch_size=16, num_register_tokens=4, **kwargs):
    if "init_values" not in kwargs:
        kwargs["init_values"] = 1e-5
    model = ViTv2(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
