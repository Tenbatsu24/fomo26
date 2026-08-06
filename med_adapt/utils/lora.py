import re

from typing import Mapping

import torch

from torch import nn

from med_adapt.layers import MemEffAttention
from med_adapt.layers.attention import LoRAMemEffAttention, LoRALinear
from med_adapt.utils.config import get_logger

logger = get_logger(__name__)


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    qkv_names=("qkv", "proj"),
    to_lora: bool = True,
) -> dict:
    """
    Remap a state dict's keys between plain Attention/MemEffAttention naming
    and LoRA(MemEff)Attention naming, without loading it into any model.
    Works both for nested modules (e.g. "blocks.0.attn.qkv.weight") and for
    a bare attention module used standalone (e.g. "qkv.weight").

    to_lora=True:  plain -> LoRA        (adds ".base" before weight/bias)
    to_lora=False: LoRA  -> plain       (strips ".base", drops lora_A/lora_B)
    """
    name_group = "|".join(map(re.escape, qkv_names))

    if to_lora:
        # (?:.*\.)?  -- optional "parent." prefix, so "qkv.weight" and
        # "blocks.0.attn.qkv.weight" both match
        pattern = re.compile(
            r"^(?P<prefix>(?:.*\.)?(?:" + name_group + r"))\.(?P<suffix>weight|bias)$"
        )
        new_state_dict = {}
        for key, value in state_dict.items():
            m = pattern.match(key)
            new_key = (
                f"{m.group('prefix')}.base.{m.group('suffix')}"
                if m and ("patch_embed" not in key)
                else key
            )
            new_state_dict[new_key] = value
        return new_state_dict

    else:
        base_pattern = re.compile(
            r"^(?P<prefix>(?:.*\.)?(?:"
            + name_group
            + r"))\.base\.(?P<suffix>weight|bias)$"
        )
        lora_pattern = re.compile(r"^.*\.(lora_A|lora_B)$")
        new_state_dict = {}
        for key, value in state_dict.items():
            if lora_pattern.match(key):
                continue
            m = base_pattern.match(key)
            new_key = f"{m.group('prefix')}.{m.group('suffix')}" if m else key
            new_state_dict[new_key] = value
        return new_state_dict


def merge_all_lora(model: nn.Module) -> int:
    """
    Recursively find every LoRALinear submodule in `model` and merge its
    LoRA update into the base weight in place -- no need to know the model's
    structure or traverse layers/blocks yourself.

    After this call, the model behaves as if it were the plain (non-LoRA)
    architecture with the trained deltas baked in; you can then export via
    convert_state_dict(model.state_dict(), to_lora=False) and load into a
    plain Attention model.

    Returns:
        Number of LoRALinear modules merged (useful as a sanity check that
        it found what you expected).
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_into_base()
            count += 1
    return count


def load_lora_state_dict(
    model: nn.Module,
    state_dict: dict,
    strict: bool = False,
    ignore_loading: list[str] | None = None,
) -> tuple:
    """
    Convenience wrapper: remaps a plain-Attention checkpoint's keys to the
    LoRA-wrapped naming scheme, then loads it into `model` (a model built
    with LoRA(MemEff)Attention modules).

    Since the checkpoint has no lora_A/lora_B entries, `strict=False` is the
    default -- missing_keys will list the (freshly initialized) LoRA params,
    which is expected and fine. unexpected_keys should be empty; if it's not,
    double check `qkv_names` matches the Linear submodule names actually used
    in your attention module.

    Returns:
        (missing_keys, unexpected_keys) as returned by `load_state_dict`.
    """
    remapped = convert_state_dict(state_dict, to_lora=True)
    if ignore_loading is not None:
        filtered_state_dict = {
            k: v
            for k, v in remapped.items()
            if all(ig not in k for ig in ignore_loading)
        }
    else:
        filtered_state_dict = remapped
    result = model.load_state_dict(filtered_state_dict, strict=strict)

    missing_lora = [k for k in result.missing_keys if "lora_A" in k or "lora_B" in k]
    missing_other = [k for k in result.missing_keys if k not in missing_lora]

    if missing_other:
        logger.warning(
            "[load_lora_state_dict] WARNING: missing non-LoRA keys (unexpected): %s",
            missing_other,
        )
    if result.unexpected_keys:
        logger.warning(
            "[load_lora_state_dict] WARNING: unexpected keys: %s",
            result.unexpected_keys,
        )
    logger.info(
        "[load_lora_state_dict] OK -- %d LoRA params left at random init, "
        "%d base params loaded from checkpoint.",
        len(missing_lora),
        len(remapped) - len(missing_other),
    )

    return result.missing_keys, result.unexpected_keys


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # simulate a "pretrained" plain-attention model + checkpoint
    plain_model = MemEffAttention(dim=32, num_heads=4).to(device)
    plain_ckpt = plain_model.state_dict()
    logger.info("plain ckpt keys: %s", list(plain_ckpt.keys()))

    # build the LoRA version and load the plain checkpoint into it
    lora_model = LoRAMemEffAttention(dim=32, num_heads=4, lora_r=4).to(device)
    lora_state_dict = lora_model.state_dict()
    logger.info("lora ckpt keys: %s", list(lora_state_dict.keys()))

    missing, unexpected = load_lora_state_dict(lora_model, plain_ckpt)

    # sanity check: base weights match exactly, LoRA path is a no-op (B=0),
    # so outputs should be identical to the plain model right after loading
    x = torch.randn(4, 16, 32, device=device)
    out_plain = plain_model(x)
    out_lora = lora_model(x)
    logger.info(
        "max abs diff (should be ~0): %.6f", (out_plain - out_lora).abs().max().item()
    )

    # round trip back to plain naming
    n = merge_all_lora(lora_model)
    logger.info("merged %d LoRALinear layers", n)

    plain_state = convert_state_dict(lora_model.state_dict(), to_lora=False)
    plain_model.load_state_dict(plain_state, strict=True)
    logger.info("round-trip load into plain model: OK")
