# Implementation Plan: 2D ViT Adaptation with Cross-Attention and Deep Supervision

## Context

The 2D ViT adaptation model (`med_adapt/models/extended/image.py`) is currently a stub — `__init__` and `forward` contain `...`. The 3D volume adaptation (`med_adapt/models/extended/volume.py`) already implements the target pattern: query tokens inserted mid-network, deep supervision (list of predictions), task-specific MLP heads, and 3D upscale for segmentation. The 2D variant must match this API while adding **cross-attention over 3D slices** — the key difference is that query tokens aggregate information across all depth slices rather than relying on 3D self-attention.

Additionally, the trainers currently assume a single output from `forward()`. With deep supervision (list of predictions), the loss must be weighted with `2^i` scaling where earlier predictions have lower weight.

## Files to Modify

| File | Change |
|------|--------|
| `med_adapt/layers/attention.py` | Add `CrossAttentionBlock` class |
| `med_adapt/layers/__init__.py` | Export `CrossAttentionBlock` |
| `med_adapt/models/extended/image.py` | Complete `ViTv2Adaption` implementation |
| `med_adapt/trainer/template.py` | Handle list outputs in `batch_to_loss()` |
| `med_adapt/trainer/classification.py` | Deep supervision loss |
| `med_adapt/trainer/regression.py` | Deep supervision loss |
| `med_adapt/trainer/segmentation.py` | Deep supervision loss |
| `med_adapt/utils/trainable.py` | Add `cross_attn` to default trainable keys |

---

## Step 1: Cross-Attention Block

**File**: `med_adapt/layers/attention.py`

Add `CrossAttentionBlock` after the existing `LoRAMemEffAttention` class. This is a standard cross-attention module (queries attend over keys/values from a different sequence) with the same interface pattern as the existing attention classes.

```python
class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, queries: Tensor, keys_values: Tensor, attn_bias=None) -> Tensor:
        """
        queries:      (B, Q, E)  — learnable volume-level query tokens
        keys_values:  (B, N, E)  — all tokens from all slices flattened (D * spatial)
        attn_bias:    kept for signature consistency but unused (always None)
        returns:      (B, Q, E)
        """
        B, Q, C = queries.shape
        _, N, _ = keys_values.shape

        q = self.q_proj(queries).reshape(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(keys_values).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(keys_values).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Q, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x
```

**File**: `med_adapt/layers/__init__.py` — add `CrossAttentionBlock` to imports and `__all__`.

---

## Step 2: Complete `ViTv2Adaption` in `image.py`

**File**: `med_adapt/models/extended/image.py`

### 2a. Imports

Add to existing imports:
- `import torch.nn as nn`
- `import torch.nn.functional as F`
- `from med_adapt.layers import Block, ScaleBlock, MemEffAttention, LoRAMemEffAttention, CrossAttentionBlock`

### 2b. `__init__`

Replace the `...` in `__init__` with:

```python
self.task = task

# Query tokens: one per volume (not per slice)
self.num_q_tokens = 1 if task == "classification" else classes
self.query_tokens = nn.Parameter(
    torch.zeros(1, self.num_q_tokens, self.embed_dim), requires_grad=True
)
nn.init.normal_(self.query_tokens, std=1e-6)

# Cross-attention blocks: one per transformer block from query_from onward
self.num_blocks = len(self.blocks)
self.query_from = kwargs.pop("query_from", -6)
self.query_from = (
    self.num_blocks + self.query_from if self.query_from < 0 else self.query_from
)
self.num_cross_attn_blocks = self.num_blocks - self.query_from

self.cross_attn_blocks = nn.ModuleList(
    [
        CrossAttentionBlock(
            dim=self.embed_dim,
            num_heads=self.num_heads,
        )
        for _ in range(self.num_cross_attn_blocks)
    ]
)

# Task-specific heads
if task == "segmentation":
    self.upscale = nn.Sequential(
        ScaleBlock(self.embed_dim, conv_type="3d"),
        ScaleBlock(self.embed_dim // 2, conv_type="3d"),
        ScaleBlock(self.embed_dim // 4, conv_type="3d"),
        ScaleBlock(self.embed_dim // 8, conv_type="3d"),
    )
    self.query_mlp = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
        nn.GELU(),
        nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
        nn.GELU(),
        nn.Linear(self.embed_dim // 4, self.embed_dim // 16, bias=False),
    )
elif task == "classification":
    self.query_mlp = nn.Sequential(
        nn.Linear(self.embed_dim, self.embed_dim, bias=True),
        nn.GELU(),
        nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
        nn.GELU(),
        nn.Linear(self.embed_dim // 4, classes, bias=False),
    )
else:  # regression
    self.query_mlp = nn.ModuleDict(
        {
            f"class_{i}": nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
                nn.GELU(),
                nn.Linear(self.embed_dim // 4, 1, bias=True),
            )
            for i in range(self.num_q_tokens)
        }
    )
```

### 2c. Fix `forward` — change `prepare_tokens_with_masks` to `prepare_tokens`

Line 69: `x = self.prepare_tokens_with_masks(reshaped_x)` → `x = self.prepare_tokens(reshaped_x)`

### 2d. Implement `forward`

Replace the `...` in `forward` with:

```python
def forward(self, x, **kwargs):
    b, c, h, w, d = x.shape

    adapted_x = self.input_adapter(x)
    reshaped_x = rearrange(adapted_x, "b c ... d -> (b d) c ...")

    x = self.prepare_tokens(reshaped_x)

    preds = []
    attn_bias = None

    for i, blk in enumerate(self.blocks):
        if i == self.query_from:
            # Insert query tokens repeated for each folded slice
            x = torch.cat((self.query_tokens.repeat(b * d, 1, 1), x), dim=1)

        x = blk(x, attn_bias=attn_bias)

        if i >= self.query_from:
            num_q = self.num_q_tokens
            num_reg = self.num_register_tokens

            # Split folded sequence
            query_tokens = x[:, :num_q, :]                                    # (B*D, Q, E)
            register_and_cls = x[:, num_q : num_q + num_reg + 1, :]           # (B*D, 1+num_reg, E)
            patch_tokens = x[:, num_q + num_reg + 1 :, :]                      # (B*D, N, E)

            # Aggregate queries across depth: (B*D, Q, E) -> (B, Q, E)
            queries = query_tokens.reshape(b, d, num_q, -1).mean(dim=1)

            # Flatten patches across depth: (B*D, N, E) -> (B, D*N, E)
            patches = patch_tokens.reshape(b, d, -1, -1).reshape(
                b, d * patch_tokens.shape[1], -1
            )

            # Cross-attention
            cross_idx = i - self.query_from
            queries = self.cross_attn_blocks[cross_idx](queries, patches)

            # Replace queries in sequence (repeat for each slice) for subsequent blocks
            x = torch.cat((queries.repeat(d, 1, 1), register_and_cls, patch_tokens), dim=1)

            # Generate prediction
            if self.task == "segmentation":
                seg_patches = x[:, num_q + num_reg + 1 :, :]  # (B*D, N, E)
                h_p = h // self.patch_size
                w_p = w // self.patch_size
                # Reshape to 3D volume: (B, D, H_p, W_p, E) -> (B, E, D, H_p, W_p)
                seg_patches = seg_patches.reshape(b, d, h_p, w_p, self.embed_dim)
                seg_patches = seg_patches.permute(0, 4, 1, 2, 3)

                upscaled = self.upscale(seg_patches)  # (B, E/16, D*16, H_p*16, W_p*16)
                mask_logits = F.interpolate(
                    upscaled, size=(h, w, d), mode="trilinear", align_corners=False
                )  # (B, E/16, h, w, d)

                query_logits = self.query_mlp(queries)  # (B, Q, E/16)
                seg_pred = torch.einsum("b c d h w, b q c -> b q d h w", mask_logits, query_logits)
                preds.append(seg_pred)

            elif self.task == "classification":
                cls_pred = self.query_mlp(queries)  # (B, classes)
                preds.append(cls_pred)

            else:  # regression
                reg_pred = [
                    self.query_mlp[f"class_{i}"](queries[:, i, :])
                    for i in range(self.num_q_tokens)
                ]
                preds.append(reg_pred)

    return preds
```

### 2e. Add `additional_trainable` and `do_not_load`

```python
def additional_trainable(self):
    return [
        "patch_embed",
        "pos_embed",
        "query_mlp",
        "query_tokens",
        "cross_attn_blocks",
        "upscale",
        "input_adapter",
    ]

def do_not_load(self):
    return ["pos_embed", "patch_embed"]
```

---

## Step 3: Trainer Deep Supervision

### 3a. `TemplateTrainer.batch_to_loss()` — `template.py`

Replace the current `batch_to_loss` with logic that handles list outputs:

```python
def batch_to_loss(self, batch, train=False):
    image, label = self.preprocess_batch(batch, train)
    outputs = self(image)

    if isinstance(outputs, list):
        # Deep supervision: weighted sum of per-block losses
        num_preds = len(outputs)
        total_loss = None
        for i, pred in enumerate(outputs):
            weight = 2 ** (i - (num_preds - 1))
            if isinstance(pred, list):
                # Regression: list of per-class tensors
                pred_loss = sum(self.criterion(p, label) for p in pred) / len(pred)
            else:
                pred_loss = self.criterion(pred, label)
            if total_loss is None:
                total_loss = weight * pred_loss
            else:
                total_loss = total_loss + weight * pred_loss
        logits = outputs[-1]
    else:
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        total_loss = self.criterion(logits, label)

    return total_loss, (logits, label)
```

### 3b. `ClassificationTrainer.batch_to_loss()` — `classification.py`

Override to handle list outputs (the label must be `.long()`):

```python
def batch_to_loss(self, batch, train=False):
    image, label = self.preprocess_batch(batch, train)
    label = label.long()

    outputs = self(image)

    if isinstance(outputs, list):
        num_preds = len(outputs)
        total_loss = None
        for i, pred in enumerate(outputs):
            weight = 2 ** (i - (num_preds - 1))
            loss = self.criterion(pred, label)
            total_loss = loss if total_loss is None else total_loss + weight * loss
        logits = outputs[-1]
    else:
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        total_loss = self.criterion(logits, label)

    return total_loss, (logits, label)
```

### 3c. `RegressionTrainer.batch_to_loss()` — `regression.py`

Same pattern, with `label.float()`:

```python
def batch_to_loss(self, batch, train=False):
    image, label = self.preprocess_batch(batch, train)
    label = label.float()

    outputs = self(image)

    if isinstance(outputs, list):
        num_preds = len(outputs)
        total_loss = None
        for i, pred in enumerate(outputs):
            weight = 2 ** (i - (num_preds - 1))
            if isinstance(pred, list):
                pred_loss = sum(self.criterion(p, label) for p in pred) / len(pred)
            else:
                pred_loss = self.criterion(pred, label)
            total_loss = pred_loss if total_loss is None else total_loss + weight * pred_loss
        logits = outputs[-1]
    else:
        logits = (
            outputs
            if isinstance(outputs, torch.Tensor)
            else outputs.get("logits", outputs)
        )
        total_loss = self.criterion(logits, label)

    return total_loss, (logits, label)
```

### 3d. `SegmentationTrainer.batch_to_loss()` — `segmentation.py`

Handle list outputs while preserving the DiceCE + presence BCE + deep supervision structure:

```python
def batch_to_loss(self, batch, train=False):
    image, label = self.preprocess_batch(batch, train)
    outputs = self(image)

    if isinstance(outputs, list):
        num_preds = len(outputs)
        # Final prediction
        final_seg = outputs[-1]  # (B, C, H, W, D)

        dice_ce_loss = self.criterion(final_seg, label)

        gt = label[:, 0].long() if label.ndim == 5 else label.long()
        presence_gt = torch.stack(
            [(gt == i).any(dim=(1, 2, 3)).float() for i in range(1, self.num_classes + 1)],
            dim=1,
        )

        # Presence logits from final segmentation (class 0 channel as presence proxy)
        final_presence = final_seg[:, 0, :, :, :] if final_seg.dim() == 5 else final_seg
        final_bce = self.bce_loss(final_presence, presence_gt)

        # Deep supervision: weighted intermediate presence losses
        inter_bce_sum = torch.tensor(0.0, device=label.device)
        for i, seg_pred in enumerate(outputs[:-1]):
            weight = 2 ** (i - (num_preds - 2))
            inter_presence = seg_pred[:, 0, :, :, :] if seg_pred.dim() == 5 else seg_pred
            inter_bce_sum = inter_bce_sum + weight * self.bce_loss(inter_presence, presence_gt)
        inter_bce = inter_bce_sum / max(num_preds - 1, 1) if num_preds > 1 else inter_bce_sum

        loss = dice_ce_loss + final_bce + 0.5 * inter_bce
        return {
            "loss": loss,
            "dice_ce": dice_ce_loss,
            "bce_final": final_bce,
            "bce_inter": inter_bce,
        }, (final_seg, label)
    else:
        # Original dict-based path (backward compat)
        final_seg_logits = outputs["seg_logits"]
        final_presence_logits = outputs["presence_logits"]
        intermediate = outputs.get("intermediate", [])

        dice_ce_loss = self.criterion(final_seg_logits, label)
        gt = label[:, 0].long() if label.ndim == 5 else label.long()
        presence_gt = torch.stack(
            [(gt == i).any(dim=(1, 2, 3)).float() for i in range(1, self.num_classes + 1)],
            dim=1,
        )
        final_bce = self.bce_loss(final_presence_logits, presence_gt)

        inter_bce = torch.tensor(0.0, device=label.device)
        for _, pres_logits in intermediate:
            inter_bce += self.bce_loss(pres_logits, presence_gt)
        inter_bce /= max(len(intermediate), 1)

        loss = dice_ce_loss + final_bce + 0.5 * inter_bce
        return {
            "loss": loss,
            "dice_ce": dice_ce_loss,
            "bce_final": final_bce,
            "bce_inter": inter_bce,
        }, (final_seg_logits, label)
```

---

## Step 4: Trainable Parameters

**File**: `med_adapt/utils/trainable.py`

Add `"cross_attn"` to the default `trainable_keys` tuple:

```python
trainable_keys: tuple[str, ...] = (
    "lora_A",
    "lora_B",
    "attn_pool",
    "head",
    "input_adapter",
    "upscale",
    "cross_attn",
),
```

---

## Verification

1. **Model forward test**: Run a forward pass with dummy input `(1, 3, 224, 224, 16)` for each task:
   ```python
   # Classification
   model = vitv2_a_2d_small(med_in_channels=3, task="classification", classes=2, lora=True)
   out = model(torch.randn(1, 3, 224, 224, 16))
   assert isinstance(out, list)
   assert out[-1].shape == (1, 2)

   # Segmentation
   model = vitv2_a_2d_small(med_in_channels=3, task="segmentation", classes=2, lora=True)
   out = model(torch.randn(1, 3, 224, 224, 16))
   assert isinstance(out, list)
   assert out[-1].shape == (1, 2, 224, 224, 16)

   # Regression
   model = vitv2_a_2d_small(med_in_channels=3, task="regression", classes=1, lora=True)
   out = model(torch.randn(1, 3, 224, 224, 16))
   assert isinstance(out, list)
   assert out[-1][0].shape == (1, 1)
   ```

2. **Trainer integration**: Verify `batch_to_loss` returns correct weighted loss for list outputs.

3. **Trainable params**: Verify `additional_trainable()` includes `cross_attn_blocks` and only those + backbone LoRA params are trainable.

4. **Existing tests**: `pytest tests/` should still pass (no breaking changes to base model or 3D model).

5. **Lint**: `make lint` should pass.
