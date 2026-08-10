# Implementation Plan: 2D ViT Adaptation with Cross-Attention and Deep Supervision

## Overview

Upgrade `med_adapt/models/extended/image.py` (`ViTv2Adaption`) to match the query-token + deep supervision pattern of `volume.py` (`ViTv2Adaption3D`), with the key architectural difference being **cross-attention over 3D slices** instead of simple self-attention on a 3D volume.

## Architecture Summary

### Current State
- **image.py**: Stub `ViTv2Adaption` with `...` in `__init__` and `forward`
- **volume.py**: Full reference `ViTv2Adaption3D` with query tokens, deep supervision, MLP heads, upscale
- **Base ViTv2**: `prepare_tokens`, `forward` returning dict with `latent`, `patch_latent`, `raw_latent`
- **Trainers**: Template expects single tensor/dict output; segmentation expects `{"seg_logits", "presence_logits", "intermediate"}`

### Target State
- Both 2D and 3D models return a **list of predictions** from deep supervision
- 2D model uses **cross-attention** to aggregate information across slices
- Unified trainer interface with exponential weighting for deep supervision

---

## Design Decisions

### 1. Cross-Attention Placement
**Decision**: Place `CrossAttentionBlock` in `med_adapt/layers/` alongside existing attention modules.

**Rationale**:
- `med_adapt/layers/attention.py` already contains `Attention`, `MemEffAttention`, `LoRAAttention`, `LoRAMemEffAttention`
- `med_adapt/adapter/` is for adapter-specific modules (InputChannelAdapter, PatchEmbed3D, AttentionPooling)
- Cross-attention is a general attention mechanism, not an adapter-specific construct
- Keeps the pattern consistent with other attention types

### 2. cls_token and Register_Tokens in Cross-Attention
**Decision**: Include cls_token and register_tokens in K/V but NOT in queries.

**Rationale**:
- Queries are learnable per-volume tokens representing the entire volume
- cls_token and register_tokens are per-slice tokens that should attend to cross-slice context
- However, the cross-attention is applied to **all** tokens after the transformer block, so cls/register tokens get updated via self-attention in subsequent blocks
- Simpler: queries attend over everything (cls + registers + patches) for maximum information gathering

### 3. Reshaping Patch Tokens for Segmentation Upscale
**Decision**: After cross-attention, extract patch tokens (excluding cls and registers), reshape from `(B, D*N, E)` to `(B, E, D, H_p, W_p)` where `H_p = H//patch_size`, `W_p = W//patch_size`, then pass through 3D ScaleBlock chain.

**Rationale**:
- The 2D patches from each slice need to be organized into a 3D spatial volume
- `ScaleBlock` with `conv_type="3d"` expects `(B, C, D, H, W)` input
- The patch tokens after cross-attention have shape `(B, D*N, E)` where `N = H_p * W_p`
- Reshape: `(B, D, H_p, W_p, E)` -> permute to `(B, E, D, H_p, W_p)`

### 4. Deep Supervision Weight Formula
**Decision**: Weight for prediction at index `i` (0-indexed from `query_from`) is `2^(i - (num_preds - 1))`.

**Formula**:
```
weight_i = 2^(i - (num_preds - 1))
```
Where `num_preds = num_blocks - query_from`.

**Examples** (for 6 predictions, indices 0-5):
- i=0: weight = 2^(-5) = 1/32 = 0.03125
- i=1: weight = 2^(-4) = 1/16 = 0.0625
- i=2: weight = 2^(-3) = 1/8 = 0.125
- i=3: weight = 2^(-2) = 1/4 = 0.25
- i=4: weight = 2^(-1) = 1/2 = 0.5
- i=5: weight = 2^0 = 1.0

**Sum of weights**: 1/32 + 1/16 + 1/8 + 1/4 + 1/2 + 1 = 1.96875

### 5. prepare_tokens Modification
**Decision**: Override `prepare_tokens` in `ViTv2Adaption` (already done in stub) but fix the call to `prepare_tokens_with_masks` -> `prepare_tokens`. No other changes needed.

**Rationale**:
- The base `prepare_tokens` already handles cls_token, pos_embed interpolation, and register_tokens
- The 2D model's `prepare_tokens` is already overridden to handle the depth-folded input `(B*D, C, H, W)`
- The call at line 69 of image.py references a non-existent method; fix to `prepare_tokens`

---

## Step-by-Step Implementation Plan

### Phase 1: Cross-Attention Layer

#### Step 1.1: Create `CrossAttentionBlock` in `med_adapt/layers/attention.py`

Add a new class `CrossAttentionBlock` that implements cross-attention with xformers support.

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
        queries: (B, Q, E)
        keys_values: (B, N, E)  -- N = D * spatial_tokens
        attn_bias: kept for signature consistency but unused (set to None)
        returns: (B, Q, E)
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

#### Step 1.2: Export `CrossAttentionBlock` from `med_adapt/layers/__init__.py`

Add to imports and `__all__`.

---

### Phase 2: 2D Model Implementation

#### Step 2.1: Complete `ViTv2Adaption.__init__` in `image.py`

Add after `self.input_adapter = InputChannelAdapter(in_channels=med_in_channels)`:

```python
self.task = task

# Query tokens: one per volume, not per slice
self.num_q_tokens = 1 if task == "classification" else classes
self.query_tokens = nn.Parameter(
    torch.zeros(1, self.num_q_tokens, self.embed_dim), requires_grad=True
)
nn.init.normal_(self.query_tokens, std=1e-6)

# Cross-attention blocks: one per transformer block after query_from
self.num_blocks = len(self.blocks)
self.query_from = kwargs.pop("query_from", -6)
self.query_from = self.num_blocks + self.query_from if self.query_from < 0 else self.query_from
self.num_cross_attn_blocks = self.num_blocks - self.query_from

self.cross_attn_blocks = nn.ModuleList([
    CrossAttentionBlock(
        dim=self.embed_dim,
        num_heads=self.num_heads,
    )
    for _ in range(self.num_cross_attn_blocks)
])

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
    self.query_mlp = nn.ModuleDict({
        f"class_{i}": nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim // 4, bias=True),
            nn.GELU(),
            nn.Linear(self.embed_dim // 4, 1, bias=True),
        )
        for i in range(self.num_q_tokens)
    })
```

#### Step 2.2: Fix `prepare_tokens` call in `forward`

Change line 69 from:
```python
x = self.prepare_tokens_with_masks(reshaped_x)
```
to:
```python
x = self.prepare_tokens(reshaped_x)
```

#### Step 2.3: Implement `forward` in `image.py`

```python
def forward(self, x, **kwargs):
    b, c, h, w, d = x.shape

    adapted_x = self.input_adapter(x)
    reshaped_x = rearrange(adapted_x, "b c ... d -> (b d) c ...")

    # Prepare tokens for folded input (B*D, C, H, W)
    x = self.prepare_tokens(reshaped_x)

    preds = []
    attn_bias = None

    for i, blk in enumerate(self.blocks):
        if i == self.query_from:
            # Insert query tokens repeated for each folded slice
            x = torch.cat((self.query_tokens.repeat(b * d, 1, 1), x), dim=1)

        x = blk(x, attn_bias=attn_bias)

        if i >= self.query_from:
            # Extract queries and patch tokens from folded sequence
            # x shape: (B*D, Q + num_register_tokens + 1 + N_patches, E)
            query_tokens = x[:, :self.num_q_tokens, :]           # (B*D, Q, E)
            register_and_cls = x[:, self.num_q_tokens:self.num_q_tokens + self.num_register_tokens + 1, :]
            patch_tokens = x[:, self.num_q_tokens + self.num_register_tokens + 1:, :]  # (B*D, N, E)

            # Cross-attention: aggregate across slices
            # Reshape queries: (B*D, Q, E) -> (B, D, Q, E) -> mean over D -> (B, Q, E)
            queries = query_tokens.reshape(b, d, self.num_q_tokens, self.embed_dim).mean(dim=1)

            # Reshape patches: (B*D, N, E) -> (B, D, N, E) -> flatten D*N -> (B, D*N, E)
            patches = patch_tokens.reshape(b, d, -1, self.embed_dim).reshape(b, d * patch_tokens.shape[1], self.embed_dim)

            # Run cross-attention
            cross_attn_idx = i - self.query_from
            queries = self.cross_attn_blocks[cross_attn_idx](queries, patches)

            # Replace queries in the sequence (repeat for each slice)
            x = torch.cat((
                queries.repeat(d, 1, 1),  # (B*D, Q, E)
                register_and_cls,
                patch_tokens
            ), dim=1)

            # Generate prediction
            if self.task == "segmentation":
                # Extract patch tokens for 3D upscale
                seg_patches = x[:, self.num_q_tokens + self.num_register_tokens + 1:, :]  # (B*D, N, E)
                # Reshape to 3D volume: (B, D, H_p, W_p, E) -> (B, E, D, H_p, W_p)
                h_p = h // self.patch_size
                w_p = w // self.patch_size
                seg_patches = seg_patches.reshape(b, d, h_p, w_p, self.embed_dim)
                seg_patches = seg_patches.permute(0, 4, 1, 2, 3)  # (B, E, D, H_p, W_p)

                # Upscale through 3D ScaleBlock chain
                upscaled = self.upscale(seg_patches)  # (B, E/16, D, H, W)
                # Final projection to class logits
                query_logits = self.query_mlp(queries)  # (B, classes, 1, 1) or (B, embed/16)
                # Actually, for segmentation, query_mlp outputs embed/16, need to match upscale output
                # The volume model uses einsum between mask_logits and query_logits
                # For 2D, we need to think about this differently...
                # Let's use the query as a scalar per class to weight the spatial output
                # Actually, looking at volume.py more carefully:
                # query_logits = self.query_mlp(x[:, :self.num_q_tokens, :]) -> (B, q, embed/16)
                # mask_logits = self._mask_logits(...) -> (B, d, embed/16) for each spatial position
                # segmentation_pred = einsum(mask_logits, query_logits, "b d ..., b q d -> b q ...")
                # For 2D: mask_logits is (B, E/16, D, H, W), query_logits is (B, Q, E/16)
                # We need einsum that gives (B, Q, D, H, W)
                # This is: sum over E/16 dimension
                # einsum("b c d h w, b q c -> b q d h w")
                mask_logits = upscaled
                query_logits = self.query_mlp(queries)  # (B, Q, embed/16)
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

**Wait**, I need to reconsider the segmentation path. Looking at the volume model more carefully:

In `volume.py`:
- `mask_logits = self._mask_logits(x[:, num_q_tokens + num_register_tokens + 1:], h, w, d)` returns `(B, embed//16, H, W, D)`
- `query_logits = self.query_mlp(x[:, :num_q_tokens, :])` returns `(B, q, embed//16)`
- `segmentation_pred = einsum(mask_logits, query_logits, "b d ..., b q d -> b q ...")`
  - This einsum is: `(B, embed//16, H, W, D)` @ `(B, q, embed//16)` -> `(B, q, H, W, D)`
  - The einsum contracts over the `embed//16` dimension

For 2D, after cross-attention:
- Patch tokens are `(B, D*N, E)` where `N = H_p * W_p`
- We reshape to `(B, D, H_p, W_p, E)` then permute to `(B, E, D, H_p, W_p)`
- After 4x ScaleBlock(3d): `(B, E/16, D, H_p, W_p)` -- wait, ScaleBlock upscales spatially by 2x each time
  - After 1st: `(B, E/2, D*2, H_p*2, W_p*2)` 
  - After 2nd: `(B, E/4, D*4, H_p*4, W_p*4)`
  - After 3rd: `(B, E/8, D*8, H_p*8, W_p*8)`
  - After 4th: `(B, E/16, D*16, H_p*16, W_p*16)`
  
Hmm, but `d` in the input is the number of slices, and the patch embed creates patches from 2D slices. The 3D upscale would need to know the original depth to produce the right output.

Actually, looking at the volume model's `_mask_logits`:
```python
def _mask_logits(self, patch_tokens, h, w, d):
    psh, psw, psd = self.patch_size
    hp, wp, dp = h // psh, w // psw, d // psd
    spatial = rearrange(patch_tokens, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp)
    upscaled = self.upscale(spatial)
    return F.interpolate(upscaled, size=(h, w, d), mode="trilinear", align_corners=False)
```

For the 2D model, the input is `(B, C, H, W, D)` where D is the number of slices. After folding, each slice is processed as a 2D image. The patch tokens per slice are `(H//patch_size) * (W//patch_size)`.

For segmentation in 2D, we need to produce `(B, classes, H, W, D)` output. The 3D upscale needs to operate on a 3D volume. But the patches from each slice are 2D patches. We need to organize them into a 3D tensor.

Let me reconsider. After cross-attention, patch tokens have shape `(B, D*N, E)` where `N = H_p * W_p`. We can reshape to `(B, D, H_p, W_p, E)` and then permute to `(B, E, D, H_p, W_p)`. This is a 3D volume with depth D and spatial dimensions H_p x W_p.

After 4x ScaleBlock(3d):
- Each ScaleBlock doubles spatial dimensions and halves channels
- Output: `(B, E/16, D*16, H_p*16, W_p*16)`

But we want output `(B, classes, H, W, D)` where H, W are the original image dimensions and D is the original depth.

Wait, the ScaleBlock with conv_type="3d" uses stride=2 for ConvTranspose3d, so each block doubles spatial dimensions in ALL 3 dimensions (D, H, W). This means:
- Input: `(B, E, D, H_p, W_p)`
- After 1st ScaleBlock: `(B, E/2, D*2, H_p*2, W_p*2)`
- After 2nd: `(B, E/4, D*4, H_p*4, W_p*4)`
- After 3rd: `(B, E/8, D*8, H_p*8, W_p*8)`
- After 4th: `(B, E/16, D*16, H_p*16, W_p*16)`

For this to match the target `(B, classes, H, W, D)`, we need:
- `D*16 == D` => This only works if D=0 which is impossible
- `H_p*16 == H` => `H_p = H/16` which means patch_size = 16, and H_p = H/16, so H_p*16 = H. This works!
- `W_p*16 == W` => Similarly works if patch_size = 16

But `D*16 == D` is only true for D=0. This is a problem.

Actually, wait. Let me re-read the volume model. In the 3D model, the input is a 3D volume `(B, C, H, W, D)` and the patch embed creates 3D patches. The number of patches in the depth dimension is `D // patch_size[2]`. So after patch embed, the sequence length is `(H//ps[0]) * (W//ps[1]) * (D//ps[2])`.

For the 2D model, after depth folding, each slice is processed independently as a 2D image. The patches are 2D patches. When we regroup them after cross-attention, we have `D * (H//ps) * (W//ps)` "virtual 3D patches". But the depth dimension is not "patched" in the same way -- each slice is its own 2D image.

So for the 2D segmentation case, we should NOT use 3D ScaleBlocks that upscale the depth dimension. Instead, we should either:
1. Use 2D ScaleBlocks per slice and then regroup
2. Use 3D ScaleBlocks but only upscale H and W, keeping D fixed

Actually, looking at the requirements again:
> "For segmentation task, the upscale must use conv_type='3d' (ScaleBlock with 3D convs). This produces full 3D volume logits, not per-slice"

So the requirement explicitly says to use 3D convs. But the depth dimension issue remains. Let me think about this differently.

In the 2D model, after cross-attention, the patch tokens represent the spatial patches from all slices. If we reshape to `(B, E, D, H_p, W_p)`, the 3D upscale will upsample in all 3 dimensions. But we want the output to have the same depth D as the input.

One solution: Use a different approach where we don't upsample the depth dimension. We could:
1. Apply cross-attention
2. Extract queries and patch tokens
3. For segmentation: reshape patches to `(B, E, D, H_p, W_p)`, apply 3D upscale but with a final interpolation to match the original depth
4. Or: use 2D upscale per slice and then regroup

Actually, re-reading the requirements more carefully:
> "For segmentation task, the upscale must use conv_type='3d' (ScaleBlock with 3D convs). This produces full 3D volume logits, not per-slice"

The key insight is that the 3D upscale is meant to produce a full 3D volume. But the depth dimension in the 2D model is the number of slices, which should remain the same. The 3D convs will naturally upsample the depth dimension too.

Let me think about what the actual input dimensions are. In medical imaging:
- `D` (depth/slices) might be something like 32, 64, 128
- `H`, `W` (spatial) might be 512, 512 or 224, 224
- `patch_size` is typically 14 or 16

If patch_size = 14 and H = W = 224, then H_p = W_p = 16.
After 4x ScaleBlock(3d) with stride 2:
- Spatial: 16 * 16 = 256 -> but we want 224. Hmm, that's close but not exact.
- Actually, with patch_size=14 and H=224, H_p=16, and 4 upsamples of 2x each = 16x, so 16*16 = 256. But the original H is 224. This doesn't match.

Wait, let me re-check. In the volume model:
- `patch_size` for 3D might be `(14, 14, 2)` as shown in the factory functions
- `volume_size` is `(224, 224, 32)`
- So `hp = 224//14 = 16`, `wp = 224//14 = 16`, `dp = 32//2 = 16`
- After 4x ScaleBlock(3d): spatial becomes `16*16 = 256` for H and W, and `16*16 = 256` for D
- But the original volume is `(224, 224, 32)`. The `_mask_logits` then does `F.interpolate(..., size=(h, w, d), mode="trilinear")` to resize to the original dimensions.

Ah! I missed the final interpolation in `_mask_logits`. Let me re-read:
```python
def _mask_logits(self, patch_tokens, h, w, d) -> Tensor:
    psh, psw, psd = self.patch_size
    hp, wp, dp = h // psh, w // psw, d // psd
    spatial = rearrange(patch_tokens, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp)
    upscaled = self.upscale(spatial)
    return F.interpolate(upscaled, size=(h, w, d), mode="trilinear", align_corners=False)
```

So the 3D upscale produces a volume at a lower resolution, and then `F.interpolate` upsamples it to the original `(h, w, d)` size. This is the key!

For the 2D model, we should do the same:
1. Reshape patch tokens to `(B, E, D, H_p, W_p)`
2. Apply 4x ScaleBlock(3d) -> `(B, E/16, D*16, H_p*16, W_p*16)`
3. Use `F.interpolate` to resize to `(B, num_classes, D, H, W)`

Wait, but `D*16` would be much larger than the original D. For example, if D=32, then D*16 = 512. Then interpolate back to D=32. That's a 16x downsampling in the depth dimension, which seems wrong.

Hmm, let me reconsider. Maybe for the 2D model, the "depth" dimension in the patch token sequence is not a true 3D patch dimension but rather a sequence of 2D patches from different slices. The 3D upscale might not be the right approach for the depth dimension.

Actually, I think the issue is that the requirements say to use 3D convs for the upscale, but the depth dimension handling needs to be different. Let me re-read the requirements:

> "For segmentation task, the upscale must use conv_type='3d' (ScaleBlock with 3D convs). This produces full 3D volume logits, not per-slice"

I think the intent is that the 3D upscale operates on the spatial dimensions (H, W) and the slice dimension (D) together, treating the folded patches as a 3D volume. The final interpolation then resizes to the original volume size.

But the math doesn't work out cleanly because the depth dimension gets upsampled by 16x and then needs to be downsampled back. This is wasteful but might be acceptable if the cross-attention has already mixed information across slices.

Alternatively, maybe the intent is simpler: just use 3D convs for the spatial upscaling and let the depth dimension be handled naturally. The final interpolation takes care of getting back to the right size.

Let me proceed with this interpretation and note it as a design decision.

Actually, wait. Let me reconsider the patch token arrangement. In the 2D model:
- Input: `(B, C, H, W, D)` -- B volumes, C channels, HxW spatial, D slices
- After input_adapter and fold: `(B*D, C, H, W)`
- After patch_embed: `(B*D, H_p*W_p, E)` where `H_p = H//patch_size`, `W_p = W//patch_size`
- After adding cls and registers: `(B*D, 1 + num_register_tokens + H_p*W_p, E)`

After cross-attention at block i:
- Queries: `(B, Q, E)` -- aggregated across slices
- Patch tokens: `(B, D*H_p*W_p, E)` -- all patches from all slices

For segmentation, we need to produce `(B, classes, H, W, D)`.

One approach:
1. Reshape patch tokens to `(B, D, H_p, W_p, E)`
2. Permute to `(B, E, D, H_p, W_p)`
3. Apply 4x ScaleBlock(3d) -> `(B, E/16, D*16, H_p*16, W_p*16)`
4. `F.interpolate` to `(B, classes, D, H, W)` -- but wait, we need to project channels to classes first
5. Actually, the query_mlp produces `(B, Q, E/16)` for segmentation
6. Then we need to combine query logits with spatial logits

Looking at volume.py again:
```python
mask_logits = self._mask_logits(...)  # (B, d, embed//16, H, W, D) -- wait, let me re-check
query_logits = self.query_mlp(x[:, :self.num_q_tokens, :])  # (B, q, embed//16)
segmentation_pred = einsum(mask_logits, query_logits, "b d ..., b q d -> b q ...")
```

Wait, `_mask_logits` in volume.py:
```python
def _mask_logits(self, patch_tokens, h, w, d) -> Tensor:
    psh, psw, psd = self.patch_size
    hp, wp, dp = h // psh, w // psw, d // psd
    spatial = rearrange(patch_tokens, "b (hp wp dp) c -> b c hp wp dp", hp=hp, wp=wp, dp=dp)
    upscaled = self.upscale(spatial)
    return F.interpolate(upscaled, size=(h, w, d), mode="trilinear", align_corners=False)
```

So `spatial` is `(B, E, hp, wp, dp)` and `upscaled` is `(B, E/16, hp*16, wp*16, dp*16)`. Then interpolate to `(B, E/16, h, w, d)`.

Wait, but `hp*16` might not equal `h`. For example, if `h = 224`, `psh = 14`, then `hp = 16`, and `hp*16 = 256`. But `h = 224`. So the upscale produces a slightly larger volume, and then interpolate brings it back to the original size.

For the 2D model, the same logic applies but we need to handle the depth dimension differently. The 2D model's patches are 2D, not 3D. So `hp = H//patch_size`, `wp = W//patch_size`, but there's no `dp` because the patches are 2D.

For the 2D segmentation, after cross-attention, the patch tokens are `(B, D*hp*wp, E)`. We can treat this as a 3D volume with dimensions `(hp, wp, D)`:
```python
spatial = rearrange(patch_tokens, "b (hp wp d) c -> b c hp wp d", hp=hp, wp=wp, d=d)
```

Then apply 3D upscale:
```python
upscaled = self.upscale(spatial)  # (B, E/16, hp*16, wp*16, d*16)
```

Then interpolate to original size:
```python
mask_logits = F.interpolate(upscaled, size=(h, w, d), mode="trilinear", align_corners=False)
# (B, E/16, h, w, d)
```

Then combine with query logits:
```python
query_logits = self.query_mlp(queries)  # (B, Q, E/16)
seg_pred = torch.einsum("b c h w d, b q c -> b q h w d", mask_logits, query_logits)
# (B, Q, h, w, d)
```

This matches the volume model's pattern! The key difference is that in the 2D model, the "depth" dimension of the 3D volume is the slice dimension D, not a true 3D patch dimension. But the math works out the same way.

However, there's still the issue that `d*16` might be much larger than the original `d`. For example, if `d = 32`, then `d*16 = 512`, and then interpolate brings it back to 32. This is a 16x upsampling followed by a 16x downsampling, which is wasteful but functionally correct.

Actually, I realize there might be a simpler interpretation. Maybe for the 2D model, we should use 2D ScaleBlocks for the spatial dimensions and handle the depth dimension separately. But the requirements explicitly say to use 3D convs.

Let me just follow the requirements and use 3D ScaleBlocks, with the understanding that the final interpolation handles the size mismatch.

OK, let me finalize the plan. I'll write it to the plan file now.
</think>

<tool_call>
<function=Bash>
<parameter=command>
mkdir -p /mnt/c/Users/puruv/Projects/fomo26/plans
