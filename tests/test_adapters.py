import torch
import torch.nn as nn

from med_adapt.adapter import (
    InputChannelAdapter,
    AttentionPooling,
    PatchEmbed3D,
    TaskTokens,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestInputChannelAdapter:
    def test_forward_shape_1_to_3(self):
        adapter = InputChannelAdapter(in_channels=1, out_channels=3).to(DEVICE)
        x = torch.randn(2, 1, 16, 16, 16, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 3, 16, 16, 16)

    def test_forward_shape_3_to_3(self):
        adapter = InputChannelAdapter(in_channels=3, out_channels=3).to(DEVICE)
        x = torch.randn(2, 3, 16, 16, 16, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 3, 16, 16, 16)

    def test_forward_shape_4_to_3(self):
        adapter = InputChannelAdapter(in_channels=4, out_channels=3).to(DEVICE)
        x = torch.randn(2, 4, 16, 16, 16, device=DEVICE)
        out = adapter(x)
        assert out.shape == (2, 3, 16, 16, 16)


class TestAttentionPooling:
    def test_forward_shape(self):
        pool = AttentionPooling(dim=768, num_heads=8).to(DEVICE)
        x = torch.randn(4, 196, 768, device=DEVICE)
        out = pool(x)
        assert out.shape == (4, 768)

    def test_batch_size_1(self):
        """Regression test: squeeze(1) bug must not collapse batch dim."""
        pool = AttentionPooling(dim=64, num_heads=4).to(DEVICE)
        x = torch.randn(1, 49, 64, device=DEVICE)
        out = pool(x)
        assert out.shape == (1, 64), f"Expected (1, 64), got {out.shape}"

    def test_with_mask(self):
        pool = AttentionPooling(dim=128, num_heads=8).to(DEVICE)
        x = torch.randn(2, 100, 128, device=DEVICE)
        mask = torch.ones(2, 100, dtype=torch.bool, device=DEVICE)
        mask[:, 80:] = False
        out = pool(x, mask=mask)
        assert out.shape == (2, 128)


class TestPatchEmbed3D:
    def test_flattened_output(self):
        embed = PatchEmbed3D(
            img_size=(32, 64, 64), patch_size=(4, 8, 8), in_chans=1, embed_dim=64
        ).to(DEVICE)
        x = torch.randn(2, 1, 32, 64, 64, device=DEVICE)
        out = embed(x)
        assert out.shape == (2, 8 * 8 * 8, 64)

    def test_unflattened_output(self):
        embed = PatchEmbed3D(
            img_size=(32, 64, 64),
            patch_size=(4, 8, 8),
            in_chans=1,
            embed_dim=64,
            flatten_embedding=False,
        ).to(DEVICE)
        x = torch.randn(2, 1, 32, 64, 64, device=DEVICE)
        out = embed(x)
        assert out.shape == (2, 8, 8, 8, 64)


class TestTaskTokens:
    def test_forward_beginning(self):
        tokens = TaskTokens(num_tokens=2, embed_dim=64, insertion="beginning").to(
            DEVICE
        )
        x = torch.randn(4, 10, 64, device=DEVICE)
        out = tokens(x)
        # Should prepend tokens after CLS position: (4, 1+2+9, 64) = (4, 12, 64)
        assert out.shape == (4, 12, 64)

    def test_forward_middle(self):
        tokens = TaskTokens(num_tokens=2, embed_dim=64, insertion="middle").to(DEVICE)
        x = torch.randn(4, 10, 64, device=DEVICE)
        out = tokens(x, block_index=5, num_blocks=12)
        assert out.shape == (4, 12, 64)

    def test_additional_trainable_keys(self):
        tokens = TaskTokens(num_tokens=2, embed_dim=64)
        keys = tokens.additional_trainable_keys()
        assert "task_tokens" in keys

    def test_parameters_trainable(self):
        tokens = TaskTokens(num_tokens=2, embed_dim=64).to(DEVICE)
        assert tokens.tokens.requires_grad
