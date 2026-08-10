"""Tests for the ONNX export flow.

Verifies that a LoRA-wrapped model can be converted to a plain
(non-LoRA, non-MemEffAttention) model and exported to ONNX without
errors or key mismatches.
"""

import pytest
import torch

from med_adapt.registry import STORE
from med_adapt.utils.lora import convert_state_dict, merge_all_lora


@pytest.fixture(scope="module")
def lora_model_2d():
    """Build a small 2-D model with LoRA + MemEffAttention."""
    builder = STORE.get("models", "vitv2_a_2d_tiny")
    model = builder(
        med_in_channels=1,
        task="classification",
        classes=2,
        lora=True,
        mea=True,
    )
    return model


@pytest.fixture(scope="module")
def plain_model_2d():
    """Build the equivalent plain model (no LoRA, no MemEffAttn)."""
    builder = STORE.get("models", "vitv2_a_2d_tiny")
    model = builder(
        med_in_channels=1,
        task="classification",
        classes=2,
        lora=False,
        mea=False,
    )
    return model


class TestConvertStateDict:
    """Tests for ``convert_state_dict`` round-tripping."""

    def test_lora_to_plain_removes_lora_keys(self, lora_model_2d):
        """LoRA→plain conversion drops lora_A/lora_B entries."""
        sd = lora_model_2d.state_dict()
        plain_sd = convert_state_dict(sd, to_lora=False)

        lora_keys = [k for k in plain_sd if "lora_A" in k or "lora_B" in k]
        assert len(lora_keys) == 0, f"Unexpected LoRA keys remain: {lora_keys[:5]}"

    def test_lora_to_plain_adds_base_suffix(self, lora_model_2d):
        """LoRA→plain conversion strips '.base' from qkv/proj weights."""
        sd = lora_model_2d.state_dict()
        plain_sd = convert_state_dict(sd, to_lora=False)

        # Plain model should have keys like "blocks.0.attn.qkv.weight"
        plain_qkv = [k for k in plain_sd if "qkv.weight" in k and ".base." not in k]
        assert len(plain_qkv) > 0, "No plain qkv.weight keys found after conversion"

        # LoRA model should have keys like "blocks.0.attn.qkv.base.weight"
        lora_qkv = [k for k in sd if "qkv.base.weight" in k]
        assert len(lora_qkv) > 0, "No LoRA qkv.base.weight keys found"

    def test_plain_to_lora_adds_base_suffix(self, plain_model_2d):
        """Plain→LoRA conversion adds '.base' to qkv/proj weights."""
        sd = plain_model_2d.state_dict()
        lora_sd = convert_state_dict(sd, to_lora=True)

        lora_qkv = [k for k in lora_sd if "qkv.base.weight" in k]
        assert len(lora_qkv) > 0, "No qkv.base.weight keys found after conversion"

    def test_round_trip_state_dict(self, lora_model_2d, plain_model_2d):
        """LoRA → plain → LoRA round-trip preserves all original keys."""
        sd = lora_model_2d.state_dict()
        plain_sd = convert_state_dict(sd, to_lora=False)
        back_sd = convert_state_dict(plain_sd, to_lora=True)

        # Drop lora_A/lora_B from the original (they are random init)
        original_keys = {k for k in sd if "lora_A" not in k and "lora_B" not in k}
        roundtrip_keys = set(back_sd.keys())

        assert original_keys == roundtrip_keys, (
            f"Key mismatch after round-trip. "
            f"Missing: {original_keys - roundtrip_keys}, "
            f"Extra: {roundtrip_keys - original_keys}"
        )


class TestLoadIntoPlainModel:
    """Tests for loading converted weights into a plain model."""

    def test_load_converted_weights_no_errors(self, lora_model_2d, plain_model_2d):
        """Converting LoRA weights and loading into plain model is clean."""
        lora_sd = lora_model_2d.state_dict()
        plain_sd = convert_state_dict(lora_sd, to_lora=False)

        result = plain_model_2d.load_state_dict(plain_sd, strict=True)
        assert result.missing_keys == [], f"Missing keys: {result.missing_keys}"
        assert (
            result.unexpected_keys == []
        ), f"Unexpected keys: {result.unexpected_keys}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="MemEffAttention requires CUDA; skip on CPU",
    )
    def test_forward_pass_matches_after_conversion(self, lora_model_2d, plain_model_2d):
        """Outputs are identical after merging LoRA and converting to plain."""
        device = "cuda"
        lora_model_2d = lora_model_2d.to(device)
        plain_model_2d = plain_model_2d.to(device)

        # Merge LoRA into the source model so the forward pass is deterministic
        merge_all_lora(lora_model_2d)
        lora_model_2d.eval()
        plain_model_2d.eval()

        # Convert and load into plain model
        plain_sd = convert_state_dict(lora_model_2d.state_dict(), to_lora=False)
        plain_model_2d.load_state_dict(plain_sd, strict=True)

        x = torch.randn(1, 1, 64, 64, 8, device=device)
        with torch.no_grad():
            out_lora = lora_model_2d(x)
            out_plain = plain_model_2d(x)

        # Both return a list with one tensor for classification
        assert torch.allclose(out_lora[0], out_plain[0], atol=1e-5), (
            f"Output mismatch: max abs diff = "
            f"{(out_lora[0] - out_plain[0]).abs().max().item():.6f}"
        )


class TestOnnxExport:
    """Tests that the plain model can be exported to ONNX."""

    def test_onnx_export_2d(self, tmp_path):
        """Export converted plain model to ONNX without errors."""
        import torch.onnx

        from med_adapt.registry import STORE

        builder = STORE.get("models", "vitv2_a_2d_tiny")
        lora_model = builder(
            med_in_channels=1,
            task="classification",
            classes=2,
            lora=True,
            mea=True,
        )
        plain_model = builder(
            med_in_channels=1,
            task="classification",
            classes=2,
            lora=False,
            mea=False,
        )

        # Convert and load
        plain_sd = convert_state_dict(lora_model.state_dict(), to_lora=False)
        plain_model.load_state_dict(plain_sd, strict=True)
        plain_model.eval()

        dummy_input = torch.randn(1, 1, 64, 64, 8)
        onnx_path = tmp_path / "model.onnx"

        torch.onnx.export(
            plain_model,
            dummy_input,
            str(onnx_path),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
        )

        assert onnx_path.exists(), "ONNX file was not created"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"

    def test_onnx_export_3d(self, tmp_path):
        """Export a 3-D plain model to ONNX without errors."""
        import torch.onnx

        from med_adapt.registry import STORE

        builder = STORE.get("models", "vitv2_a_3d_tiny")
        model = builder(
            volume_size=(64, 64, 16),
            volume_patch_size=(14, 14, 2),
            med_in_channels=1,
            task="classification",
            classes=2,
            lora=False,
            mea=False,
        )
        model.eval()

        dummy_input = torch.randn(1, 1, 64, 64, 16)
        onnx_path = tmp_path / "model_3d.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch"},
                "output": {0: "batch"},
            },
        )

        assert onnx_path.exists(), "ONNX file was not created"
        assert onnx_path.stat().st_size > 0, "ONNX file is empty"
