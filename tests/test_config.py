"""Tests for med_adapt.utils.config."""

import json
import pytest
from pathlib import Path

from ml_collections import ConfigDict

from med_adapt.utils.config import load_json_config, get_config


class TestLoadJsonConfig:
    def test_load_basic(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text(
            json.dumps({"model": {"size": "base"}, "data": {"batch_size": 8}})
        )
        result = load_json_config(cfg)
        assert result["model"]["size"] == "base"
        assert result["data"]["batch_size"] == 8

    def test_load_relative_path(self, tmp_path, monkeypatch):
        cfg = tmp_path / "test.json"
        cfg.write_text(json.dumps({"model": {"size": "tiny"}}))
        monkeypatch.setenv("MED_ADAPT_CONFIGS", str(tmp_path))
        result = load_json_config("test.json")
        assert result["model"]["size"] == "tiny"

    def test_load_empty_file(self, tmp_path):
        cfg = tmp_path / "empty.json"
        cfg.write_text("{}")
        result = load_json_config(cfg)
        assert result == {}

    def test_load_null_values(self, tmp_path):
        cfg = tmp_path / "nulls.json"
        cfg.write_text(json.dumps({"data": {"resample_spacing": None}}))
        result = load_json_config(cfg)
        assert result["data"]["resample_spacing"] is None


class TestGetConfig:
    def test_returns_configdict(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert isinstance(result, ConfigDict)

    def test_defaults_are_set(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.model.variant == "2d"
        assert result.model.size == "small"
        assert result.model.lora is False
        assert result.data.batch_size == 4
        assert result.data.num_workers == 1
        assert result.optimizer.type == "AdamW"
        assert result.trainer.max_steps == 250
        assert result.trainer.precision == "32-true"
        assert result.seed == 42

    def test_json_overrides_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text(
            json.dumps(
                {
                    "model": {"size": "large", "lora": True},
                    "trainer": {"max_steps": 500},
                }
            )
        )
        result = get_config(cfg)
        assert result.model.size == "large"
        assert result.model.lora is True
        assert result.trainer.max_steps == 500
        # Untouched defaults should remain
        assert result.model.variant == "2d"
        assert result.data.batch_size == 4

    def test_nested_dot_access(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.model.variant == "2d"
        assert result.optimizer.type == "AdamW"
        assert result.trainer.devices == "auto"
        assert result.test.batch_size == 1

    def test_runtime_injection(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        result["num_classes"] = 2
        result["n_modalities"] = 3
        assert result.num_classes == 2
        assert result.n_modalities == 3

    def test_list_values_become_tuples(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert isinstance(tuple(result.data.crop_size), tuple)
        assert tuple(result.data.crop_size) == (378, 378, 32)

    def test_locking_prevents_new_fields(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        result.lock()
        with pytest.raises(AttributeError):
            result.new_field = 42

    def test_locking_allows_modifying_existing(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        result.lock()
        result.trainer.max_steps = 999
        assert result.trainer.max_steps == 999

    def test_all_expected_keys_present(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        # Model
        assert hasattr(result.model, "variant")
        assert hasattr(result.model, "size")
        assert hasattr(result.model, "lora")
        assert hasattr(result.model, "task_tokens")
        assert hasattr(result.model, "task_token_insertion")
        assert hasattr(result.model, "task_token_block")
        # Pretrained
        assert hasattr(result.pretrained, "checkpoint")
        # Data
        assert hasattr(result.data, "dataset_name")
        assert hasattr(result.data, "crop_size")
        assert hasattr(result.data, "volume_patch_size")
        assert hasattr(result.data, "batch_size")
        assert hasattr(result.data, "num_workers")
        assert hasattr(result.data, "resample_spacing")
        # Optimizer
        assert hasattr(result.optimizer, "type")
        assert hasattr(result.optimizer, "params")
        # Trainer
        assert hasattr(result.trainer, "accelerator")
        assert hasattr(result.trainer, "accumulate_grad_batches")
        assert hasattr(result.trainer, "benchmark")
        assert hasattr(result.trainer, "deterministic")
        assert hasattr(result.trainer, "devices")
        assert hasattr(result.trainer, "gradient_clip_val")
        assert hasattr(result.trainer, "limit_val_batches")
        assert hasattr(result.trainer, "max_epochs")
        assert hasattr(result.trainer, "max_steps")
        assert hasattr(result.trainer, "precision")
        assert hasattr(result.trainer, "strategy")
        assert hasattr(result.trainer, "check_val_every_n_epoch")
        assert hasattr(result.trainer, "val_check_interval")
        # Top-level task keys
        assert hasattr(result, "scheduler")
        assert hasattr(result, "loss")
        assert hasattr(result, "metrics")
        # Test
        assert hasattr(result.test, "batch_size")
        assert hasattr(result.test, "amp")
        # Seed
        assert hasattr(result, "seed")


class TestConfigSchemaDefaults:
    """Verify every default matches the intended schema."""

    def test_model_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.model.variant == "2d"
        assert result.model.size == "small"
        assert result.model.lora is False
        assert result.model.task_tokens is False
        assert result.model.task_token_insertion == "beginning"
        assert result.model.task_token_block == 6

    def test_pretrained_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.pretrained.checkpoint is None

    def test_data_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.data.dataset_name == "CLS002_FOMO26_Infarct"
        assert result.data.crop_size == [378, 378, 32]
        assert result.data.volume_patch_size == [14, 14, 2]
        assert result.data.batch_size == 4
        assert result.data.num_workers == 1
        assert result.data.resample_spacing is None

    def test_optimizer_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.optimizer.type == "AdamW"
        assert dict(result.optimizer.params) == {}

    def test_trainer_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.trainer.accelerator == "auto"
        assert result.trainer.accumulate_grad_batches == 1
        assert result.trainer.benchmark is False
        assert result.trainer.deterministic is False
        assert result.trainer.devices == "auto"
        assert result.trainer.gradient_clip_val == 3.0
        assert result.trainer.limit_val_batches == 1.0
        assert result.trainer.max_epochs is None
        assert result.trainer.max_steps == 250
        assert result.trainer.num_nodes == 1
        assert result.trainer.num_sanity_val_steps == 0
        assert result.trainer.precision == "32-true"
        assert result.trainer.strategy == "auto"
        assert result.trainer.sync_batchnorm is False
        assert result.trainer.check_val_every_n_epoch == 1
        assert result.trainer.val_check_interval is None

    def test_task_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.scheduler == []
        assert result.loss is None
        assert result.metrics is None

    def test_test_defaults(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.test.batch_size == 1
        assert result.test.amp is False

    def test_seed_default(self, tmp_path):
        cfg = tmp_path / "test.json"
        cfg.write_text("{}")
        result = get_config(cfg)
        assert result.seed == 42
