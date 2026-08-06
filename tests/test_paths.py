"""Tests for med_adapt.utils.paths."""

import os
import pytest

from med_adapt.utils.paths import (
    DATA_ROOT,
    MODELS_ROOT,
    RESULTS_ROOT,
    CONFIGS_ROOT,
    LABELS_ROOT,
    get_data_path,
    get_models_path,
    get_results_path,
    get_config_path,
    get_source_labels_path,
)


class TestPaths:
    """Test path resolution with and without env vars."""

    def test_ensure_dir(self, tmp_path):
        """_ensure_dir should create the directory."""
        from med_adapt.utils.paths import _ensure_dir

        p = tmp_path / "sub" / "dir"
        result = _ensure_dir(str(p))
        assert result == p
        assert p.exists()

    def test_get_env_fallback(self, monkeypatch):
        """_get_env should fall back to legacy vars."""
        from med_adapt.utils.paths import _get_env

        monkeypatch.setenv("FOMO26_DATA", "/tmp/legacy_data")
        monkeypatch.delenv("MED_ADAPT_DATA", raising=False)
        assert _get_env("MED_ADAPT_DATA", "FOMO26_DATA") == "/tmp/legacy_data"

    def test_get_env_missing(self, monkeypatch):
        """_get_env returns None for missing optional vars."""
        from med_adapt.utils.paths import _get_env

        monkeypatch.delenv("MED_ADAPT_NONEXISTENT", raising=False)
        monkeypatch.delenv("FOMO26_NONEXISTENT", raising=False)
        result = _get_env("MED_ADAPT_NONEXISTENT", "FOMO26_NONEXISTENT", optional=True)
        assert result is None

    def test_get_env_optional(self, monkeypatch):
        """_get_env should return None for optional missing vars."""
        from med_adapt.utils.paths import _get_env

        monkeypatch.delenv("MED_ADAPT_FOO", raising=False)
        result = _get_env("MED_ADAPT_FOO", optional=True)
        assert result is None

    def test_lazy_path_division(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MED_ADAPT_DATA", str(tmp_path))
        # Force re-resolve by creating a fresh _LazyPath
        from med_adapt.utils.paths import _LazyPath

        p = _LazyPath("MED_ADAPT_DATA", "FOMO26_DATA")
        child = p / "subdir"
        assert str(child) == str(tmp_path / "subdir")

    def test_lazy_path_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MED_ADAPT_DATA", str(tmp_path))
        from med_adapt.utils.paths import _LazyPath

        p = _LazyPath("MED_ADAPT_DATA", "FOMO26_DATA")
        assert p.exists() is True
