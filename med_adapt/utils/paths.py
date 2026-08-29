"""Path resolution for med_adapt.

Loads paths from environment variables with fallback to legacy ``FOMO26_*``
names.  All paths are resolved lazily on first access so that importing
this module does not require the environment variables to be set.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _get_env(
    var: str, fallback: str | None = None, optional: bool = False
) -> str | None:
    """Return the value of *var*, falling back to *fallback* if unset."""
    load_dotenv()
    value = os.environ.get(var) or (os.environ.get(fallback) if fallback else None)
    if value is None:
        if not optional:
            raise ValueError(
                f"Missing required environment variable {var} (or legacy {fallback})."
            )
        return None
    return value


def _ensure_dir(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Lazy path resolution
# ---------------------------------------------------------------------------


class _LazyPath:
    """Lazily resolved path that caches after first access."""

    __slots__ = ("_var", "_fallback", "_value")

    def __init__(self, var: str, fallback: str | None = None):
        self._var = var
        self._fallback = fallback
        self._value: Path | None = None

    def _resolve(self) -> Path:
        if self._value is None:
            raw = _get_env(self._var, self._fallback)
            self._value = _ensure_dir(raw)
        return self._value  # type: ignore

    def __fspath__(self) -> str:
        return str(self._resolve())

    def __str__(self) -> str:
        return str(self._resolve())

    def __repr__(self) -> str:
        return f"<Path {self._resolve()}>"

    # Delegate common Path methods
    def __truediv__(self, other):
        return self._resolve() / other

    def joinpath(self, *paths):
        return self._resolve().joinpath(*paths)

    def exists(self):
        return self._resolve().exists()

    def is_dir(self):
        return self._resolve().is_dir()

    def glob(self, pattern):
        return self._resolve().glob(pattern)

    def iterdir(self):
        return self._resolve().iterdir()

    def mkdir(self, *args, **kwargs):
        return self._resolve().mkdir(*args, **kwargs)

    def read_text(self, *args, **kwargs):
        return self._resolve().read_text(*args, **kwargs)

    def write_text(self, *args, **kwargs):
        return self._resolve().write_text(*args, **kwargs)

    def with_suffix(self, suffix):
        return self._resolve().with_suffix(suffix)

    @property
    def name(self):
        return self._resolve().name

    @property
    def parent(self):
        return self._resolve().parent


# ---------------------------------------------------------------------------
# Public lazy path constants
# ---------------------------------------------------------------------------

DATA_ROOT = _LazyPath("MED_ADAPT_DATA", "FOMO26_DATA")
NNSSL_ROOT = _LazyPath("nnssl_preprocessed", "FOMO26_DATA")
MODELS_ROOT = _LazyPath("MED_ADAPT_MODELS", "FOMO26_MODELS")
RESULTS_ROOT = _LazyPath("MED_ADAPT_RESULTS", "FOMO26_RESULTS")
CONFIGS_ROOT = _LazyPath("MED_ADAPT_CONFIGS", "FOMO26_CONFIGS")
LABELS_ROOT = _LazyPath("MED_ADAPT_RAW_LABELS", "FOMO26_RAW_LABELS")

_FINETUNE_RAW = _get_env(
    "MED_ADAPT_FINETUNE_CONFIGS", "FOMO26_FINETUNE_CONFIGS", optional=True
)
if _FINETUNE_RAW:
    FINETUNE_CONFIGS: list[Path] = [
        _ensure_dir(p.strip()) for p in _FINETUNE_RAW.split(":")
    ]
else:
    FINETUNE_CONFIGS: list[Path] = []


def get_data_path() -> Path:
    """Return the data root directory."""
    return DATA_ROOT._resolve()  # type: ignore


def get_nnssl_preprocessed_path() -> Path:
    """Return the data root directory."""
    return NNSSL_ROOT._resolve()  # type: ignore


def get_models_path() -> Path:
    """Return the pretrained models root directory."""
    return MODELS_ROOT._resolve()  # type: ignore


def get_results_path() -> Path:
    """Return the results output directory."""
    return RESULTS_ROOT._resolve()  # type: ignore


def get_config_path() -> Path:
    """Return the configs root directory."""
    return CONFIGS_ROOT._resolve()  # type: ignore


def get_source_labels_path() -> Path:
    """Return the raw labels source directory."""
    return LABELS_ROOT._resolve()  # type: ignore


def get_additional_finetune_config_path() -> list[Path]:
    """Return optional fine-tune config directories."""
    return FINETUNE_CONFIGS
