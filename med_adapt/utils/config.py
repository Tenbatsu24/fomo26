"""Configuration loading and management for med_adapt.

Provides JSON config loading via ml_collections.ConfigDict with a centralised
schema and defaults. Consumers access values via dot-notation (e.g.
``cfg.model.variant``) — no more ``.get('key', default)`` scattered around.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from ml_collections import ConfigDict

from med_adapt.utils.paths import CONFIGS_ROOT

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

# Remove default handler and add a tqdm-aware one.
logger.remove()
logger.add(
    lambda msg: print(msg, end=""),  # stdout sink
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)


def get_logger(name: str | None = None) -> logger:
    """Return a named child logger consistent with the package-wide logger."""
    if name is None:
        return logger
    return logger.opt(depth=1).bind(name=name)


# ---------------------------------------------------------------------------
# Default config (single source of truth)
# ---------------------------------------------------------------------------


def _make_default_config() -> ConfigDict:
    """Return a ConfigDict with all default values defined in one place."""
    cfg = ConfigDict()

    # -- model ---------------------------------------------------------------
    cfg.model = ConfigDict()
    cfg.model.variant = "2d"  # "2d" or "3d"
    cfg.model.size = "small"  # "tiny", "small", "base", "large"
    cfg.model.lora = False
    cfg.model.task_tokens = False
    cfg.model.task_token_insertion = "middle"  # "beginning" or "middle"
    cfg.model.task_token_block = 6

    # -- pretrained ----------------------------------------------------------
    cfg.pretrained = ConfigDict()
    cfg.pretrained.checkpoint = None

    # -- data ----------------------------------------------------------------
    cfg.data = ConfigDict()
    cfg.data.dataset_name = "CLS002_FOMO26_Infarct"
    cfg.data.crop_size = [378, 378, 32]
    cfg.data.volume_patch_size = [14, 14, 2]
    cfg.data.batch_size = 4
    cfg.data.num_workers = 1
    cfg.data.resample_spacing = None

    # -- optimizer -----------------------------------------------------------
    cfg.optimizer = ConfigDict()
    cfg.optimizer.type = "AdamW"
    cfg.optimizer.params = {}

    # -- trainer (mirrors pl.Trainer kwargs) ---------------------------------
    cfg.trainer = ConfigDict()
    cfg.trainer.accelerator = "auto"
    cfg.trainer.accumulate_grad_batches = 1
    cfg.trainer.benchmark = False
    cfg.trainer.deterministic = False
    cfg.trainer.devices = "auto"
    cfg.trainer.gradient_clip_algorithm = None
    cfg.trainer.gradient_clip_val = 3.0
    cfg.trainer.limit_val_batches = 1.0
    cfg.trainer.max_epochs = None
    cfg.trainer.max_steps = 250
    cfg.trainer.num_nodes = 1
    cfg.trainer.num_sanity_val_steps = 0
    cfg.trainer.precision = "32-true"
    cfg.trainer.strategy = "auto"
    cfg.trainer.sync_batchnorm = False
    cfg.trainer.check_val_every_n_epoch = 1
    cfg.trainer.val_check_interval = None

    # -- task-specific (loss, metrics, scheduler) ----------------------------
    cfg.scheduler = []
    cfg.loss = None
    cfg.metrics = None

    # -- test ----------------------------------------------------------------
    cfg.test = ConfigDict()
    cfg.test.batch_size = 1
    cfg.test.amp = False

    # -- misc ----------------------------------------------------------------
    cfg.seed = 42

    return cfg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """Load a JSON config file and return it as a plain dict."""
    path = Path(config_path)
    if not path.is_absolute():
        path = CONFIGS_ROOT / path
    with open(path, "r") as f:
        return json.load(f) or {}


def get_config(config_path: str | Path) -> ConfigDict:
    """Load a JSON config, merge with defaults, and return a ConfigDict.

    The returned ConfigDict has all defaults populated. Values from the JSON
    file override the defaults. Runtime-injected keys (e.g. ``num_classes``)
    can be set afterwards via item assignment.
    """
    cfg = _make_default_config()
    overrides = load_json_config(config_path)
    cfg.update(overrides)
    return cfg
