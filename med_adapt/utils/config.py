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
    return ConfigDict(load_json_config(config_path), convert_dict=True)
