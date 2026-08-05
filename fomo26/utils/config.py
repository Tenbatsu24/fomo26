import logging
from typing import TypedDict

import yaml

LOGGER = logging.getLogger(__name__)


class ConfigSchema(TypedDict, total=False):
    """Documented schema for the flattened training config.

    Keys are grouped in YAML but flattened into a single namespace by
    ``load_yaml_config``. This TypedDict makes the valid keys and their
    expected types explicit for IDE autocomplete and static checking.
    """

    # -- model --
    model_variant: str  # "2d" or "3d"
    model_size: str  # "tiny", "small", "base", "large"
    lora: bool
    task_tokens: bool

    # -- pretrained --
    checkpoint: str

    # -- data --
    dataset_name: str
    crop_size: list[int]
    batch_size: int
    num_workers: int
    resample_spacing: tuple[float, float, float] | None

    # -- optim --
    opt: str  # "AdamW" or "SGD"
    lr: float
    weight_decay: float
    betas: tuple[float, float]

    # -- trainer --
    max_steps: int
    precision: str
    devices: int | str
    strategy: str
    gradient_clip_val: float | None

    # -- injected at runtime --
    num_classes: int | None
    n_modalities: int


def flatten_config(nested_config: dict) -> dict:
    flat = {}
    for group in nested_config.values():
        flat.update(group)
    return flat


def load_yaml_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        nested_config = yaml.safe_load(f)
    return flatten_config(nested_config)
