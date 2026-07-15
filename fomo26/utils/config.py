import yaml


def flatten_config(nested_config):
    flat = {}
    for group in nested_config.values():
        flat.update(group)
    return flat


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        nested_config = yaml.safe_load(f)
    return flatten_config(nested_config)
