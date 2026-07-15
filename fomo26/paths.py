import os

from dotenv import load_dotenv


def var_is_set(var):
    return var in os.environ.keys()


def get_environment_variable(var, optional=False):
    load_dotenv()
    if not var_is_set(var):
        if optional:
            return None
        raise ValueError(f"Missing required environment variable {var}.")

    path = os.environ[var]
    os.makedirs(path, exist_ok=True)
    return path


def get_environment_variables(var, optional=False):
    load_dotenv()
    if not var_is_set(var):
        if optional:
            return []
        raise ValueError(f"Missing required environment variable {var}.")

    paths = os.environ[var].split(":")
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return paths


def get_data_path():
    return get_environment_variable("FOMO26_DATA")


def get_models_path():
    return get_environment_variable("FOMO26_MODELS")


def get_results_path():
    return get_environment_variable("FOMO26_RESULTS")


def get_config_path():
    return get_environment_variable("FOMO26_CONFIGS")


def get_source_labels_path():
    return get_environment_variable("FOMO26_RAW_LABELS")


def get_additional_finetune_config_path():
    return get_environment_variables("FOMO26_FINETUNE_CONFIGS")
