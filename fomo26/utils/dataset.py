import json
import os

from fomo26.paths import get_data_path


def get_dataset_dir(dataset_name):
    return os.path.join(get_data_path(), dataset_name)


def load_dataset_json(dataset_name):
    path = os.path.join(get_dataset_dir(dataset_name), "dataset.json")
    with open(path, "r") as f:
        return json.load(f)


def get_dataset_metadata(dataset_name):
    dataset_json = load_dataset_json(dataset_name)
    metadata = dataset_json["metadata"]
    return metadata["n_modalities"], metadata["n_classes"]


def get_split_name(dataset_name):
    dataset_json = load_dataset_json(dataset_name)
    return dataset_json["dataset_config"]["split"]


def load_split_file(dataset_name):
    split_name = get_split_name(dataset_name)
    path = os.path.join(get_dataset_dir(dataset_name), f"{split_name}.json")
    with open(path, "r") as f:
        return json.load(f)


def load_fold(dataset_name, fold):
    folds = load_split_file(dataset_name)
    fold_dict = folds[fold]
    return fold_dict["train"], fold_dict["val"]


def load_test_files(dataset_name):
    split_name = get_split_name(dataset_name)
    suffix = split_name.replace("split_", "")
    path = os.path.join(get_dataset_dir(dataset_name), f"TEST_{suffix}.json")
    with open(path, "r") as f:
        return json.load(f)
