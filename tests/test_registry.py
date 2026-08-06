"""Tests for the med_adapt registry."""

import pytest

from med_adapt.registry import STORE, register_model, register_dataset, register_aug


class TestRegistry:
    def test_get_model(self):
        cls = STORE.get("models", "vitv2_a_2d_tiny")
        assert callable(cls)

    def test_get_dataset(self):
        cls = STORE.get("datasets", "CLS002_FOMO26_Infarct")
        assert cls.__name__ == "Task1InfarctClassification"

    def test_get_aug(self):
        fn = STORE.get("augs", "default_norm")
        assert callable(fn)

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            STORE.get("models", "nonexistent_model")

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            STORE.get("datasets", "nonexistent_dataset")

    def test_all_expected_models_registered(self):
        model_keys = set(STORE._instances["models"].keys())
        expected_extended = {
            "vitv2_a_2d_tiny",
            "vitv2_a_2d_small",
            "vitv2_a_2d_base",
            "vitv2_a_2d_large",
            "vitv2_a_3d_tiny",
            "vitv2_a_3d_small",
            "vitv2_a_3d_base",
            "vitv2_a_3d_large",
        }
        assert expected_extended.issubset(model_keys)

    def test_all_expected_datasets_registered(self):
        ds_keys = STORE._instances["datasets"].keys()
        expected = {
            "CLS002_FOMO26_Infarct",
            "SEG002_Meningioma",
            "REG002_BrainAge",
            "SEG002_TrigeminalNeuralgia",
            "CLS002_Polymicrogyria",
        }
        assert expected.issubset(set(ds_keys))

    def test_all_expected_augs_registered(self):
        aug_keys = STORE._instances["augs"].keys()
        expected = {"default_norm", "default_disable_aug", "default_enable_aug"}
        assert expected.issubset(set(aug_keys))

    def test_decorator_registration(self):
        @register_model("test_model_xyz")
        class FakeModel:
            pass

        assert STORE.get("models", "test_model_xyz") is FakeModel

    def test_decorator_dataset_registration(self):
        @register_dataset("test_ds_xyz")
        class FakeDataset:
            TASK_TYPE = "classification"
            NUM_MODALITIES = 1
            NUM_CLASSES = 2
            MODALITIES = ("t1",)

        assert STORE.get("datasets", "test_ds_xyz") is FakeDataset

    def test_decorator_aug_registration(self):
        @register_aug("test_aug_xyz")
        def fake_aug():
            return None

        assert STORE.get("augs", "test_aug_xyz") is fake_aug
