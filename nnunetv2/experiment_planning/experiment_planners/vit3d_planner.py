import numpy as np
from typing import Union, List, Tuple

from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import (
    ExperimentPlanner,
)

from med_adapt.models.extended.volume import ViT3DAdaption


def round_up_to_multiple(values, multiple: int = 8):
    """Round each element of a tuple up to the nearest multiple (ceiling)."""
    return tuple(int(np.ceil(v / multiple) * multiple) for v in values)


class ViT3DPlanner(ExperimentPlanner):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        gpu_memory_target_in_gb: float = 110,
        preprocessor_name: str = "DefaultPreprocessor",
        plans_name: str = "nnUNetViT3DAdaption",
        overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
        suppress_transpose: bool = False,
    ):
        super().__init__(
            dataset_name_or_id,
            gpu_memory_target_in_gb,
            preprocessor_name,
            plans_name,
            overwrite_target_spacing,
            suppress_transpose,
        )
        self.UNet_class = ViT3DPlanner

        self.UNet_reference_val_3d = 680000000
        self.UNet_reference_val_2d = 135000000

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        if configuration_name == "2d" or configuration_name == "3d_fullres":
            # we do not deviate from ExperimentPlanner so we can reuse its data
            return "nnUNetPlans" + "_" + configuration_name
        else:
            return self.plans_identifier + "_" + configuration_name

    def get_plans_for_configuration(
        self,
        spacing: Union[np.ndarray, Tuple[float, ...], List[float]],
        median_shape: Union[np.ndarray, Tuple[int, ...]],
        data_identifier: str,
        approximate_n_voxels_dataset: float,
        _cache: dict,
    ) -> dict:

        assert all([i > 0 for i in spacing]), f"Spacing must be > 0! Spacing: {spacing}"

        initial_patch_size = median_shape[: len(spacing)]

        patch_size = round_up_to_multiple(initial_patch_size)

        architecture_kwargs = {
            "network_class_name": self.UNet_class.__module__
            + "."
            + self.UNet_class.__name__,
            "arch_kwargs": {},
            "_kw_requires_import": [],
        }

        batch_size = 2

        # we need to cap the batch size to cover at most 5% of the entire dataset. Overfitting precaution. We cannot
        # go smaller than self.UNet_min_batch_size though
        bs_corresponding_to_5_percent = round(
            approximate_n_voxels_dataset
            * self.max_dataset_covered
            / np.prod(patch_size, dtype=np.float64)
        )
        batch_size = max(
            min(batch_size, bs_corresponding_to_5_percent), self.UNet_min_batch_size
        )

        (
            resampling_data,
            resampling_data_kwargs,
            resampling_seg,
            resampling_seg_kwargs,
        ) = self.determine_resampling()
        resampling_softmax, resampling_softmax_kwargs = (
            self.determine_segmentation_softmax_export_fn()
        )

        normalization_schemes, mask_is_used_for_norm = (
            self.determine_normalization_scheme_and_whether_mask_is_used_for_norm()
        )

        plan = {
            "data_identifier": data_identifier,
            "preprocessor_name": self.preprocessor_name,
            "batch_size": batch_size,
            "patch_size": patch_size,
            "median_image_size_in_voxels": median_shape,
            "spacing": spacing,
            "normalization_schemes": normalization_schemes,
            "use_mask_for_norm": mask_is_used_for_norm,
            "resampling_fn_data": resampling_data.__name__,
            "resampling_fn_seg": resampling_seg.__name__,
            "resampling_fn_data_kwargs": resampling_data_kwargs,
            "resampling_fn_seg_kwargs": resampling_seg_kwargs,
            "resampling_fn_probabilities": resampling_softmax.__name__,
            "resampling_fn_probabilities_kwargs": resampling_softmax_kwargs,
            "architecture": architecture_kwargs,
        }
        return plan


if __name__ == "__main__":
    net = ViT3DAdaption(n_modalities=1, classes=4, task="segmentation")
