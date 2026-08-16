import torch

from torchvision import transforms

from med_adapt.registry import register_aug
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import (
    Torch_AdditiveNoise,
    Torch_MultiplicativeNoise,
)


class Torch_Resize(BaseTransform):
    """Resize the entire 3D volume to a fixed target shape on the CPU."""

    def __init__(
        self,
        data_key="image",
        label_key="label",
        target_size: tuple | list = None,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.target_size = tuple(target_size) if target_size else None

    def get_params(self):
        return

    def __call__(self, data_dict):
        image = data_dict[self.data_key]
        if self.target_size is not None:
            # image shape: (C, H, W, D) -> interpolate expects (N, C, D, H, W)
            resized = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            # resized: (C, target_D, target_H, target_W) -> back to (C, target_H, target_W, target_D)
            data_dict[self.data_key] = resized

        if data_dict.get(self.label_key) is not None and self.target_size is not None:
            label = data_dict[self.label_key]
            # label shape: (H, W, D) -> (N, C, D, H, W) for interpolate
            # Cast to float for interpolate, then back to original dtype.
            label_float = label.to(dtype=torch.float32)
            resized_label = torch.nn.functional.interpolate(
                label_float.unsqueeze(0),
                size=self.target_size,
                mode="nearest",
            ).squeeze(0)
            # resized_label: (target_D, target_H, target_W) -> (target_H, target_W, target_D)
            data_dict[self.label_key] = resized_label.to(label.dtype)

        return data_dict


@register_aug("default_norm")
def default_norm():
    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
        ]
    )


@register_aug("default_disable_aug")
def default_disable_aug(ndim=3):
    return transforms.Compose([])


@register_aug("default_enable_aug")
def default_enable_aug(ndim=3):
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.15),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.15),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.5, p_per_axis=0.25),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    return tforms
