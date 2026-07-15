from torchvision import transforms
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import (
    Torch_AdditiveNoise,
    Torch_MultiplicativeNoise,
)


def default_norm():
    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
        ]
    )


def default_aug(ndim=3, deep_supervision=False):
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
