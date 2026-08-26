import warnings

from typing import Union

import torch

try:
    from dynamic_network_architectures.architectures.primus import (
        PrimusS,
        PrimusM,
        PrimusL,
        PrimusB,
    )
except ImportError:
    warnings.warn(
        "Unable to import Primus architectures. Make sure you have the correct dynamic_network_architectures package installed."
    )
    PrimusS = None
    PrimusM = None
    PrimusL = None
    PrimusB = None
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from med_adapt.models.extended.volume import ViT3DAdaption


def get_network_from_name(
    arch_class_name,
    input_channels,
    output_channels,
    input_patchsize: tuple[int, int, int] | None = None,
    allow_init=True,
    deep_supervision: Union[bool, None] = None,
):
    if arch_class_name == "vit3d":
        network = ViT3DAdaption(
            n_modalities=input_channels,
            classes=output_channels,
            task="segmentation",
        )
    elif arch_class_name == "PrimusS":
        network = PrimusS(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusM":
        network = PrimusM(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusL":
        network = PrimusL(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "PrimusB":
        network = PrimusB(
            input_channels=input_channels,
            output_channels=output_channels,
            patch_embed_size=(8, 8, 8),
            input_shape=input_patchsize,
        )
    elif arch_class_name == "ResEncL":
        n_stages = 6
        network = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
            strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
            num_classes=output_channels,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
            conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )
    else:
        raise ValueError("Unknown architecture class name: {}".format(arch_class_name))

    if hasattr(network, "initialize") and allow_init:
        network.apply(network.initialize)

    return network


if __name__ == "__main__":
    model = get_network_from_name(
        "vit3d", 1, 4, allow_init=False, deep_supervision=False
    )
    model = model.to("cuda").to(torch.float16)
    data = torch.rand((1, 1, 96, 96, 96), device="cuda", dtype=torch.float16)
    target = torch.rand(size=(1, 1, 96, 96, 96), device="cuda", dtype=torch.float16)
    outputs = model(data)  # this should be a list of torch.Tensor
    print([output.shape for output in outputs])
