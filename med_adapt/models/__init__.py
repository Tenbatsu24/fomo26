"""Model registry and factories.

Base models live in :mod:`med_adapt.models.base`.
Extended (adapter-wrapped) models live in :mod:`med_adapt.models.extended`.
"""

from med_adapt.models.base import (
    ViTv2,
    vitv2_tiny,
    vitv2_small,
    vitv2_base,
    vitv2_large,
)
from med_adapt.models.extended import (
    ViTv2Adaption,
    ViTv2Adaption3D,
    vitv2_a_2d_tiny,
    vitv2_a_2d_small,
    vitv2_a_2d_base,
    vitv2_a_2d_large,
    vitv2_a_3d_tiny,
    vitv2_a_3d_small,
    vitv2_a_3d_base,
    vitv2_a_3d_large,
)

__all__ = [
    # base
    "ViTv2",
    "vitv2_tiny",
    "vitv2_small",
    "vitv2_base",
    "vitv2_large",
    # extended
    "ViTv2Adaption",
    "ViTv2Adaption3D",
    "vitv2_a_2d_tiny",
    "vitv2_a_2d_small",
    "vitv2_a_2d_base",
    "vitv2_a_2d_large",
    "vitv2_a_3d_tiny",
    "vitv2_a_3d_small",
    "vitv2_a_3d_base",
    "vitv2_a_3d_large",
]
