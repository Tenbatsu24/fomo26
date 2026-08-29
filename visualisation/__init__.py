"""Position-embedding, patch-embedding, and PCA visualisation for ViT checkpoints."""

from .utils import colorbar, fig_kw, save_figure
from .patch_embed_2d_analysis import plot_patch_embed
from .pos_embed_2d_analysis import plot_pos_embed
from .pca_2d import plot_volume_pca
from .patch_embed_3d_analysis import plot_patch_embed_3d
from .pos_embed_3d_analysis import analyse_3d_pos_embed
from .pca_3d import plot_volume_pca_3d

__all__ = [
    "colorbar",
    "fig_kw",
    "save_figure",
    "plot_patch_embed",
    "plot_pos_embed",
    "plot_volume_pca",
    "plot_patch_embed_3d",
    "analyse_3d_pos_embed",
    "plot_volume_pca_3d",
]
