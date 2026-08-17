"""Shared plotting utilities for the visualisation package."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fig_kw(**override: object) -> dict:
    """Return default figure kwargs, allowing overrides."""
    base: dict = {"figsize": (8, 6), "dpi": 150}
    base.update(override)  # type: ignore[typeddict-item]
    return base


def colorbar(fig, mappable, ax=None, label: str = "") -> None:
    """Add a colour bar to *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    mappable : matplotlib.cm.ScalarMappable
    ax : matplotlib.axes.Axes | None
        If given, scope the colour bar to this axis.  Otherwise a single
        colour bar is shared across the whole figure.
    label : str
    """
    if ax is not None:
        fig.colorbar(mappable, ax=ax, label=label, shrink=0.8)
    else:
        fig.colorbar(mappable, label=label, shrink=0.8)


def save_figure(fig, path) -> None:
    """Save *fig* to *path* and close it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
