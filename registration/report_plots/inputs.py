"""Stage-1 input figures: STL + OSM heightmaps, side-by-side aligned.

Part of the report_plots/ subpackage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Public render functions (one per asset PNG)
# ---------------------------------------------------------------------------

from ._common import _imshow_heightmap, _render_single_heightmap

def render_stl_heightmap_png(out_path: str | Path, stl_hm: np.ndarray) -> Path:
    """STL 2D projection -- viridis, NaN shown in light gray."""
    return _render_single_heightmap(
        out_path, stl_hm,
        title="STL Heightmap (model units)",
        cmap="viridis",
        nan_color="#cccccc",
        cbar_label="Height (model units)",
    )


def render_osm_heightmap_png(out_path: str | Path, osm_hm: np.ndarray) -> Path:
    """OSM building heights -- viridis, NaN shown in white."""
    return _render_single_heightmap(
        out_path, osm_hm,
        title="OSM Building Heights",
        cmap="viridis",
        nan_color="#ffffff",
        cbar_label="Height (m)",
    )


def render_aligned_png(
    out_path: str | Path,
    stl_aligned: np.ndarray,
    osm_hm: np.ndarray,
) -> Path:
    """Side-by-side: STL aligned | OSM (same pixel space)."""
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _imshow_heightmap(axes[0], stl_aligned, "STL Aligned (model units)", "viridis", "#cccccc")
    _imshow_heightmap(axes[1], osm_hm, "OSM Building Heights (m)", "viridis", "#ffffff")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
