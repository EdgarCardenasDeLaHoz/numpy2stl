"""Shared heightmap-rendering helpers.

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

def _render_single_heightmap(
    out_path, arr, title, cmap, nan_color, cbar_label
) -> Path:
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    fig, ax = plt.subplots(figsize=(6, 5))
    _imshow_heightmap(ax, arr, title, cmap, nan_color, cbar_label=cbar_label, fig=fig)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_three_panel(out_path, left, right, diff, *, left_title, right_title,
                       diff_title, height_label="height (model units)",
                       diff_label="Δ height (m)", suptitle=None, dpi=130) -> Path:
    """Shared original | simplified | difference 3-panel figure.

    `left`/`right` are heightmaps (viridis, NaN=grey); `diff` is a signed difference
    (diverging RdBu_r, symmetric range = p99 of |diff|).  Used by the decimation and
    prism report figures.
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")
    finite = np.isfinite(diff)
    vmax = max(float(np.nanpercentile(np.abs(diff[finite]), 99)) if finite.any() else 1.0, 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    hc = plt.get_cmap("viridis").copy(); hc.set_bad("#dddddd")
    for a, arr, ttl in ((ax[0], left, left_title), (ax[1], right, right_title)):
        im = a.imshow(arr, cmap=hc, origin="lower")
        a.set_title(ttl, fontsize=10); a.set_xlabel("col"); a.set_ylabel("row")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04, label=height_label)
    dc = plt.get_cmap("RdBu_r").copy(); dc.set_bad("#dddddd")
    im = ax[2].imshow(diff, cmap=dc, origin="lower", vmin=-vmax, vmax=vmax)
    ax[2].set_title(diff_title, fontsize=10); ax[2].set_xlabel("col"); ax[2].set_ylabel("row")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04, label=diff_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _imshow_heightmap(ax, arr, title, cmap, nan_color, cbar_label=None, fig=None):
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color=nan_color)
    im = ax.imshow(arr, cmap=cm, origin="lower")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if cbar_label is not None and fig is not None:
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    return im
