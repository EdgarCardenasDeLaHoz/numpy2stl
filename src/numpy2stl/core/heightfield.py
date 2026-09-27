"""Heightfield rasters -> closed solids.

``tin_solid`` is the adaptive one (error-bounded TIN top, see
``processing.decimate.heightfield_tin``); ``core.generate.array_to_mesh`` remains
the one-quad-per-pixel grid solid.
"""

from __future__ import annotations

import numpy as np

from .extrude import close_surface, orient_ccw

__all__ = ["tin_solid"]


def tin_solid(z: np.ndarray, max_error: float, mm_per_px: float = 1.0, floor: float = 0.0,
              seed_step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Watertight block: adaptive top within ``max_error`` of every pixel, side
    walls, flat bottom at ``floor``.

    Row 0 of ``z`` is north: pixel (i, j) -> x = j*s, y = (H-1-i)*s. The top
    surface's vertices are the first ``len(vertices) // 2`` rows of the result.
    """
    from ..processing.decimate import heightfield_tin

    h, w = z.shape
    idx, tris = heightfield_tin(z, max_error, seed_step=seed_step)
    remap = np.full(h * w, -1)
    remap[idx] = np.arange(len(idx))
    ii, jj = np.divmod(idx, w)
    top = np.column_stack([jj * mm_per_px, (h - 1 - ii) * mm_per_px, z.ravel()[idx]]).astype(np.float64)
    f = orient_ccw(top, remap[tris])
    return close_surface(top, f, lambda xy: np.full(len(xy), floor))
