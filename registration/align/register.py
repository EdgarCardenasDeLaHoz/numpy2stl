"""Top-level register() — a thin wrapper over the global edge-IoU search.

Part of the align/ subpackage (split from the former align.py).  The former
multi-stage ECC / log-polar / grid-period pipeline was removed: production always
used the deterministic `register_global`, which is more robust (it doesn't bias
scale upward or chase dense-mask overlap).
"""
from __future__ import annotations

import logging

import numpy as np

from .global_search import register_global

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def register(
    source: np.ndarray,
    target: np.ndarray,
    max_scale_ratio: float = 5.0,
    known_scale: float | None = None,
    scale_search: float = 0.35,
    cell_size_m: float | None = None,
    forced_rotation: float | None = None,
    source_mask: np.ndarray | None = None,
    free_scale: bool = False,
) -> dict:
    """Find the 2-D similarity transform aligning `source` (STL heightmap) onto
    `target` (OSM raster) via the deterministic global edge-IoU search.

    Returns
    -------
    dict: ``transform`` (2,3), ``confidence``, ``scale``, ``angle_deg``,
    ``converged``, ``n_iterations``, plus the diagnostic sweeps for the report.
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")
    return register_global(
        source, target, scale_prior=known_scale, scale_search=scale_search,
        cell_size_m=cell_size_m, forced_rotation=forced_rotation,
        source_mask=source_mask, free_scale=free_scale,
    )
