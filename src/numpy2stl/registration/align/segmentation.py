"""Deprecated alias of ``numpy2stl.raster.segment`` / ``numpy2stl.raster.vectorize``.

Kept for one release so ``from numpy2stl.registration.align.segmentation import X``
keeps working; import from ``numpy2stl.raster`` instead.
"""
from __future__ import annotations

from ...raster.segment import (  # noqa: F401
    HAS_CV2,
    _adaptive_residual_threshold,
    _base_plate_threshold,
    _component_features,
    building_edges,
    building_mask,
    hill_relief_mask,
    split_touching_buildings,
    terrain_residual,
)
from ...raster.vectorize import _regularize_polygon, vectorize_buildings  # noqa: F401
