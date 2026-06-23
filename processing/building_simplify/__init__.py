"""Footprint-preserving building-mesh simplification.

Split (B5) into decimate / prism / _io submodules; this facade preserves the
original ``from numpy2stl.processing.building_simplify import X`` imports.
"""
from .decimate import (
    SimplifyStats, _symmetric_hausdorff, decimate_to_tolerance,
    decimation_sweep, flatten_roof_clutter, simplify_building_mesh,
)
from .prism import PrismStats, _fit_plane, prism_decompose
from ._io import (
    _save_prism_models, _save_mesh, _save_prism_lod, _rasterize_mesh,
)

__all__ = [
    "SimplifyStats", "decimate_to_tolerance", "decimation_sweep",
    "flatten_roof_clutter", "simplify_building_mesh",
    "PrismStats", "prism_decompose",
]
