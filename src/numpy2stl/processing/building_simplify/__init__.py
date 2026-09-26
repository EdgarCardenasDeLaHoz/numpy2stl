"""Footprint-preserving building-mesh simplification.

Split (B5) into decimate / prism / _io submodules; this facade preserves the
original ``from numpy2stl.processing.building_simplify import X`` imports.
"""
# Private names are re-exported for backward compatibility (tests import them).
from .decimate import (  # noqa: F401
    SimplifyStats, _symmetric_hausdorff, decimate_to_tolerance,
    decimation_sweep, flatten_roof_clutter, simplify_building_mesh,
)
from .prism import PrismStats, _fit_plane, prism_decompose  # noqa: F401
from ._io import (  # noqa: F401
    _save_prism_models, _save_mesh, _save_prism_lod, _rasterize_mesh,
)

__all__ = [
    "SimplifyStats", "decimate_to_tolerance", "decimation_sweep",
    "flatten_roof_clutter", "simplify_building_mesh",
    "PrismStats", "prism_decompose",
]
