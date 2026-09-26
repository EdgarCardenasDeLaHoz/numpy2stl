"""Footprint-preserving building-mesh simplification.

Split (B5) into decimate / prism / _io submodules; the public API below keeps
``from numpy2stl.processing.building_simplify import X`` working. Private
helpers are imported from their defining submodule.
"""
from .decimate import (
    SimplifyStats, decimate_to_tolerance, decimation_sweep,
    flatten_roof_clutter, simplify_building_mesh,
)
from .prism import PrismStats, prism_decompose

__all__ = [
    "SimplifyStats", "decimate_to_tolerance", "decimation_sweep",
    "flatten_roof_clutter", "simplify_building_mesh",
    "PrismStats", "prism_decompose",
]
