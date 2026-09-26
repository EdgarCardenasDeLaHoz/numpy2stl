"""align: 2D registration (translation + rotation + scale).

Split from the former monolithic align.py into topical submodules.
This faade re-exports the full public + test surface so
`from numpy2stl.registration.align import X` keeps working.
"""
from __future__ import annotations

from .ecc import discover_projection, refine_transform
from .fourier_mellin import FourierMellinResult, fourier_mellin_register
from .global_search import register_global
from .lines import gradient_angle_histogram, rotation_from_angle_histograms
from .mask_source import (
    active_mask_producer,
    produce_edges,
    produce_mask,
    use_config,
    use_mask_producer,
)
from .metrics import _dice, _mask_sdf, _tolerant_iou, score_alignment
from .polygon_icp import refine_registration_polygons
from .polygon_register import register_polygons
from .register import register
from .scale import _fourier_profile_scale, estimate_scale
from .segmentation import (
    HAS_CV2,  # noqa: F401
    _base_plate_threshold,
    _component_features,
    _regularize_polygon,
    building_edges,
    building_mask,
    hill_relief_mask,
    split_touching_buildings,
    terrain_residual,
    vectorize_buildings,
)
from .transform import (
    _compose_resize_scale,
    _decompose_matrix,
    _preprocess_for_registration,
    apply_transform,
)

__all__ = [
    "_preprocess_for_registration",
    "apply_transform",
    "_decompose_matrix",
    "_compose_resize_scale",
    "_base_plate_threshold",
    "terrain_residual",
    "building_mask",
    "_component_features",
    "building_edges",
    "vectorize_buildings",
    "split_touching_buildings",
    "_regularize_polygon",
    "hill_relief_mask",
    "produce_mask",
    "produce_edges",
    "use_mask_producer",
    "use_config",
    "active_mask_producer",
    "gradient_angle_histogram",
    "rotation_from_angle_histograms",
    "_tolerant_iou",
    "_dice",
    "_mask_sdf",
    "score_alignment",
    "estimate_scale",
    "_fourier_profile_scale",
    "fourier_mellin_register",
    "FourierMellinResult",
    "register_global",
    "refine_transform",
    "discover_projection",
    "refine_registration_polygons",
    "register_polygons",
    "register",
    "HAS_CV2",
]
