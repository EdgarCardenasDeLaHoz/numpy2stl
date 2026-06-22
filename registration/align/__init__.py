"""align: 2D registration (translation + rotation + scale).

Split from the former monolithic align.py into topical submodules.
This faade re-exports the full public + test surface so
`from numpy2stl.registration.align import X` keeps working.
"""
from __future__ import annotations

from .transform import _preprocess_for_registration, apply_transform, _decompose_matrix, _compose_resize_scale
from .segmentation import _base_plate_threshold, terrain_residual, building_mask, _component_features, building_edges, vectorize_buildings, split_touching_buildings, _regularize_polygon
from .lines import gradient_angle_histogram, rotation_from_angle_histograms
from .metrics import _tolerant_iou, _dice, _mask_sdf, score_alignment
from .scale import estimate_scale, _fourier_profile_scale
from .fourier_mellin import fourier_mellin_register, FourierMellinResult
from .global_search import register_global
from .ecc import refine_transform, discover_projection
from .polygon_icp import refine_registration_polygons
from .polygon_register import register_polygons
from .register import register
from .segmentation import HAS_CV2  # noqa: F401

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
