"""Frozen dataclasses for the registration pipeline.

Mirrors skyline/region_types.py: immutable, cacheable, no heavy imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegistrationResult:
    """Output of align.register()."""

    transform: np.ndarray       # (2,3) affine or (3,3) homography warp matrix
    confidence: float           # ECC correlation coefficient, 0–1
    scale: float                # spatial scale factor (source px → target px)
    angle_deg: float            # rotation found (degrees)
    n_iterations: int           # ECC iterations used
    converged: bool             # True if ECC hit eps threshold before max_iter
    projection: str = "affine"  # projection model the data supported (cross-validated)


@dataclass(frozen=True)
class ComparisonResult:
    """Output of compare.compare()."""

    difference: np.ndarray      # (rows, cols) float64 — (STL_m − OSM) per pixel, NaN outside overlap
    building_diff_map: np.ndarray  # (rows, cols) — per-BUILDING (STL_m − OSM), one value per
                                   # footprint, fills + fit-outliers excluded; NaN elsewhere
    overlap_mask: np.ndarray    # bool — True where both arrays have valid data
    missing_in_osm: np.ndarray  # bool — STL has data, OSM is NaN (new/unlabelled buildings)
    missing_in_stl: np.ndarray  # bool — OSM has data, STL is NaN (outside STL extent)

    rmse: float                 # root mean squared error (metres)
    mae: float                  # mean absolute error (metres)
    bias: float                 # mean(STL − OSM); positive = STL buildings taller than OSM
    correlation: float          # Pearson r over overlap region
    rank_correlation: float     # Spearman ρ (rank-based, robust to outliers)
    mape: float                 # mean absolute percentage error (%)
    coverage_pct: float         # % of OSM building pixels with matching STL data
    footprint_iou: float        # honest UNION IoU of STL vs OSM footprints |A∩B|/|A∪B| (registration quality)
    dice_score: float           # UNION Dice 2*|A∩B|/(|A|+|B|) — same masks as footprint_iou
    n_overlap: int              # pixel count of valid overlap
    height_scale_used: float    # the Z slope applied to the STL (metres per model unit)
    height_offset_used: float   # the Z intercept added after scaling (metres): stl_m = stl*scale + offset
    height_ratio_mean: float    # mean(STL / OSM) — 1.0 = perfect scale
    height_ratio_std: float     # std(STL / OSM) — variability in per-pixel scaling


@dataclass(frozen=True)
class CityRegistrationReport:
    """Full pipeline result returned by register_city_stl().

    Mirrors the skyline CityRegistrationReport pattern: all pipeline
    intermediate results plus timing.  Pass to write_registration_report()
    to produce an HTML + PNG report folder.
    """

    region_name: str
    stl_file: str
    stl_heightmap: np.ndarray       # (rows, cols) — STL projection
    osm_heightmap: np.ndarray       # (rows, cols) — OSM building heights
    stl_aligned: np.ndarray         # (rows, cols) — STL warped into OSM pixel space
    registration: RegistrationResult
    comparison: ComparisonResult
    step_timings: list              # list[tuple[str, float]] — [(step_name, wall_sec), ...]
    scale_sweep:    tuple = ()          # [(scale, edge_iou), ...]
    rot_sweep:      tuple = ()          # [(rotation_deg, edge_iou), ...]
    _hist_src:      object = None       # line-angle histogram for STL edges
    _hist_tgt:      object = None       # line-angle histogram for OSM edges
    _hist_xcorr:    object = None       # circular cross-correlation curve
    _hist_rot_deg:  float  = 0.0        # histogram rotation estimate (degrees)
    _xcorr_map:     object = None       # 2-D FFT translation xcorr at best rotation
    _best_dx:       float  = 0.0        # found translation x (pixels)
    _best_dy:       float  = 0.0        # found translation y (pixels)
    _rot_l1_xcorr:  object = None  # {rot_deg: xcorr} at L0+L1 resolution (0.25° step)
    _rot_l2_xcorr:  object = None  # {rot_deg: xcorr} at L2 resolution (0.05° step)
    _dice_fine:     object = None  # {rot_deg: dice} at Dice-refinement resolution (0.25°, ±3°)
    known_scale: float | None = None  # physical anchor scale (1/osm_margin), or None
    landmark_check: dict | None = None  # city-hall / landmark sanity check result
    area_scale: float | None = None     # scale independently derived from footprint-area ratio
    fourier_scale: float | None = None  # scale independently derived from Fourier profile match
    _stl_polygons: object = None        # hi-res adaptive STL footprint polygons (working-res coords)
    _stl_heightmap_original: object = None  # pre-simplification STL heightmap (when simplify_mesh)
    _simplify_stats: object = None          # SimplifyStats-as-dict (faces, ratio, hausdorff, budget)
    _decimation_sweep: object = None        # [{ratio, faces, hausdorff_units, hausdorff_m}, ...]
    _prism_stats: object = None             # PrismStats-as-dict (simplify_mode="prism")
