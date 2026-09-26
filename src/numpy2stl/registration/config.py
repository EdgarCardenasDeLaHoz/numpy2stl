"""Centralized, named tuning constants for the registration pipeline.

Every value here was previously a magic number scattered inline across the
pipeline.  They are collected so the behaviour can be tuned in one place and so
the assumptions are documented (see docs/ARCHITECTURE.md for the audit table).

`RegistrationConfig` is a frozen dataclass; pass an instance to
`register_city_stl(..., config=...)` to override defaults.  Functions deep in
the pipeline take the individual values as parameters defaulting to the
module-level constants below, so nothing has to thread the whole object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# --- Geometry / fetch ------------------------------------------------------
# OSM frame is fetched at this multiple of the STL footprint.  Larger frames
# shrink the STL within the grid and weaken registration; 1.5x balances
# surrounding context against keeping the STL large enough to register.
DEFAULT_OSM_MARGIN = 1.5

# Metres per degree of latitude (spherical-earth approximation).  Longitude is
# scaled by cos(latitude).  Used to convert the OSM degree grid to metres.
M_PER_DEG_LAT = 111_320.0

# --- Segmentation (terrain residual + building mask) -----------------------
# Terrain top-hat opening kernel, in METRES (resolution-independent).  Wider
# than the largest building footprint so the morphological opening removes
# structures and leaves the ground surface.
DEFAULT_TERRAIN_KERNEL_M = 80.0
# Pre-smoothing Gaussian sigma, in METRES, applied before the top-hat.
DEFAULT_TERRAIN_BLUR_M = 6.0
# Percentile of the height-above-terrain residual above which a cell counts as
# a building.  p50 = upper half (real structures); the near-zero mass below is
# ground / streets / base plate.
DEFAULT_BUILDING_THRESHOLD_PCT = 50.0

# --- Scale search ----------------------------------------------------------
# Half-width of the post-hoc scale sweep around the prior (for the determinism
# plot and the peak-xcorr scale pick).
DEFAULT_SCALE_SWEEP_HALFWIDTH = 0.45
DEFAULT_SCALE_SWEEP_STEP = 0.025

# --- Rotation --------------------------------------------------------------
# Peak-Dice rotation is searched only within +/- this window (degrees) of the
# translation-invariant xcorr estimate, to avoid periodic-grid alias peaks.
DEFAULT_DICE_ROT_WINDOW_DEG = 3.0

# --- Hough line detection (weak-line rejection) ----------------------------
# Minimum segment length as a fraction of the smaller image dimension.
DEFAULT_HOUGH_MIN_LENGTH_FRAC = 0.03

# --- Segmentation -----------------------------------------------------------
# Ground/building threshold method for the STL terrain residual:
# 'triangle' | 'multiotsu' | 'pNN' | float.  Used for the comparison/footprint
# mask (NOT registration, which keeps the stable p50 split).
DEFAULT_BUILDING_THRESHOLD = "triangle"
# STL building detection runs at this multiple of the working resolution, then
# the mask is downsampled — separates merged buildings, sharpens edges.
DEFAULT_DETECT_RESOLUTION_FACTOR = 2

# Canonical resolution for the registration SEARCH.  The search (gradient
# rotation, scale sweep, FFT translation) always runs at this resolution so the
# found transform is IDENTICAL regardless of the output `resolution`, and a 1k
# output doesn't pay a 4x search cost.  The transform's linear part (scale +
# rotation) is a pixel ratio (resolution-independent); only translation is scaled
# to the output grid.  Comparison / segmentation still use the full output res.
REGISTER_RES = 512

# --- Mesh simplification ---------------------------------------------------
# Surface-deviation budget (metres) for footprint-preserving mesh simplification.
# The decimator removes as much geometry as possible while the symmetric Hausdorff
# distance to the original stays under this; ~one storey keeps building shape while
# stripping roof clutter.  Larger = more aggressive flattening.
DEFAULT_SIMPLIFY_TOL_M = 3.5

# --- Center search (find_best_city_center) ----------------------------------
# Minimum (peak - sweep median) margin to trust a scale-Dice / rotation-IoU peak
# as a genuine, sharp signal rather than sweep noise.  Reuses global_search.py's
# own internal gates verbatim (_DICE_PEAK_MARGIN, _ROT_PEAK_MARGIN in
# register_global(), both 0.10) — a candidate center is only "locked" when its
# coarse register_global() pass shows the same kind of sharp peak that
# register_global() itself already requires before trusting a Dice/IoU peak
# over its fallback.  Named separately here (not just re-imported) because they
# gate a DIFFERENT decision (is this CENTER right?) than the ones inside
# global_search.py (is this SCALE/ROTATION right, given the center already?).
CENTER_SEARCH_DICE_MARGIN = 0.10
CENTER_SEARCH_ROT_MARGIN = 0.10


@dataclass(frozen=True)
class RegistrationConfig:
    """Tunable parameters for register_city_stl().  Defaults match the
    module-level constants (i.e. current production behaviour)."""

    osm_margin: float = DEFAULT_OSM_MARGIN
    terrain_kernel_m: float = DEFAULT_TERRAIN_KERNEL_M
    terrain_blur_m: float = DEFAULT_TERRAIN_BLUR_M
    building_threshold_pct: float = DEFAULT_BUILDING_THRESHOLD_PCT
    scale_sweep_halfwidth: float = DEFAULT_SCALE_SWEEP_HALFWIDTH
    scale_sweep_step: float = DEFAULT_SCALE_SWEEP_STEP
    dice_rot_window_deg: float = DEFAULT_DICE_ROT_WINDOW_DEG
    hough_min_length_frac: float = DEFAULT_HOUGH_MIN_LENGTH_FRAC

    # Constants that previously had no home on the config even though they are
    # exactly as tunable as the ones above.
    threshold_method: str = DEFAULT_BUILDING_THRESHOLD
    detect_resolution_factor: int = DEFAULT_DETECT_RESOLUTION_FACTOR
    register_res: int = REGISTER_RES

    # --- The mask-producer seam ---------------------------------------------
    # Optional replacement for the classic `building_mask` segmentation.  None
    # (the default) means classic behaviour, byte for byte.  A producer has
    # `building_mask`'s signature — (heightmap, source=..., **kwargs) -> bool
    # ndarray of the same shape — and must tolerate keyword arguments it does
    # not use, since call sites pass the classic tuning knobs.
    #
    # Only STL heightmaps route through it: OSM masks are `~np.isnan(...)` of
    # rasterized footprints, i.e. ground truth rather than an estimate.
    #
    # Install it with `align.mask_source.use_config(cfg)`; the pipeline is not
    # yet threaded with a config object, so nothing reads this field
    # automatically.  See align/mask_source.py for the seam itself and
    # docs/registration-learning-plan.md for why segmentation is the axis
    # worth making swappable.
    mask_producer: Callable | None = None


DEFAULT_CONFIG = RegistrationConfig()
