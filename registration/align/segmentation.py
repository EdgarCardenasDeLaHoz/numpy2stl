"""Building-mask segmentation: terrain residual, thresholding, edges.

Part of the align/ subpackage (split from the former align.py).
"""
from __future__ import annotations

import logging
import re
import time
from math import atan2, degrees, sqrt

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from scipy.ndimage import sobel, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    sobel = gaussian_filter = None
    HAS_SCIPY = False

def _base_plate_threshold(values: np.ndarray) -> float:
    """
    Otsu's threshold to separate base plate / terrain from buildings.

    Otsu's method maximises the inter-class variance between the background
    (terrain, roads, base plate — the dominant low-height pixels) and the
    foreground (buildings — the sparse, taller pixels).  It needs no hard-coded
    fraction of the height range, so it works equally well for flat-plate STL
    models and terrain-inclusive city models where terrain variation is large.

    Falls back to mode + 1 bin width for degenerate (flat) distributions.
    """
    if values.size == 0:
        return 0.0
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo

    bins = 256
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    total = float(hist.sum())
    if total == 0:
        return lo

    prob   = hist.astype(np.float64) / total
    bin_c  = (edges[:-1] + edges[1:]) * 0.5
    mu_tot = float(np.dot(prob, bin_c))

    best_var, best_thresh = -1.0, lo
    w0 = m0 = 0.0
    for i in range(bins - 1):
        w0 += prob[i]
        m0 += prob[i] * bin_c[i]
        if w0 <= 0.0 or w0 >= 1.0:
            continue
        w1 = 1.0 - w0
        m1 = (mu_tot - m0) / w1
        var_b = w0 * w1 * (m0 / w0 - m1) ** 2
        if var_b > best_var:
            best_var = var_b
            best_thresh = float(edges[i + 1])

    # Guard: if Otsu returns a threshold below 3% of the range the distribution
    # has no bimodal structure — fall back to histogram mode + 1 bin width.
    if best_thresh < lo + (hi - lo) * 0.03:
        hist64, edges64 = np.histogram(values, bins=64, range=(lo, hi))
        mb = int(np.argmax(hist64))
        best_thresh = float(edges64[mb + 1]) + float(edges64[1] - edges64[0])

    return best_thresh


def terrain_residual(
    heightmap: np.ndarray,
    cell_size_m: float | None = None,
    building_max_m: float = 80.0,
    blur_m: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate height-above-local-terrain via a morphological white top-hat.

    The STL heightmap stores *absolute* elevation (terrain + structures). Buildings
    must be compared as height-above-ground, so we estimate the terrain by
    morphological OPENING (erode→dilate) with a kernel wider than the largest
    building footprint — opening removes everything narrower than the kernel
    (buildings, trees) and leaves the smooth terrain surface. The residual
    (raw − terrain) is the height each cell rises above its local ground.

    Resolution independence
    -----------------------
    The kernel and blur are sized in **metres**, not pixels.  Given
    ``cell_size_m`` (metres per pixel), a fixed physical kernel spans
    ``building_max_m / cell_size_m`` pixels, so the terrain estimate — and
    therefore the recovered building height — is consistent across grid
    resolutions.  When ``cell_size_m`` is None we fall back to the legacy
    pixel-based sizing (``min(shape)//4`` kernel, σ=1.5 px) for callers that
    have no scale information.

    Returns
    -------
    (residual, valid) : float64 residual array (NaN-filled cells set to 0 in
        the residual) and the bool mask of originally-valid (non-NaN) cells.
    """
    arr = heightmap.astype(np.float64)
    valid = ~np.isnan(arr)
    residual = np.zeros_like(arr)
    if not valid.any():
        return residual, valid

    if not HAS_CV2:
        # No morphology available — residual = height above global base plate.
        base = _base_plate_threshold(arr[valid])
        residual[valid] = np.maximum(arr[valid] - base, 0.0)
        return residual, valid

    arr_f32 = arr.astype(np.float32)
    arr_f32[~valid] = float(np.median(arr[valid]))

    if cell_size_m is not None and cell_size_m > 0:
        sigma_px = max(1.0, blur_m / cell_size_m)
        ksize = int(building_max_m / cell_size_m) | 1   # odd, ~constant metres
        ksize = max(31, min(ksize, max(3, min(arr_f32.shape) - 1)))
    else:
        sigma_px = 1.5
        ksize = max(31, min(arr_f32.shape) // 4) | 1     # legacy pixel sizing

    blur_k = max(3, int(sigma_px * 3) | 1)
    arr_f32 = cv2.GaussianBlur(arr_f32, (blur_k, blur_k), sigma_px)
    kernel = np.ones((ksize, ksize), np.uint8)
    terrain = cv2.morphologyEx(arr_f32, cv2.MORPH_OPEN, kernel)
    res = (arr_f32 - terrain).astype(np.float64)
    residual[valid] = res[valid]
    logger.debug(
        "terrain_residual: cell_size_m=%s  ksize=%dpx (%.0fm)  sigma=%.1fpx",
        f"{cell_size_m:.3f}" if cell_size_m else "None",
        ksize, ksize * cell_size_m if cell_size_m else float("nan"), sigma_px,
    )
    return residual, valid


def hill_relief_mask(
    heightmap: np.ndarray,
    cell_size_m: float | None = None,
    hill_sigma_m: float = 120.0,
    hill_thresh_m: float = 3.0,
    base_percentile: float = 10.0,
    adaptive: bool = False,
    relief_percentile: float = 60.0,
    min_hill_thresh_m: float = 2.0,
) -> np.ndarray:
    """
    Flag cells that sit on broad, hill-scale elevated terrain — independent of
    any OSM tags, which don't reliably cover every square metre of a hillside
    (exposed rock, paths, or a real building like a hilltop fortress).

    terrain_residual()'s top-hat kernel is sized to BUILDING scale (~80m) —
    a hillside spanning hundreds of metres is far wider than that kernel can
    ever remove by opening, so it floods straight through as "building".
    This is a second, much-wider-scale signal: heavily Gaussian-blur the
    heightmap (sigma sized in real metres, hill-scale not building-scale) so
    individual buildings average out but a whole hillside stays elevated in
    the blurred result, then flag cells where that SMOOTHED elevation is
    meaningfully above the frame's low/base elevation. Meant to be OR'd into
    an exclude_mask alongside OSM vegetation/water tags (see
    building_mask()'s exclude_mask param) — the two catch different parts of
    a real hillside (OSM often tags forest; this catches the untagged rock/
    grass/path area between), not a replacement for either alone.

    Default is a FIXED threshold (hill_thresh_m=3.0m), validated on Salzburg
    (Kapuzinerberg/Mönchsberg — two steep, compact hills through the historic
    core): matches the two hills' visible extent closely and doesn't touch
    the flat valley-floor building grid; OSM vegetation/water/bridge tags
    alone only covered ~32% of the STL's actual high-elevation pixels there.
    Confirmed safe across all 8 Micropolitan test cities this session,
    including flat ones (Miami, Bilbao) where it correctly stays a no-op.

    `adaptive=True` tries a PERCENTILE-RELATIVE threshold instead (flag cells
    where blurred elevation exceeds base + relief_percentile% of the way to
    the frame's own p95 blurred elevation, floored at min_hill_thresh_m) —
    this generalizes better to cities with broad, gentle relief the fixed
    3.0m under-excludes (measured on Lisbon: fixed 3.0m leaves 53.7% of frame
    as "building" post-exclusion, adaptive leaves a much more plausible
    37.0%). BUT this is NOT safe as a blanket default: measured directly on
    3 real cities, the adaptive threshold computes to Miami=1.27m (flat, must
    stay excluded/no-op), Bilbao=1.65m (ALSO flat/uniform despite a mid-value
    number, must also stay excluded), Lisbon=2.31m (real broad hills, must
    be included) — Bilbao's value sits BETWEEN the two flat cities' and
    Lisbon's real-hill value, so no single min_hill_thresh_m floor can
    correctly separate all three; a floor that lets Lisbon's signal through
    (2.0m) also incorrectly flags 26.5% of flat, non-hilly Bilbao and
    corrupted its rotation (-0.22° -> 13.16°) the same way an unfloored
    version corrupted Miami's (0.02° -> -104.98°). Left here as an opt-in
    experiment for future recalibration (e.g. a spatial-contiguity/shape
    check distinguishing a real hill from scattered building-height noise,
    rather than a magnitude-only threshold) — not wired into the pipeline's
    default call.

    Returns
    -------
    bool ndarray, same shape as `heightmap` — True where terrain is
    "hill-like" (broadly, smoothly elevated at scales much larger than a
    building).
    """
    arr = heightmap.astype(np.float32)
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.zeros(heightmap.shape, dtype=bool)
    fill = float(np.nanmedian(arr[valid])) if valid.any() else 0.0
    arr = np.where(valid, arr, fill).astype(np.float32)

    if cell_size_m is not None and cell_size_m > 0:
        sigma_px = max(1.0, hill_sigma_m / cell_size_m)
    else:
        sigma_px = max(1.0, min(arr.shape) / 8.0)   # legacy pixel sizing fallback
    k = int(sigma_px * 6) | 1
    k = max(3, min(k, max(3, min(arr.shape) - 1)))
    if HAS_CV2:
        blurred = cv2.GaussianBlur(arr, (k, k), sigma_px)
    else:
        blurred = arr  # no-op fallback — degrades to "never hill-like", not a crash

    base = float(np.nanpercentile(arr[valid], base_percentile))
    if not adaptive:
        return valid & ((blurred - base) > hill_thresh_m)
    top = float(np.nanpercentile(blurred[valid], 95))
    span = max(top - base, 1e-6)
    adaptive_thresh = (relief_percentile / 100.0) * span
    thresh_above_base = max(adaptive_thresh, min_hill_thresh_m)
    return valid & ((blurred - base) > thresh_above_base)


# The clip percentile rides in the method name, so a caller that needs a stronger clip than the
# default does not need a branch of its own here.
_MULTIOTSU_CLIP = re.compile(r"multiotsu_p(\d+(?:\.\d+)?)")


def _adaptive_residual_threshold(res_valid: np.ndarray, method: str = "triangle") -> tuple[float, str]:
    """
    Pick a ground/building threshold on the terrain-residual distribution
    (a dominant near-zero ground peak + a building tail) WITHOUT assuming a
    coverage fraction.

    method:
      'triangle'  — skimage threshold_triangle (built for one peak + a tail).
      'multiotsu' — 3-class multi-Otsu; take the lower (ground/low) boundary.
      'multiotsu_pNN' — the same, after clipping the residual at its NNth
                      percentile.  Multi-Otsu minimises within-class variance, so a
                      long tail of towers buys a large variance reduction by taking a
                      class of its own and pushes both boundaries up with it: on Miami
                      the plain rule cuts at 1.17 m where every other city cuts between
                      0.28 and 0.67, and most of the low-rise is called ground.  The
                      clip costs nothing where there is no tail — it moves the healthy
                      cities by at most two percent of what any threshold could reach —
                      and takes Miami from 74% of that ceiling to 99%.  The align tool
                      asks for 'multiotsu_p85', because p95 leaves enough tail to
                      reproduce the Miami failure on plates whose tail is longer still:
                      the Philadelphia miniature cuts at 1.90 where the other plates cut
                      between 0.54 and 0.85, and its building solve lands 5.2 km out.
                      At p85 it cuts at 0.77 and solves 110 m from City Hall, while the
                      plates that already solved do not move -- micropolitan
                      Philadelphia holds its position exactly as its agreement rises
                      from 9/16 to 13/16, and the Boston miniature stays at 10 m.
      'pNN'       — the NNth percentile (e.g. 'p50' = legacy median).
      <float str> — explicit residual value.

    Returns (threshold, label).  Falls back to p50 when the result is degenerate
    (mask would be <2% or >90% of valid cells).
    """
    p50 = float(np.percentile(res_valid, 50))
    n = res_valid.size
    def _cov(t):  # fraction of valid cells kept
        return float((res_valid > t).mean())

    thr, label = p50, "p50"
    try:
        m = str(method).lower()
        if m == "triangle":
            from skimage.filters import threshold_triangle
            thr, label = float(threshold_triangle(res_valid)), "triangle"
        elif m == "multiotsu" or _MULTIOTSU_CLIP.fullmatch(m):
            from skimage.filters import threshold_multiotsu
            v, clip = res_valid, _MULTIOTSU_CLIP.fullmatch(m)
            if clip:
                v = np.clip(v, None, np.percentile(v, float(clip.group(1))))
            thr = float(threshold_multiotsu(v, classes=3)[0]); label = m
        elif m.startswith("p") and m[1:].replace(".", "", 1).isdigit():
            pct = float(m[1:]); thr, label = float(np.percentile(res_valid, pct)), m
        else:
            thr, label = float(m), "fixed"
    except Exception:
        thr, label = p50, "p50(fallback)"

    cov = _cov(thr)
    if not (0.02 <= cov <= 0.90):   # degenerate — revert to the robust median split
        thr, label = p50, f"p50(guard:{label}cov={cov:.2f})"
    return thr, label


def building_mask(
    heightmap: np.ndarray,
    source: str = "stl",
    threshold: float | None = None,
    min_blob_px: int = -1,
    target_coverage: float | None = None,
    cell_size_m: float | None = None,
    segment_features: bool = False,
    threshold_method: str = "triangle",
    fill_holes_px: int | None = 0,
    split_watershed: bool = False,
    allow_forced_split: bool = True,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert a heightmap into a binary building-presence mask.

    Building *footprints* register far more reliably than continuous heights:
    both datasets agree on "is there a building here?" even when they disagree
    on exact height.  This is the robust coarse signal for registration when
    height data is noisy or incomplete.

    Parameters
    ----------
    heightmap : (rows, cols) float64
        STL projection (NaN or ~0 = ground) or OSM raster (NaN = no building).
    source : {'stl', 'osm'}
        'osm' — a building is any non-NaN cell (a footprint was rasterized).
        'stl' — a building is any cell whose height rises meaningfully above
                the ground/base plane.
    threshold : float, optional
        Height above which an STL cell counts as a building.  Default: auto.
        When None, the STL path uses a morphological top-hat to estimate
        height-above-local-terrain, then applies Otsu's threshold to the
        residual.  This correctly handles terrain-inclusive models where
        absolute height is dominated by topography rather than buildings.
    min_blob_px : int
        Remove connected components smaller than this (speckle).
    exclude_mask : (rows, cols) bool, optional
        STL-source only. Cells to treat as non-building regardless of height
        — e.g. vegetation/hillside/water from OSM semantic tags. Excluded
        BEFORE the top-hat/threshold computation (not just masked out of the
        final result), because a large excluded region (a hillside wider
        than the top-hat kernel) otherwise contaminates the terrain estimate
        and the adaptive threshold's residual distribution for the WHOLE
        frame, not just its own footprint — measured on Salzburg: a
        forested hillside occupying ~26% of the frame, wider than the
        building-scale top-hat kernel could ever remove, pushed STL mask
        coverage to 66.6% of the frame (vs OSM ground truth 29.1%) and
        drove the height-correlation rotation disambiguator to score the
        CORRECT rotation candidate negative (see F-SKY docs / global_search.py
        L0 orientation logic — this is what exclude_mask is for).
    allow_forced_split : bool
        When True (default) and split_watershed=False, still force a
        watershed split if a single connected component dominates the frame
        (see the dense-core note below) — this is for report/vectorization
        callers.  The registration SEARCH (global_search.py's building_edges
        calls) sets this False: changing edge structure mid-search can flip
        the L0 rotation disambiguator's candidate scores (measured: enabling
        the forced split unconditionally flipped Bilbao's already-fixed
        180°-flip bug back to 179.8°), so the search must keep exactly the
        mask it was tuned against and only the report/footprint paths adopt
        the better segmentation.

    Returns
    -------
    bool ndarray, same shape — True where a building is present.
    """
    residual = None  # height-above-terrain; populated on the STL top-hat path
    if source == "osm":
        mask = ~np.isnan(heightmap)
    else:
        arr = heightmap.copy().astype(np.float64)
        if exclude_mask is not None and exclude_mask.shape == arr.shape:
            # NaN-out excluded cells (not just AND into `valid` below) so
            # terrain_residual() — which derives its OWN valid mask from
            # np.isnan(arr) and doesn't otherwise know about exclude_mask —
            # excludes them from the top-hat's terrain fill AND the returned
            # residual/valid used for the adaptive threshold statistics. A
            # large excluded region (e.g. a hillside) contaminates both if
            # it's only filtered out of the FINAL mask instead.
            arr[exclude_mask] = np.nan
        valid = ~np.isnan(arr)
        if threshold is None:
            if not valid.any():
                return np.zeros(heightmap.shape, dtype=bool)
            if HAS_CV2:
                # Height above local terrain via morphological top-hat.  Kernel
                # sized in metres when cell_size_m is known (resolution-independent),
                # else legacy pixel sizing.  See terrain_residual() for rationale.
                #
                # Default threshold_method="triangle" (skimage threshold_triangle):
                # only cells rising meaningfully above local ground survive.
                # Coverage matching is NOT used by default — forcing OSM coverage
                # pulls in terrain features to make up the numbers. Measured against
                # real STL-pack data (Miami downtown, OSM coverage 24.9% ground
                # truth): triangle → 19.2% (closest), p50 → 50.7%, p24 → 77.4%,
                # multiotsu → 11.6% — p50 was the function's default until this was
                # fixed, silently doubling building-mask coverage almost everywhere
                # (the two call sites that already knew to override it to "triangle"
                # were the exception, not the rule).
                residual, valid = terrain_residual(arr, cell_size_m=cell_size_m)
                res_valid = residual[valid]
                if target_coverage is not None:
                    frac_of_valid = float(np.clip(
                        target_coverage * arr.size / valid.sum(), 0.0, 1.0))
                    pct = max(0.0, (1.0 - frac_of_valid) * 100.0)
                    threshold = float(np.percentile(res_valid, pct))
                    thr_label = f"target_cov_p{pct:.0f}"
                else:
                    # Adaptive ground/building split on the residual — no fixed
                    # coverage assumption.  See _adaptive_residual_threshold.
                    threshold, thr_label = _adaptive_residual_threshold(
                        res_valid, method=threshold_method)
                mask = valid & (residual > threshold)
                logger.debug(
                    "STL building_mask: thr=%.4f (%s)  mask=%.1f%%  (p50=%.4f p90=%.4f)",
                    threshold, thr_label, 100.0 * mask.sum() / mask.size,
                    float(np.percentile(res_valid, 50)),
                    float(np.percentile(res_valid, 90)),
                )
            else:
                threshold = _base_plate_threshold(arr[valid])
                mask = valid & (arr > threshold)
        else:
            mask = valid & (arr > threshold)

    # Adaptive min blob: at 512-res 1px ≈ 4.3m; require ≥ ~200m² building footprint
    # (≈ 11px). Scale with image area so it stays meaningful at other resolutions.
    if min_blob_px < 0:
        min_blob_px = max(10, heightmap.size // (512 * 512 // 25))

    # Clean speckle (STL only — OSM is clean vector data, no morphological cleanup needed):
    #   1. Close (dilate→erode) with 3×3 to merge adjacent building pixels into
    #      solid blocks before size-filtering (avoids breaking real buildings apart).
    #   2. Open (erode→dilate) with 3×3 to remove isolated noise dots.
    #   3. Remove connected components smaller than min_blob_px.
    #   4. (STL only) Filter textured components (trees) by height variance and gradient.
    if HAS_CV2 and min_blob_px > 0:
        m8 = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        if source != "osm":
            m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        keep = np.zeros_like(m8, dtype=bool)

        # Texture + geometry segmentation: classify each large component as
        # building (smooth, planar roof) vs non-building (rough, non-planar — e.g.
        # tree canopy).  Only runs for STL when explicitly requested and a residual
        # is available; small components and the no-residual path keep everything.
        do_seg = (segment_features and source != "osm" and residual is not None)
        # A lone tree / tree-clump is a SMALL, isolated, rough, non-planar blob.
        # Large components are buildings or merged downtown blocks (which merge
        # into one giant component in dense cores) and are always kept — texture
        # can't distinguish a city block, and OSM vegetation masking handles parks.
        # tree_max_px ≈ a ~30 m clump; sized in metres when cell_size_m is known.
        if cell_size_m and cell_size_m > 0:
            tree_max_px = int((30.0 / cell_size_m) ** 2)
        else:
            tree_max_px = max(min_blob_px * 4, 400)

        candidate = []          # (label_id, roughness, planarity, area) — small blobs only
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_blob_px:
                keep[labels == i] = True       # tiny speckle kept (legacy behaviour)
                continue
            if not do_seg or area > tree_max_px:
                keep[labels == i] = True       # large structure / no segmentation → keep
                continue
            rough, planar = _component_features(residual, labels == i)
            candidate.append((i, rough, planar, area))

        if do_seg and len(candidate) >= 4:
            roughs = np.array([c[1] for c in candidate])
            planars = np.array([c[2] for c in candidate])
            # A tree is rough AND non-planar; reject only small blobs in the upper
            # tail of BOTH features so the building population is preserved.
            r_thr = float(np.percentile(roughs, 70))
            p_thr = float(np.percentile(planars, 70))
            rejected = 0
            for (i, rough, planar, area) in candidate:
                if rough > r_thr and planar > p_thr:
                    rejected += 1               # rough AND non-planar small blob → tree
                else:
                    keep[labels == i] = True
            logger.info(
                "Feature segmentation: examined %d small blobs (<%dpx), "
                "rejected %d as trees (rough>%.3f & non-planar>%.3f).",
                len(candidate), tree_max_px, rejected, r_thr, p_thr)
        else:
            for (i, *_rest) in candidate:        # too few to threshold reliably
                keep[labels == i] = True

        # Denoise: fill small INTERIOR holes (gaps inside footprints from threshold
        # noise / mesh dropouts).  A hole is a background component that does not
        # touch the image border; fill those below fill_holes_px.  Large genuine
        # gaps (courtyards, plazas, streets reaching the border) are preserved.
        if fill_holes_px is None:
            fill_holes_px = min_blob_px
        if fill_holes_px and fill_holes_px > 0:
            inv = (~keep).astype(np.uint8)
            nb, blab, bstats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
            h_, w_ = keep.shape
            filled = 0
            for j in range(1, nb):
                x0 = bstats[j, cv2.CC_STAT_LEFT]; y0 = bstats[j, cv2.CC_STAT_TOP]
                bw = bstats[j, cv2.CC_STAT_WIDTH]; bh = bstats[j, cv2.CC_STAT_HEIGHT]
                area = int(bstats[j, cv2.CC_STAT_AREA])
                touches_border = (x0 == 0 or y0 == 0 or x0 + bw >= w_ or y0 + bh >= h_)
                if not touches_border and area < fill_holes_px:
                    keep[blab == j] = True
                    filled += 1
            if filled:
                logger.debug("building_mask: filled %d small holes (<%dpx)", filled, fill_holes_px)

        mask = keep

    # Separate merged building blobs (downtown blocks + the street between them)
    # so each footprint vectorizes individually.  STL only; OSM is already split
    # vector data.  Off by default to keep registration masks stable.
    force_split = False
    if HAS_CV2 and source != "osm" and not split_watershed and allow_forced_split and mask.any():
        # Dense cores (e.g. Haussmannian Paris) can threshold to a single
        # connected blob spanning a large fraction of the frame even though
        # coverage itself looks reasonable (~35%) -- the triangle threshold
        # picks a residual cut low enough that adjacent buildings' footprints
        # touch through the raster, well before any morphological close runs.
        # Measured on Paris: one component covered 35.6% of frame BEFORE any
        # close/open, producing 19 vectorized "polygons" (two giant degenerate
        # triangles + speckle) instead of ~150+ real building footprints, even
        # though the underlying mask agreed well with OSM (IoU 0.53 as a raw
        # raster). A single blob this large is never a real building -- force
        # the watershed split regardless of the caller's split_watershed flag,
        # since leaving city blocks fused never helps registration or the
        # footprint/vectorization report.
        _n, _, _stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        if _n > 1:
            _dominant_frac = float(_stats[1:, cv2.CC_STAT_AREA].max()) / mask.size
            force_split = _dominant_frac > 0.15
    if (split_watershed or force_split) and source != "osm":
        mask = split_touching_buildings(mask, cell_size_m=cell_size_m)
        if force_split:
            logger.info(
                "building_mask: forced watershed split -- a single connected "
                "component covered >15%% of the frame (dense-core under-"
                "segmentation, e.g. Haussmannian block fusion).")

    return mask


def split_touching_buildings(
    mask: np.ndarray,
    cell_size_m: float | None = None,
    min_separation_m: float = 12.0,
    min_floor_px: int = 3,
) -> np.ndarray:
    """Carve 1-px gaps between merged building footprints (watershed split).

    A single global threshold fuses adjacent downtown buildings (and the street
    between them) into one giant blob, so they vectorize as a single ragged
    polygon.  This separates them: a distance transform of the mask peaks at each
    building centre; watershed from those peak markers partitions the blob along
    its narrow waists (the streets), and the watershed ridge lines are set to
    background — so `vectorize_buildings` then traces each building separately.

    The footprint area only shrinks by the 1-px ridge lines, so overlap with the
    original is essentially preserved.  Falls back to the input mask if skimage
    is unavailable or the mask is trivial.

    Parameters
    ----------
    min_separation_m : minimum spacing between building-centre markers (metres
        when ``cell_size_m`` is known, else taken as pixels).  Prevents one
        building from splitting into several.

        Tuned against real Paris data (dense Haussmannian blocks — many
        individual, party-wall-attached buildings per block, only a couple
        metres apart with no height gap between them): the original 25m/6px
        values were sized to separate whole BLOCKS from each other (across a
        real street gap) and left individual buildings within a block still
        fused — vectorizing as one block-scale polygon (median 2540px^2 at
        2x-detect resolution) instead of OSM's individual-building scale
        (median ~680px^2 at the same resolution). Tightening to 12m/3px
        roughly doubled the polygon count (55->103, much closer to OSM's own
        157) with no measurable IoU cost (0.286->0.284) — the true limit
        below ~12m is the heightmap's own physical resolution, not this
        parameter (going lower produced no further splits).
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return m
    try:
        from scipy import ndimage as ndi
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed, find_boundaries
    except Exception:
        return m

    if cell_size_m and cell_size_m > 0:
        min_dist = max(min_floor_px, int(round(min_separation_m / cell_size_m)))
    else:
        min_dist = max(min_floor_px, int(round(min_separation_m)))

    distance = ndi.distance_transform_edt(m)
    # Seed one marker per genuine building CORE using the H-maxima transform: it
    # keeps only maxima whose "depth" above the surrounding saddle is ≥ h, merging
    # the pixel-scale ripples of a single large rooftop into one marker (avoids the
    # 129→1025 shatter) while still separating two buildings joined at a thin neck
    # (their cores are deep maxima separated by a low-distance waist).  h is set to
    # half the min building separation, in distance-transform (pixel) units.
    h = max(2.0, 0.5 * min_dist)
    seeds = h_maxima(distance, h)
    markers, n_markers = ndi.label(seeds)
    if n_markers < 2:
        return m   # single core → nothing to split
    labels = watershed(-distance, markers, mask=m)
    ridges = find_boundaries(labels, mode="outer") & m
    out = m & ~ridges
    _, n_before = ndi.label(m)
    _, n_after = ndi.label(out)
    coords = np.argwhere(seeds)   # for logging only
    logger.info("split_touching_buildings: %d markers, components %d -> %d (min_sep=%dpx)",
                len(coords), int(n_before), int(n_after), min_dist)
    return out


def _component_features(residual: np.ndarray, comp: np.ndarray) -> tuple[float, float]:
    """
    Two discriminative features for a connected component, computed on the
    height-above-terrain residual (no topography confound):

    - roughness : std-dev of residual within the component.  Tree canopies vary
      cell-to-cell (high); building roofs are flat or smoothly sloped (low).
    - planarity : RMS distance to a least-squares plane fit through the
      component's (x, y, residual) points, normalised by the residual scale.
      Roofs lie near a plane (low); canopies do not (high).
    """
    ys, xs = np.where(comp)
    z = residual[ys, xs].astype(np.float64)
    if z.size < 4:
        return 0.0, 0.0
    rough = float(np.std(z))
    # Plane fit z ≈ a*x + b*y + c
    A = np.column_stack([xs.astype(np.float64), ys.astype(np.float64), np.ones(z.size)])
    try:
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        resid = z - A @ coef
        scale = float(np.median(np.abs(z - np.median(z)))) + 1e-6
        planar = float(np.sqrt(np.mean(resid ** 2)) / scale)
    except np.linalg.LinAlgError:
        planar = 0.0
    return rough, planar


def building_edges(heightmap: np.ndarray, source: str = "stl", **kwargs) -> np.ndarray:
    """
    Binary edge (footprint outline) mask — the registration signal of choice.

    Filled building masks are dense (often >50% of the frame), so two of them
    overlap heavily regardless of alignment: high IoU there is largely a
    blob-overlap artefact (random-chance IoU ~0.2–0.45). The *outlines* of the
    footprints are sparse and distinctive — building edges and street boundaries
    coincide between datasets only when truly aligned. Empirically the edge mask
    registers 7x above its random baseline vs ~2.5x for filled masks.

    Returns
    -------
    bool ndarray — True on the 1-px boundary of each building footprint.
    """
    mask = building_mask(heightmap, source=source, **kwargs).astype(np.uint8)
    if not HAS_CV2:
        # numpy gradient fallback
        gx = np.abs(np.diff(mask, axis=1, prepend=0))
        gy = np.abs(np.diff(mask, axis=0, prepend=0))
        return ((gx + gy) > 0)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask - eroded).astype(bool)


def _regularize_polygon(poly: np.ndarray, max_snap_px: float = 3.0) -> np.ndarray:
    """Snap a footprint polygon to its dominant orthogonal orientation.

    Real building footprints are mostly rectilinear; the raster contour staircase
    leaves edges a few degrees off and a pixel or two ragged.  We find the
    dominant edge direction (length-weighted), rotate the polygon so that
    direction is axis-aligned, snap each vertex that moves less than
    ``max_snap_px`` onto its neighbour's x/y (collapsing near-axis edges to exactly
    axis-aligned), then rotate back.  The displacement cap guarantees the
    footprint can't move more than a couple of pixels, so overlap is preserved.
    """
    if len(poly) < 4:
        return poly
    p = poly.astype(np.float64)
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    if lengths.sum() <= 0:
        return poly
    # Dominant orientation mod 90°.  Orientations are 90°-periodic, so a plain
    # arithmetic mean of folded angles is wrong (it averages 5° and 85° to 45°).
    # Use the length-weighted CIRCULAR mean of 4×angle (period 90° → 360°).
    raw = np.arctan2(edges[:, 1], edges[:, 0])
    z = np.sum(lengths * np.exp(1j * 4.0 * raw))
    theta = (np.angle(z) / 4.0) % (np.pi / 2.0)
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    q = p @ R.T
    # Snap small edge offsets onto the axis (rectilinearize).
    for i in range(len(q)):
        j = (i + 1) % len(q)
        dx, dy = q[j, 0] - q[i, 0], q[j, 1] - q[i, 1]
        if abs(dx) <= max_snap_px:      # near-vertical edge → equalise x
            q[j, 0] = q[i, 0]
        elif abs(dy) <= max_snap_px:    # near-horizontal edge → equalise y
            q[j, 1] = q[i, 1]
    out = q @ np.linalg.inv(R).T
    # Guard: cap total displacement so overlap is preserved.
    disp = np.hypot(*(out - p).T)
    if np.max(disp) > 2 * max_snap_px + 1.0:
        return poly
    return np.rint(out).astype(np.int32)


def vectorize_buildings(mask, simplify_frac: float = 0.02, min_area_px: int = 20,
                        regularize: bool = False, max_snap_px: float = 3.0):
    """
    Convert a binary building-footprint mask into simplified VECTOR polygons.

    The building outlines (the edges of the segmented regions) are traced and the
    resulting connectivity is simplified into a small set of vertices — the
    polygon form OSM building footprints natively use, so STL footprints can be
    compared / exported in the same representation.

    Method: trace each footprint's outer contour (cv2.findContours) and simplify
    it with Douglas–Peucker (cv2.approxPolyDP), tolerance = ``simplify_frac`` of
    the contour perimeter.  This collapses the ragged staircase boundary into a
    few straight edges (a rectangle becomes ~4 points).

    Parameters
    ----------
    mask         : 2-D bool / 0-1 array — building presence.
    simplify_frac: Douglas–Peucker epsilon as a fraction of each contour's
                   perimeter (larger = fewer vertices).
    min_area_px  : drop footprints smaller than this (speckle).

    Returns
    -------
    list[np.ndarray] : each (K, 2) int32 array of (x, y) polygon vertices.
    """
    if not HAS_CV2:
        return []
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = simplify_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) >= 3:
            poly = approx.reshape(-1, 2).astype(np.int32)
            if regularize:
                poly = _regularize_polygon(poly, max_snap_px=max_snap_px)
            polys.append(poly)
    return polys
