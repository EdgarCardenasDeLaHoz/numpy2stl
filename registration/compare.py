"""Post-registration height comparison between STL and OSM rasters."""

from __future__ import annotations

import logging

import numpy as np

from .types import ComparisonResult

logger = logging.getLogger(__name__)


def compare(
    stl_aligned: np.ndarray,
    osm: np.ndarray,
    height_scale: float | None = 1.0,
    height_offset: float | None = None,
    height_agg: str = "p95",
    robust_fit: bool = False,
) -> ComparisonResult:
    """
    Compute pixel-wise height difference between an aligned STL heightmap
    and an OSM building-height raster.

    Both arrays must be the same shape (rows, cols).  NaN = no data.

    Conversion to metres uses an affine fit ``stl_m = stl * scale + offset``:
    the offset (intercept) absorbs a systematic height bias — base-plate
    thickness, terrain-estimate offset — that a through-origin fit would
    otherwise force into the slope, distorting the scale.

    Parameters
    ----------
    stl_aligned  : (rows, cols) float64 — STL heightmap after apply_transform(),
                   in STL model units (arbitrary scale).
    osm          : (rows, cols) float64 — OSM building-height raster (metres).
    height_scale : float or None
        Slope: multiply stl_aligned by this. None → least-squares fit (with the
        intercept below) over the overlap region.
    height_offset : float or None
        Intercept (metres) added after scaling.  None with height_scale=None →
        fit jointly.  None with an explicit height_scale → 0.0.

    Returns
    -------
    ComparisonResult dataclass.
    """
    if stl_aligned.shape != osm.shape:
        raise ValueError(
            f"Shape mismatch: stl_aligned {stl_aligned.shape} vs osm {osm.shape}"
        )

    stl_valid = ~np.isnan(stl_aligned)
    osm_valid = ~np.isnan(osm)
    overlap = stl_valid & osm_valid

    # --- Per-building aggregation -----------------------------------------
    # Compare one representative height PER BUILDING, not per pixel.  Per-pixel
    # comparison is skewed by low street/edge cells that fall inside a footprint
    # (and by the STL's ragged residual at building borders).  For each OSM
    # building footprint we take the median (default) or max of the STL
    # height-above-terrain inside it; the OSM height is already constant per
    # footprint.  Buildings whose OSM height equals the dataset minimum are
    # OSM *fill defaults* (no real height tag — all set to the same value) and
    # are excluded.  (agg_stl_raw, agg_osm) are the per-building samples used for
    # the fit and every height statistic below.
    agg_stl_raw, agg_osm, _blabels, _bids = _aggregate_buildings(
        stl_aligned, osm, stl_valid, osm_valid, how=height_agg)
    have_buildings = agg_osm.size >= 2

    # --- Auto height fit: stl_m = stl * scale + offset ---
    # Affine least-squares (slope+intercept) on the PER-BUILDING samples, then a
    # residual-based outlier TRIM and refit.  The intercept absorbs a constant
    # base-plate / terrain bias; fitting on building aggregates keeps street cells
    # from dragging the slope; the trim removes the few off-line buildings (STL
    # trees/artefacts vs OSM short, or OSM mis-tags) WITHOUT discarding the tall
    # buildings (they sit on the line — low residual — so the trim keeps them).
    n_outliers = 0
    if height_scale is None:
        if have_buildings:
            fit_x, fit_y, fit_src = agg_stl_raw, agg_osm, f"per-building {height_agg}"
        elif overlap.sum() > 1:
            fit_x, fit_y, fit_src = stl_aligned[overlap], osm[overlap], "per-pixel"
        else:
            fit_x = fit_y = None
            fit_src = "none"
        if fit_x is not None:
            s_fit, b_fit, inl = _trimmed_linfit(fit_x, fit_y, trim=robust_fit)
            height_scale = float(s_fit)
            if height_offset is None:
                height_offset = float(b_fit)
            # Apply the same inlier mask to the per-building samples + ids.
            if have_buildings and inl is not None and inl.size == agg_osm.size:
                n_outliers = int((~inl).sum())
                agg_stl_raw, agg_osm = agg_stl_raw[inl], agg_osm[inl]
                _bids = [b for b, keep in zip(_bids, inl) if keep]
                have_buildings = agg_osm.size >= 2
            logger.info("Auto height fit (%s): scale=%.4f offset=%+.2f m "
                        "(trimmed %d outliers)", fit_src, height_scale,
                        height_offset if height_offset is not None else 0.0, n_outliers)
        else:
            height_scale = 1.0

    if height_offset is None:
        height_offset = 0.0

    # Per-building STL heights in metres (used for all height statistics).
    agg_stl = agg_stl_raw * height_scale + height_offset

    stl_m = stl_aligned * height_scale + height_offset

    # --- Difference maps ---
    # Per-pixel (honest pixel-level disagreement, incl. street cells):
    difference = np.full_like(osm, np.nan)
    difference[overlap] = stl_m[overlap] - osm[overlap]
    # Per-building "corrected" difference: one value per kept footprint
    # (fills + fit-outliers excluded) painted over the OSM footprint.
    building_diff_map = np.full_like(osm, np.nan)
    if _blabels is not None and have_buildings:
        for k, bid in enumerate(_bids):
            building_diff_map[_blabels == bid] = agg_stl[k] - agg_osm[k]

    # --- Missing coverage masks ---
    missing_in_osm = stl_valid & ~osm_valid
    missing_in_stl = osm_valid & ~stl_valid

    # --- Statistics over overlap ---
    n_overlap = int(overlap.sum())
    if n_overlap == 0:
        logger.warning("No overlapping pixels found; all stats will be NaN/0.")
        return ComparisonResult(
            difference=difference,
            building_diff_map=building_diff_map,
            overlap_mask=overlap,
            missing_in_osm=missing_in_osm,
            missing_in_stl=missing_in_stl,
            rmse=float("nan"),
            mae=float("nan"),
            bias=float("nan"),
            correlation=float("nan"),
            rank_correlation=float("nan"),
            mape=float("nan"),
            coverage_pct=0.0,
            footprint_iou=0.0,
            dice_score=0.0,
            n_overlap=0,
            height_scale_used=float(height_scale),
            height_offset_used=float(height_offset),
            height_ratio_mean=float("nan"),
            height_ratio_std=float("nan"),
        )

    # --- Height statistics: PER-BUILDING (median/max), fills excluded ---
    if have_buildings:
        bdiff = agg_stl - agg_osm
        rmse = float(np.sqrt(np.mean(bdiff ** 2)))
        mae = float(np.mean(np.abs(bdiff)))
        bias = float(np.mean(bdiff))
        if np.std(agg_stl) > 1e-9 and np.std(agg_osm) > 1e-9:
            corr = float(np.corrcoef(agg_stl, agg_osm)[0, 1])
            from scipy.stats import spearmanr
            rank_corr = float(spearmanr(agg_stl, agg_osm)[0])
        else:
            corr = rank_corr = float("nan")
        nz = agg_osm > 0.1
        mape = float(np.median(100.0 * np.abs((agg_stl[nz] - agg_osm[nz]) / agg_osm[nz]))) \
            if nz.any() else float("nan")
        ratio = agg_stl / np.maximum(agg_osm, 0.1)
        height_ratio_mean = float(np.median(ratio))
        height_ratio_std = float(np.std(ratio))
    else:
        # Fallback to per-pixel when buildings can't be segmented.
        diff_vals = difference[overlap]
        rmse = float(np.sqrt(np.mean(diff_vals ** 2)))
        mae = float(np.mean(np.abs(diff_vals)))
        bias = float(np.mean(diff_vals))
        corr = float(np.corrcoef(stl_m[overlap], osm[overlap])[0, 1])
        from scipy.stats import spearmanr
        rank_corr = float(spearmanr(stl_m[overlap], osm[overlap])[0])
        mape = float("nan")
        height_ratio_mean = float("nan")
        height_ratio_std = float("nan")

    n_osm_valid = int(osm_valid.sum())
    coverage_pct = 100.0 * n_overlap / n_osm_valid if n_osm_valid > 0 else 0.0

    # Footprint agreement over the UNION of building cells (honest registration
    # quality).  The STL footprint is the masked-aligned region (stl_valid); the OSM
    # footprint is its non-zero cells.  This is NOT conditioned on co-presence — the
    # old "& overlap" Dice counted only cells where both had buildings, so it read
    # ~1.0 by construction and hid real misalignment.
    #
    # osm_fp is restricted to stl_valid's bounding box: OSM is always fetched with
    # a margin around the STL's own footprint (DEFAULT_OSM_MARGIN), so a large
    # fraction of every frame is padding the STL was never going to cover —
    # counting OSM buildings there against the union permanently caps the
    # achievable IoU by how much padding was fetched, not by alignment quality.
    # Measured on Paris: the STL fills ~1/1.5≈67% of the frame per axis (~42% of
    # area) at osm_margin=1.5, and OSM building coverage is dense (58%) even in
    # the padding — deflating IoU from a true ~0.53 (score_alignment's cropped
    # overlap_iou over the same STL/OSM pair) down to ~0.25-0.33 for reasons
    # having nothing to do with the transform. This mirrors the crop already
    # applied to score_alignment's overlap_iou (align/metrics.py) and the
    # rotation-IoU sweep in global_search.py — same fix, applied here too.
    stl_fp = stl_valid
    osm_fp = osm > 0
    if stl_fp.any():
        _rows = np.where(stl_fp.any(axis=1))[0]
        _cols = np.where(stl_fp.any(axis=0))[0]
        _r0, _r1 = int(_rows[0]), int(_rows[-1]) + 1
        _c0, _c1 = int(_cols[0]), int(_cols[-1]) + 1
        _crop = np.zeros_like(osm_fp)
        _crop[_r0:_r1, _c0:_c1] = True
        osm_fp = osm_fp & _crop
    inter = int((stl_fp & osm_fp).sum())
    union = int((stl_fp | osm_fp).sum())
    n_stl_fp = int(stl_fp.sum()); n_osm_fp = int(osm_fp.sum())
    footprint_iou = inter / union if union > 0 else 0.0
    dice = (2.0 * inter) / (n_stl_fp + n_osm_fp) if (n_stl_fp + n_osm_fp) > 0 else 0.0

    logger.info(
        "Comparison: scale=%.4f offset=%+.2f m  RMSE=%.2f m  bias=%+.2f m  r=%.4f  "
        "rho=%.4f  MAPE=%.1f%%  ratio=%.3f +/-%.3f  footprint_IoU=%.3f dice=%.3f  "
        "coverage=%.1f%%  n_buildings=%d  n_px=%d",
        height_scale, height_offset, rmse, bias, corr, rank_corr, mape,
        height_ratio_mean, height_ratio_std, footprint_iou, dice, coverage_pct,
        int(agg_osm.size), n_overlap,
    )

    return ComparisonResult(
        difference=difference,
        building_diff_map=building_diff_map,
        overlap_mask=overlap,
        missing_in_osm=missing_in_osm,
        missing_in_stl=missing_in_stl,
        rmse=rmse,
        mae=mae,
        bias=bias,
        correlation=corr,
        rank_correlation=rank_corr,
        mape=mape,
        coverage_pct=coverage_pct,
        footprint_iou=footprint_iou,
        dice_score=dice,
        n_overlap=n_overlap,
        height_scale_used=float(height_scale),
        height_offset_used=float(height_offset),
        height_ratio_mean=height_ratio_mean,
        height_ratio_std=height_ratio_std,
    )


def _trimmed_linfit(x, y, trim=True, k=3.0):
    """
    Affine fit y ≈ s·x + b with an optional residual-based outlier trim.

    Fits ordinary least-squares, then (if trim) drops points whose residual
    exceeds k·(1.4826·MAD) — a robust σ — and refits on the survivors.  Unlike
    RANSAC's "largest consensus set", this keeps the sparse TALL buildings (they
    lie on the line → small residual) and only removes genuinely off-line
    outliers (STL trees/artefacts vs OSM-short, OSM mis-tags).

    Returns (slope, intercept, inlier_mask).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    inl = np.ones(n, dtype=bool)
    if not trim or n < 6:
        return float(s), float(b), inl

    resid = y - (s * x + b)
    mad = float(np.median(np.abs(resid - np.median(resid))))
    if mad <= 1e-9:
        return float(s), float(b), inl
    thresh = k * 1.4826 * mad
    inl = np.abs(resid) <= thresh
    if inl.sum() >= max(4, int(0.5 * n)):   # only trust the trim if it keeps the bulk
        A2 = np.vstack([x[inl], np.ones(int(inl.sum()))]).T
        (s, b), *_ = np.linalg.lstsq(A2, y[inl], rcond=None)
    else:
        inl = np.ones(n, dtype=bool)
    return float(s), float(b), inl


def _aggregate_buildings(stl_aligned, osm, stl_valid, osm_valid,
                         how="median", min_area=8):
    """
    One STL / OSM height sample per OSM building footprint.

    For each connected OSM footprint: OSM height is its median (constant per
    building); the STL value is the median (default) or max of the STL
    height-above-terrain inside the footprint — robust to low street/edge cells
    that would skew a per-pixel comparison.

    OSM *fill defaults* (untagged buildings all stamped with the same modal
    height, e.g. 10 m) are detected as the over-represented mode and excluded.

    Returns (stl_samples_raw, osm_samples, labels, kept_ids) — the 1-D sample
    arrays (STL still in model units; caller applies the fit), the OSM
    connected-component label image, and the label id for each sample (so a
    per-building map can be painted back onto the footprints).
    """
    empty = (np.array([]), np.array([]), None, [])
    try:
        import cv2
    except ImportError:
        return empty

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        osm_valid.astype(np.uint8), connectivity=8)
    if n <= 1:
        return empty

    hl = str(how).lower()
    if hl == "max":
        redux = np.nanmax
    elif hl == "median":
        redux = np.nanmedian
    elif hl.startswith("p") and hl[1:].replace(".", "", 1).isdigit():
        _pct = float(hl[1:])
        redux = lambda a: np.nanpercentile(a, _pct)   # noqa: E731
    else:
        redux = np.nanmedian

    ids, osm_h = [], []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < min_area:
            continue
        ids.append(i)
        osm_h.append(float(np.nanmedian(osm[labels == i])))
    if not ids:
        return empty
    osm_h = np.array(osm_h)

    # Fill default = the over-represented MODE of the heights (untagged buildings
    # are all stamped with the same default_height, e.g. 10 m — a sharp spike).
    # This is NOT the minimum: real short buildings (3.5 m) sit below the fill.
    # Exclude the modal value when it is a clear spike (>=10% of buildings at one
    # exact height — real heights never cluster that tightly on a single value).
    rounded = np.round(osm_h, 1)
    vals, counts = np.unique(rounded, return_counts=True)
    mode_val = float(vals[np.argmax(counts)])
    mode_cnt = int(counts.max())
    if mode_cnt >= max(5, int(0.10 * len(osm_h))):
        drop_fill = np.isclose(rounded, mode_val, atol=0.05)
        fill_val = mode_val
    else:
        drop_fill = np.zeros(len(osm_h), dtype=bool)
        fill_val = float("nan")

    s_samples, o_samples, kept_ids = [], [], []
    for k, i in enumerate(ids):
        if drop_fill[k]:
            continue
        comp = (labels == i) & stl_valid
        if int(comp.sum()) < min_area:
            continue
        s_samples.append(float(redux(stl_aligned[comp])))
        o_samples.append(osm_h[k])
        kept_ids.append(i)

    if int(drop_fill.sum()):
        logger.info("Per-building: %d footprints, excluded %d OSM fill-default "
                    "(=%.1f m mode) buildings, kept %d.",
                    len(ids), int(drop_fill.sum()), fill_val, len(s_samples))
    return (np.array(s_samples, dtype=float), np.array(o_samples, dtype=float),
            labels, kept_ids)
