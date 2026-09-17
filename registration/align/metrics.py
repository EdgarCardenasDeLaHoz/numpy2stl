"""Alignment scoring: tolerant IoU, Dice, SDF, score_alignment.

Part of the align/ subpackage (split from the former align.py).
"""
from __future__ import annotations

import logging
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

from .segmentation import building_edges, building_mask
from .transform import apply_transform

def _tolerant_iou(s_mask: np.ndarray, t_mask: np.ndarray, tol_px: int = 1) -> float:
    """
    IoU with a small edge tolerance, so sub-block street gaps don't penalize.

    The STL models often fill whole blocks solid while OSM leaves street gaps;
    dilating both masks by tol_px before intersecting lets matching blocks count
    as agreement even when their footprint edges differ by a pixel or two. The
    union stays on the original masks so the metric isn't inflated arbitrarily.
    """
    if tol_px <= 0 or not HAS_CV2:
        inter = np.logical_and(s_mask, t_mask).sum()
        union = np.logical_or(s_mask, t_mask).sum()
        return float(inter / union) if union > 0 else 0.0

    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    s_d = cv2.dilate(s_mask.astype(np.uint8), k).astype(bool)
    t_d = cv2.dilate(t_mask.astype(np.uint8), k).astype(bool)
    # A pixel "agrees" if either mask hits the other's dilated footprint.
    agree = (s_mask & t_d) | (t_mask & s_d)
    union = np.logical_or(s_mask, t_mask).sum()
    return float(agree.sum() / union) if union > 0 else 0.0


def _dice(s_mask: np.ndarray, t_mask: np.ndarray, tol_px: int = 1) -> float:
    """
    Dice / F1 coefficient of two footprint masks: 2|A∩B| / (|A|+|B|).

    Symmetric and building-focused (unlike IoU it weights agreement vs each
    mask's own size, so it isn't dominated by union area).  Uses the same small
    dilation tolerance as _tolerant_iou so 1–2 px street-gap differences between
    solid STL blocks and gapped OSM footprints don't penalize a true match.
    """
    a = s_mask.astype(bool)
    b = t_mask.astype(bool)
    na, nb = int(a.sum()), int(b.sum())
    if na + nb == 0:
        return 0.0
    if tol_px > 0 and HAS_CV2:
        k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
        a_d = cv2.dilate(a.astype(np.uint8), k).astype(bool)
        b_d = cv2.dilate(b.astype(np.uint8), k).astype(bool)
        inter = int(((a & b_d) | (b & a_d)).sum())
    else:
        inter = int((a & b).sum())
    # Tolerant dilation can let the agreement count exceed min(na, nb); clamp so
    # Dice stays in [0, 1].
    return float(min(1.0, 2.0 * inter / (na + nb)))


def _mask_sdf(mask: np.ndarray) -> np.ndarray:
    """Signed distance field of a binary mask, normalized to [-1, 1]."""
    m = mask.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    sdf = inside - outside
    rng = np.abs(sdf).max()
    if rng > 0:
        sdf = sdf / rng
    return sdf.astype(np.float32)


def score_alignment(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    cell_size_m: float | None = None,
    allow_forced_split: bool = True,
    source_kind: str = "stl",
) -> dict:
    """
    Objective registration-quality metrics for a given source→target transform.

    Warps the source heightmap by `transform` into the target frame, then
    measures how well the two agree.  Independent of how the transform was
    produced, so it's the fair comparator for different strategies.

    `cell_size_m` sizes the STL top-hat kernel in real metres (see
    terrain_residual()) — pass the target grid's metres/pixel (e.g.
    report.cell_size_m) so this scores the same segmentation the search
    actually used; omitting it falls back to resolution-naive pixel sizing.

    `source_kind` says how to read the source raster.  The default 'stl' treats it
    as absolute model z and estimates a ground surface first.  Pass 'osm' when the
    source already holds height-above-ground with NaN for no building — the export's
    `stl_heightmap.npy` does — since estimating a ground surface for a raster whose
    ground has already been removed throws most of the buildings away.

    Returns
    -------
    dict:
        'footprint_iou'      : IoU of filled building masks over the FULL frame
                               (deflated by the large empty margin around the STL)
        'footprint_baseline' : random-chance IoU at these mask densities
        'footprint_lift'     : footprint_iou / footprint_baseline (>1 = real signal)
        'overlap_iou'        : IoU of the filled masks CROPPED to the STL extent —
                               the honest filled-footprint-overlap number the
                               footprint_rgchannel.png overlay reports in its title,
                               and the metric that visually tracks alignment quality
                               (~0.30–0.53 for good cities). This is the gate metric.
        'overlap_baseline'   : random-chance IoU at the two mask densities in the crop
        'overlap_lift'       : overlap_iou / overlap_baseline (context only — low for
                               dense-grid cities even when genuine, so NOT a gate)
        'overlap_precision'  : |both| / |STL footprint| in the crop — fraction of the
                               model's footprint OSM confirms. Collapses on a broken /
                               wrong-quadrant fit; the second gate factor.
        'overlap_dice'       : Dice of the cropped filled masks, 2|both|/(|STL|+|OSM|)
        'edge_iou'           : IoU of footprint *outline* masks (sparse; the old gate)
        'edge_baseline'      : random-chance edge IoU
        'edge_lift'          : edge_iou / edge_baseline (>~3 = genuine alignment)
        'footprint_dice'     : Dice coefficient of the filled masks (0–1)
        'height_corr'        : Pearson r of heights over the overlap (−1..1)
        'n_overlap'          : overlapping valid-pixel count
    """
    aligned = apply_transform(source, transform, output_shape=target.shape)

    s_mask = building_mask(aligned, source=source_kind, cell_size_m=cell_size_m,
                           allow_forced_split=allow_forced_split)
    t_mask = building_mask(target, source="osm")

    def _iou(a, b):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter / union) if union > 0 else 0.0

    def _baseline(a, b):
        # Expected IoU if the two masks were independent at their densities.
        pa, pb = a.mean(), b.mean()
        inter = pa * pb
        union = pa + pb - inter
        return float(inter / union) if union > 0 else 0.0

    iou = _iou(s_mask, t_mask)
    base = _baseline(s_mask, t_mask)
    dice = float(2 * np.logical_and(s_mask, t_mask).sum()
                 / (s_mask.sum() + t_mask.sum())) if (s_mask.sum() + t_mask.sum()) > 0 else 0.0

    # Filled-overlap IoU cropped to the STL extent — the honest alignment number.
    # The full-frame footprint_iou above is diluted by the large empty OSM margin
    # around the (smaller) STL footprint, so a badly-misaligned model can still read
    # a middling full-frame IoU.  Cropping to the STL's own extent (plus a small
    # margin) is exactly what the footprint_rgchannel.png overlay shows in its title,
    # and it is what visually reflects whether the two footprints actually agree.
    overlap_iou = iou
    overlap_base = base
    overlap_prec = float(np.logical_and(s_mask, t_mask).sum() / s_mask.sum()) if s_mask.sum() > 0 else 0.0
    overlap_dice = dice
    rows = np.where(s_mask.any(axis=1))[0]
    cols = np.where(s_mask.any(axis=0))[0]
    if len(rows) and len(cols):
        mr = max(10, int((rows[-1] - rows[0]) * 0.05))
        mc = max(10, int((cols[-1] - cols[0]) * 0.05))
        r0, r1 = max(0, rows[0] - mr), min(s_mask.shape[0], rows[-1] + mr + 1)
        c0, c1 = max(0, cols[0] - mc), min(s_mask.shape[1], cols[-1] + mc + 1)
        s_crop, t_crop = s_mask[r0:r1, c0:c1], t_mask[r0:r1, c0:c1]
        overlap_iou = _iou(s_crop, t_crop)
        # Random-chance IoU at the two densities WITHIN the crop (kept for context).
        overlap_base = _baseline(s_crop, t_crop)
        inter_c = float(np.logical_and(s_crop, t_crop).sum())
        ns, nt = float(s_crop.sum()), float(t_crop.sum())
        # Precision = fraction of the STL model's footprint that OSM CONFIRMS.  A
        # wrong-quadrant / misaligned model stamps large blocks where OSM has no
        # building, so precision collapses even when the raw IoU (buoyed by a few
        # coincidental grid hits) looks middling.  This is the factor that cleanly
        # separates a visually-broken city (low precision) from a genuine one.
        overlap_prec = inter_c / ns if ns > 0 else 0.0
        # Dice of the cropped filled masks — symmetric agreement, in [0,1].
        overlap_dice = (2.0 * inter_c / (ns + nt)) if (ns + nt) > 0 else 0.0

    # Edge masks — the honest signal (sparse outlines, low random baseline).
    s_edge = building_edges(aligned, source="stl", allow_forced_split=allow_forced_split)
    t_edge = building_edges(target, source="osm")
    edge_iou = _tolerant_iou(s_edge, t_edge, tol_px=2)
    edge_base = _baseline(s_edge, t_edge)

    overlap = (~np.isnan(aligned)) & (~np.isnan(target))
    n_overlap = int(overlap.sum())
    if n_overlap > 10 and aligned[overlap].std() > 0 and target[overlap].std() > 0:
        height_corr = float(np.corrcoef(aligned[overlap], target[overlap])[0, 1])
    else:
        height_corr = float("nan")

    return {
        "footprint_iou": iou,
        "footprint_baseline": base,
        "footprint_lift": float(iou / base) if base > 0 else float("nan"),
        "overlap_iou": overlap_iou,
        "overlap_baseline": overlap_base,
        "overlap_lift": float(overlap_iou / overlap_base) if overlap_base > 0 else float("nan"),
        "overlap_precision": overlap_prec,
        "overlap_dice": overlap_dice,
        "edge_iou": edge_iou,
        "edge_baseline": edge_base,
        "edge_lift": float(edge_iou / edge_base) if edge_base > 0 else float("nan"),
        "footprint_dice": dice,
        "height_corr": height_corr,
        "n_overlap": n_overlap,
    }
