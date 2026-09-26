"""Polygon-based registration fine-tuning (building-matched ICP).

After the coarse registration (gradient rotation + xcorr scale/translation + ECC)
has locked the STL footprints onto the OSM footprints, the alignment can be
sharpened by working with the *vectorized building polygons* directly: match each
STL building to the OSM building it overlaps, then iteratively align the polygon
boundary points (ICP) to solve a small corrective similarity transform.

This runs ONLY when the coarse registration already succeeded — gated on
``dice > min_dice`` (default 0.95).  Below that the footprints don't correspond
well enough for building matching to be safe, so we leave the transform untouched.

The correction is accepted only if it does not increase the mean residual
(point-to-boundary distance) — a guard against ICP drifting on a poor match.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None
    HAS_CV2 = False


def _warp_pts(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2×3 affine to (N,2) points."""
    return pts @ M[:, :2].T + M[:, 2]


def _densify(poly: np.ndarray, step: float = 2.0) -> np.ndarray:
    """Resample a closed polygon's boundary to ~`step`-pixel spacing.

    ICP needs comparable point density on both shapes; raw polygons have a few
    vertices far apart, so we interpolate points along each edge.
    """
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 2:
        return p
    loop = np.vstack([p, p[:1]])
    out = []
    for a, b in zip(loop[:-1], loop[1:]):
        d = float(np.hypot(*(b - a)))
        n = max(1, int(d / step))
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        out.append(a[None, :] * (1 - ts[:, None]) + b[None, :] * ts[:, None])
    return np.vstack(out) if out else p


def _centroid(poly: np.ndarray) -> np.ndarray:
    return np.asarray(poly, dtype=np.float64).mean(axis=0)


def _match_buildings(stl_polys, osm_polys, max_dist):
    """Greedy nearest-centroid matching of warped STL polys to OSM polys.

    Returns list of (stl_idx, osm_idx) pairs whose centroids are within
    ``max_dist`` pixels of each other.
    """
    if not stl_polys or not osm_polys:
        return []
    sc = np.array([_centroid(p) for p in stl_polys])
    oc = np.array([_centroid(p) for p in osm_polys])
    pairs = []
    used_osm = set()
    # distance matrix (footprint counts are small — a few hundred at most)
    d = np.linalg.norm(sc[:, None, :] - oc[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used_stl = set()
    for si, oi in order:
        if d[si, oi] > max_dist:
            break
        if si in used_stl or oi in used_osm:
            continue
        used_stl.add(si); used_osm.add(oi)
        pairs.append((int(si), int(oi)))
    return pairs


def refine_registration_polygons(
    stl_polys,
    osm_polys,
    transform: np.ndarray,
    dice: float,
    min_dice: float = 0.95,
    n_iter: int = 12,
    match_dist_px: float = 25.0,
) -> dict:
    """Building-matched ICP refinement of an STL→OSM similarity transform.

    Parameters
    ----------
    stl_polys : list[(K,2)] polygons in STL/working pixel space (from
        `vectorize_buildings`, ideally regularized).
    osm_polys : list[(K,2)] polygons in OSM pixel space.
    transform : (2,3) affine mapping STL pixels → OSM pixels (the coarse result).
    dice      : footprint Dice of the coarse registration (the success gate).

    Returns
    -------
    dict: {transform, applied (bool), n_matched, rmse_before, rmse_after, reason}
    """
    out = {"transform": np.asarray(transform, dtype=np.float64), "applied": False,
           "n_matched": 0, "rmse_before": float("nan"), "rmse_after": float("nan"),
           "reason": ""}
    if not HAS_CV2:
        out["reason"] = "opencv unavailable"; return out
    if dice is None or dice <= min_dice:
        out["reason"] = f"dice {dice} <= {min_dice}; skipped"; return out
    if not stl_polys or not osm_polys:
        out["reason"] = "no polygons"; return out

    M = np.asarray(transform, dtype=np.float64).copy()

    # Warp STL polygons into OSM space with the coarse transform, then match
    # them to OSM buildings by centroid.
    stl_w = [_warp_pts(np.asarray(p, np.float64), M) for p in stl_polys]
    pairs = _match_buildings(stl_w, osm_polys, max_dist=match_dist_px)
    out["n_matched"] = len(pairs)
    if len(pairs) < 3:
        out["reason"] = f"only {len(pairs)} building matches (<3)"; return out

    # Build dense correspondence clouds from matched polygon boundaries.
    src = np.vstack([_densify(stl_w[si]) for si, _ in pairs])     # warped STL pts
    tgt_clouds = [_densify(np.asarray(osm_polys[oi], np.float64)) for _, oi in pairs]
    tgt_all = np.vstack(tgt_clouds)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(tgt_all)
    except Exception:
        out["reason"] = "scipy KDTree unavailable"; return out

    def _rmse(pts):
        dist, _ = tree.query(pts)
        return float(np.sqrt(np.mean(dist ** 2)))

    out["rmse_before"] = _rmse(src)

    # ICP: nearest OSM boundary point for each src point → estimate a corrective
    # similarity (scale+rot+trans, no shear) → apply → repeat.
    delta = np.array([[1.0, 0, 0], [0, 1.0, 0]], dtype=np.float64)
    cur = src.copy()
    for _ in range(n_iter):
        _, idx = tree.query(cur)
        dst = tgt_all[idx]
        Mest, inliers = cv2.estimateAffinePartial2D(
            cur.astype(np.float32), dst.astype(np.float32),
            method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if Mest is None:
            break
        cur = _warp_pts(cur, Mest)
        delta = _compose(Mest, delta)
        # stop when the step is sub-pixel
        if abs(Mest[0, 2]) < 0.05 and abs(Mest[1, 2]) < 0.05 and \
           abs(np.hypot(Mest[0, 0], Mest[0, 1]) - 1.0) < 1e-3:
            break

    rmse_after = _rmse(cur)
    out["rmse_after"] = rmse_after
    # Accept only if ICP actually reduced the residual (guard against drift).
    if rmse_after <= out["rmse_before"] + 1e-6:
        out["transform"] = _compose(delta, M)
        out["applied"] = True
        out["reason"] = "refined"
    else:
        out["reason"] = f"rejected (rmse {out['rmse_before']:.2f} -> {rmse_after:.2f})"
    logger.info("polygon ICP: matched=%d  rmse %.2f -> %.2f px  %s",
                len(pairs), out["rmse_before"], rmse_after, out["reason"])
    return out


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """2×3 affine for x → outer(inner(x))."""
    O = np.vstack([outer, [0, 0, 1]])
    I = np.vstack([inner, [0, 0, 1]])
    return (O @ I)[:2].astype(np.float64)
