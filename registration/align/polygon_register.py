"""Polygon-based global registration (building point-pattern matching).

An alternative to the raster `register_global`: instead of cross-correlating
heightmaps, it matches the STL and OSM **building footprints** directly and solves
the STL→OSM similarity transform from the correspondences.  This removes the
raster method's three fragilities — it needs no dominant street grid (rotation),
no base-plate translation anchor, and is far less sensitive to segmentation
quality (it works on the clean vector footprints).

Method: RANSAC over centroid correspondences.  A similarity (scale, rotation,
translation) is fixed by two STL↔OSM centroid pairs; each candidate is scored by
how many STL centroids land within ε of an OSM centroid after the transform
(`scipy.spatial.cKDTree`), then refit from all inliers (Umeyama via
`skimage.transform.estimate_transform`).  When `scale_prior` is given (the
geometric anchor 1/osm_margin, which the assumption tests showed is exact), scale
is PINNED — the only unknowns are rotation+translation, so a candidate pair is
accepted only when its separation ratio matches the known scale.  Pinning scale
makes matching tractable despite the STL carrying only a fraction of OSM's
buildings (low inlier ratio).
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


def _descriptors(polys):
    """Per-polygon (centroid_xy, area, orientation_rad, eccentricity)."""
    cents, areas, orients, eccs = [], [], [], []
    for p in polys:
        p = np.asarray(p, dtype=np.float64)
        m = cv2.moments(p.astype(np.float32))
        a = m["m00"]
        if a <= 1e-6:                       # degenerate → fall back to vertex mean
            c = p.mean(axis=0)
            cents.append(c); areas.append(max(a, 1.0)); orients.append(0.0); eccs.append(0.0)
            continue
        cx, cy = m["m10"] / a, m["m01"] / a
        mu20, mu02, mu11 = m["mu20"] / a, m["mu02"] / a, m["mu11"] / a
        orient = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        # eccentricity from the covariance eigenvalues
        tr, det = mu20 + mu02, mu20 * mu02 - mu11 ** 2
        disc = max(0.0, (tr / 2) ** 2 - det)
        l1 = tr / 2 + np.sqrt(disc); l2 = tr / 2 - np.sqrt(disc)
        ecc = float(np.sqrt(1 - l2 / l1)) if l1 > 1e-9 else 0.0
        cents.append([cx, cy]); areas.append(float(a)); orients.append(float(orient)); eccs.append(ecc)
    return (np.asarray(cents, dtype=np.float64), np.asarray(areas),
            np.asarray(orients), np.asarray(eccs))


def _similarity_matrix(s, theta, tx, ty):
    c, sn = s * np.cos(theta), s * np.sin(theta)
    return np.array([[c, -sn, tx], [sn, c, ty]], dtype=np.float64)


def register_polygons(
    stl_polys,
    osm_polys,
    scale_prior: float | None = None,
    max_scale_ratio: float = 5.0,
    n_iter: int = 6000,
    inlier_frac: float = 0.5,
    min_confidence: float = 0.40,
    seed: int = 0,
) -> dict:
    """Register STL footprints onto OSM footprints by building point-pattern matching.

    Parameters
    ----------
    stl_polys, osm_polys : list[(K,2)] polygon vertex arrays in their own pixel grids.
    scale_prior : if given, scale is PINNED to it (only rotation+translation solved).
    inlier_frac : an STL centroid is an inlier if within `inlier_frac` × (median OSM
                  building radius) of an OSM centroid.

    Returns
    -------
    dict: transform (2,3 STL→OSM), scale, angle_deg, n_inliers, confidence (inlier
          fraction of STL buildings), applied (bool).
    """
    out = {"transform": np.array([[1.0, 0, 0], [0, 1.0, 0]]), "scale": 1.0,
           "angle_deg": 0.0, "n_inliers": 0, "confidence": 0.0, "applied": False,
           "reason": ""}
    if not HAS_CV2:
        out["reason"] = "opencv unavailable"; return out
    if len(stl_polys) < 3 or len(osm_polys) < 3:
        out["reason"] = f"too few polygons (stl={len(stl_polys)}, osm={len(osm_polys)})"
        return out

    sc, _sa, _so, _se = _descriptors(stl_polys)
    oc, _oa, _oo, _oe = _descriptors(osm_polys)
    try:
        from scipy.spatial import cKDTree
    except Exception:
        out["reason"] = "scipy unavailable"; return out
    tree = cKDTree(oc)

    # Inlier radius: a fraction of the typical OSM building size.
    osm_radius = float(np.median(np.sqrt(_oa))) if len(_oa) else 5.0
    eps = max(3.0, inlier_frac * osm_radius)

    rng = np.random.default_rng(seed)
    ns, no = len(sc), len(oc)
    smin, smax = (scale_prior * 0.9, scale_prior * 1.1) if scale_prior else \
                 (1.0 / max_scale_ratio, max_scale_ratio)

    # Descriptor-guided candidates: for each STL building, the OSM buildings whose
    # AREA (≈ area_stl·scale²) and ECCENTRICITY are compatible.  Random OSM pairing
    # is ~1/no² likely to hit a true match; restricting to descriptor candidates
    # raises the RANSAC hit-rate by orders of magnitude (essential for matching).
    s2 = float(scale_prior) ** 2 if scale_prior else None
    cand = []
    for i in range(ns):
        if s2 is not None:
            area_ok = np.abs(_oa - _sa[i] * s2) <= 0.6 * _sa[i] * s2
        else:
            area_ok = np.ones(no, dtype=bool)
        ecc_ok = np.abs(_oe - _se[i]) <= 0.35
        c = np.where(area_ok & ecc_ok)[0]
        cand.append(c if len(c) else np.arange(no))
    stl_pool = [i for i in range(ns) if len(cand[i]) > 0] or list(range(ns))

    def _count_inliers(M):
        warped = sc @ M[:, :2].T + M[:, 2]
        d, _ = tree.query(warped)
        return int((d <= eps).sum())

    best = (0, None)  # (n_inliers, M)
    counts = []       # inlier count of every valid candidate (the null distribution)
    for _ in range(n_iter):
        i, j = rng.choice(stl_pool, 2, replace=False)
        k = int(rng.choice(cand[i])); l = int(rng.choice(cand[j]))
        if k == l:
            continue
        vs = sc[j] - sc[i]; vo = oc[l] - oc[k]
        ls, lo = np.hypot(*vs), np.hypot(*vo)
        if ls < 1e-3 or lo < 1e-3:
            continue
        s = lo / ls
        if not (smin <= s <= smax):
            continue
        if scale_prior is not None:
            s = float(scale_prior)            # pin scale; pair only gates rotation
        theta = np.arctan2(vo[1], vo[0]) - np.arctan2(vs[1], vs[0])
        c, sn = s * np.cos(theta), s * np.sin(theta)
        # translation so STL point i maps onto OSM point k
        tx = oc[k, 0] - (c * sc[i, 0] - sn * sc[i, 1])
        ty = oc[k, 1] - (sn * sc[i, 0] + c * sc[i, 1])
        M = np.array([[c, -sn, tx], [sn, c, ty]], dtype=np.float64)
        n_in = _count_inliers(M)
        counts.append(n_in)
        if n_in > best[0]:
            best = (n_in, M)

    # Significance gate: a genuine match's inlier count is a strong OUTLIER above the
    # random-candidate distribution; an unrelated "best" is just the noise tail (on a
    # dense OSM, random transforms spuriously hit ~40% by chance, so inlier *fraction*
    # alone can't reject — the z-score can).
    conf = best[0] / max(1, ns)
    counts = np.asarray(counts, dtype=np.float64)
    z = (best[0] - counts.mean()) / (counts.std() + 1e-9) if counts.size else 0.0
    out["confidence"] = float(conf)
    if best[1] is None or best[0] < 4 or z < 5.0 or conf < min_confidence:
        out["reason"] = (f"low consensus (inliers={best[0]}/{ns}, conf={conf:.2f}, "
                         f"z={z:.1f} vs random)")
        return out

    # Refit the similarity from ALL inlier correspondences (Umeyama).
    M = best[1]
    warped = sc @ M[:, :2].T + M[:, 2]
    d, idx = tree.query(warped)
    inl = d <= eps
    src, dst = sc[inl], oc[idx[inl]]
    if len(src) >= 2:
        try:
            from skimage.transform import estimate_transform
            tf = estimate_transform("similarity", src, dst)
            Mr = np.asarray(tf.params[:2], dtype=np.float64)
            if scale_prior is not None:                 # re-pin scale, keep rot+trans
                rs = np.hypot(Mr[0, 0], Mr[1, 0])
                if rs > 1e-9:
                    Mr[:, :2] *= (scale_prior / rs)
            if np.all(np.isfinite(Mr)):
                M = Mr
        except Exception:
            pass

    s_final = float(np.hypot(M[0, 0], M[1, 0]))
    ang = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    out.update(transform=M, scale=s_final, angle_deg=ang,
               n_inliers=int(best[0]), confidence=float(best[0] / max(1, ns)),
               applied=True, reason="matched")
    logger.info("polygon register: %d/%d STL buildings inliers (%.0f%%)  scale=%.3f rot=%.2f° eps=%.1fpx",
                best[0], ns, 100.0 * best[0] / max(1, ns), s_final, ang, eps)
    return out
