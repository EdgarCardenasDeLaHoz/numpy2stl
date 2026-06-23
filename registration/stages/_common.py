"""Shared helpers for the registration pipeline stages.

Pure functions used by more than one stage (and by the orchestrator): affine
decomposition for the report, the landmark sanity check, and interior-NaN
inpainting of the STL heightmap.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _decompose_for_report(M: np.ndarray) -> tuple[float, float]:
    """Extract (scale, angle_deg) from a 2x3 affine or 3x3 homography matrix."""
    import math
    scale = float(math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    angle = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    return scale, angle


def _landmark_check(osm_hm: np.ndarray, transform: np.ndarray) -> dict | None:
    """
    Sanity-check registration against the largest tall building near the OSM
    bbox centre (City Hall for Philadelphia, or the dominant central landmark
    for any city whose bbox is centred on a known building).

    The OSM bbox is always centred on a known geographic point (e.g. City Hall),
    so the expected OSM landmark pixel = (cx, cy) = bbox centre ≈ (256, 256).
    We find the largest building component within 80px of the centre that has
    mean height > 40m, which is the best proxy for that landmark in the OSM data.

    After applying the inverse of the registration transform we can see where that
    landmark sits in the STL coordinate frame and report the residual error.
    """
    try:
        import cv2
        h, w = osm_hm.shape
        cy, cx = h // 2, w // 2

        from ..align import building_mask
        osm_mask = building_mask(osm_hm, source="osm").astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(osm_mask, connectivity=8)

        best = None
        for i in range(1, n):
            ccx, ccy = centroids[i]
            dist = float(((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5)
            if dist > 80:
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            mean_h = float(np.nanmean(osm_hm[labels == i]))
            if mean_h < 40 or area < 100:
                continue
            if best is None or area > best["area"]:
                best = {"area": area, "osm_col": float(ccx), "osm_row": float(ccy),
                        "mean_h": mean_h, "dist_from_center": dist}

        if best is None:
            return None

        # Map the OSM landmark pixel back into STL space via the inverse transform
        M3 = np.eye(3)
        M3[:2] = transform
        M_inv = np.linalg.inv(M3)
        ox, oy = best["osm_col"], best["osm_row"]
        stl_col = M_inv[0, 0] * ox + M_inv[0, 1] * oy + M_inv[0, 2]
        stl_row = M_inv[1, 0] * ox + M_inv[1, 1] * oy + M_inv[1, 2]

        # Pixel distance from STL centre (the model should be centred too)
        stl_cx, stl_cy = w / 2.0, h / 2.0
        stl_dist = float(((stl_col - stl_cx) ** 2 + (stl_row - stl_cy) ** 2) ** 0.5)

        return {
            "osm_row": best["osm_row"], "osm_col": best["osm_col"],
            "osm_dist_from_center": best["dist_from_center"],
            "osm_mean_h": best["mean_h"],
            "stl_row": stl_row, "stl_col": stl_col,
            "stl_dist_from_center": stl_dist,
        }
    except Exception as exc:
        logger.debug("landmark_check failed: %s", exc)
        return None


def _inpaint_stl_nan(hm: np.ndarray) -> np.ndarray:
    """
    Fill NaN holes in the STL heightmap by nearest-neighbour interpolation.

    The STL mesh doesn't cover every grid cell (gaps at boundary, thin walls,
    mesh holes). NaN pixels would propagate into the morphological top-hat and
    edge detection as false edges.  Nearest-neighbour fill is fast and correct
    for this use: we just need a plausible height so the terrain model is
    continuous, not a precise interpolated value.
    """
    nan_mask = np.isnan(hm)
    if not nan_mask.any():
        return hm
    try:
        from scipy.ndimage import distance_transform_edt, label

        # Only fill *interior* holes — NaN regions enclosed by the mesh.  NaN that
        # is connected to the image border is exterior padding (added when the
        # model is rendered isotropically into a square canvas) and must stay NaN,
        # otherwise nearest-neighbour fill would smear building heights into the
        # empty margin and re-introduce a stretch-like artefact.
        structure = np.ones((3, 3), dtype=int)
        lbl, n_lbl = label(nan_mask, structure=structure)
        border_ids = set(np.unique(np.concatenate([
            lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])).tolist())
        border_ids.discard(0)
        exterior = np.isin(lbl, list(border_ids)) if border_ids else np.zeros_like(nan_mask)
        interior = nan_mask & ~exterior

        filled = hm.copy()
        if interior.any():
            _, idx = distance_transform_edt(nan_mask, return_indices=True)
            interior_idx = (idx[0][interior], idx[1][interior])
            filled[interior] = hm[interior_idx]
        logger.debug(
            "STL inpaint: filled %d interior NaN px; left %d exterior padding px",
            int(interior.sum()), int(exterior.sum()))
        return filled
    except ImportError:
        # scipy not available — median fill as last resort
        median_val = float(np.nanmedian(hm))
        filled = hm.copy()
        filled[nan_mask] = median_val
        return filled
