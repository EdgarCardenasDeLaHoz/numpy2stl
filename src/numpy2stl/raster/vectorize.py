"""Raster mask → vector polygons (building footprints).

- ``vectorize_buildings``  trace + Douglas–Peucker simplify each footprint
- ``_regularize_polygon``  snap a footprint to its dominant orthogonal axes

Polygons are in pixel (x=col, y=row) coordinates of the input mask.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


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
