"""Indexed prisms and closed solids from 2D outlines and open surfaces.

All functions return ``(vertices (N, 3) float64, faces (M, 3) int)``: indexed,
watertight, outward-facing. Unlike ``core.generate.polygon_to_prism`` (triangle
soup), these meshes can go straight into a boolean engine.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

__all__ = ["orient_ccw", "close_surface", "prism"]


def orient_ccw(xy: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces reordered to run counter-clockwise in the xy plane (normals +z)."""
    f = np.array(faces, copy=True)
    p = np.asarray(xy)[f][:, :, :2]
    ccw = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
           - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) > 0
    f[~ccw] = f[~ccw][:, ::-1]
    return f


def close_surface(top: np.ndarray, faces: np.ndarray, bottom_z) -> tuple[np.ndarray, np.ndarray]:
    """Solid from an upward-facing open surface (CCW faces): walls down to
    ``bottom_z(xy) -> z`` along its boundary, and the surface copied there reversed."""
    n = len(top)
    bot = np.array(top, dtype=np.float64, copy=True)
    bot[:, 2] = bottom_z(bot[:, :2])
    f = np.asarray(faces)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, counts = np.unique(np.sort(edges, axis=1), axis=0,
                               return_inverse=True, return_counts=True)
    border = edges[counts[inv.ravel()] == 1]          # directed as in the CCW top faces
    a, b = border[:, 0], border[:, 1]
    walls = np.vstack([np.column_stack([a, b + n, b]), np.column_stack([a, a + n, b + n])])
    return np.vstack([top, bot]), np.vstack([f, f[:, ::-1] + n, walls])


def prism(poly: Polygon, z0: float, z1: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed prism over a polygon (holes kept) using only the outline's own vertices.

    Returns ``None`` for a zero-height or degenerate polygon.
    """
    if z1 - z0 <= 1e-9:
        return None
    poly = orient(poly, 1.0)   # exterior CCW, holes CW
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    tri = shapely.get_coordinates(shapely.constrained_delaunay_triangles(poly)).reshape(-1, 4, 2)[:, :3]
    if not len(tri):
        return None
    ring_pts = np.concatenate(rings)
    uniq, inv = np.unique(np.concatenate([ring_pts, tri.reshape(-1, 2)]), axis=0, return_inverse=True)
    inv = inv.ravel()
    ring_idx, f = inv[:len(ring_pts)], inv[len(ring_pts):].reshape(-1, 3)
    f = orient_ccw(uniq, f)
    n = len(uniq)
    walls = []
    start = 0
    for r in rings:
        k = ring_idx[start:start + len(r)]
        a, b = k, np.roll(k, -1)
        walls += [np.column_stack([a, b + n, b]), np.column_stack([a, a + n, b + n])]
        start += len(r)
    v = np.vstack([np.column_stack([uniq, np.full(n, z1)]), np.column_stack([uniq, np.full(n, z0)])])
    # Rings run with the solid on their left (exterior CCW, holes CW), so each wall
    # quad (top a, bottom b, top b)+(top a, bottom a, bottom b) faces outward.
    return v, np.vstack([f, f[:, ::-1] + n] + walls)
