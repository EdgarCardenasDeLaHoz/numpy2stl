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

__all__ = ["orient_ccw", "close_surface", "prism", "prisms"]


def prisms(polys, z0, z1) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """:func:`prism` for many polygons at once: one GEOS call per step for all of
    them, then a slice per polygon.

    ``z0`` / ``z1`` are scalars or one value per polygon. Same vertices and faces
    as :func:`prism` (up to face order); ``None`` where it returns None. The
    per-call overhead of :func:`prism` (shapely's per-geometry wrappers, one
    ``np.unique`` each) dominated a city of 25k buildings: ~20 s against ~2 s.
    """
    polys = np.asarray(polys, dtype=object)
    n = len(polys)
    if not n:
        return []
    z0 = np.broadcast_to(np.asarray(z0, np.float64), (n,))
    z1 = np.broadcast_to(np.asarray(z1, np.float64), (n,))
    polys = shapely.orient_polygons(polys)            # exterior CCW, holes CW
    tparts, towner = shapely.get_parts(shapely.constrained_delaunay_triangles(polys),
                                       return_index=True)
    tri = shapely.get_coordinates(tparts).reshape(-1, 4, 2)[:, :3].reshape(-1, 2)
    rings, rowner = shapely.get_rings(polys, return_index=True)
    rc, rring = shapely.get_coordinates(rings, return_index=True)
    last = np.r_[rring[1:] != rring[:-1], True]       # closing coordinate of each ring
    rc, rring = rc[~last], rring[~last]
    rpoly = rowner[rring]
    # One vertex table for all polygons: rows (polygon, x, y), deduplicated.
    pid = np.concatenate([rpoly, np.repeat(towner, 3)])
    xy = np.concatenate([rc, tri])
    order = np.lexsort((xy[:, 1], xy[:, 0], pid))
    ps, xs = pid[order], xy[order]
    new = np.r_[True, (ps[1:] != ps[:-1]) | (xs[1:] != xs[:-1]).any(axis=1)]
    gid = np.empty(len(pid), np.int64)
    gid[order] = np.cumsum(new) - 1
    upid, uxy = ps[new], xs[new]
    vstart = np.searchsorted(upid, np.arange(n + 1))
    ring_g, tri_g = gid[:len(rc)], gid[len(rc):].reshape(-1, 3)
    # Top triangles counter-clockwise.
    p = uxy[tri_g]
    ccw = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
           - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])) > 0
    tri_g[~ccw] = tri_g[~ccw][:, ::-1]
    # Wall edges: consecutive ring vertices (solid on their left, see prism).
    nxt = np.arange(1, len(ring_g) + 1)
    ends = np.r_[rring[1:] != rring[:-1], True]
    first = np.searchsorted(rring, rring)             # start of each vertex's ring
    nxt[ends] = first[ends]
    ea, eb = ring_g, ring_g[nxt]
    tstart = np.searchsorted(towner, np.arange(n + 1))
    estart = np.searchsorted(rpoly, np.arange(n + 1))
    out: list = []
    for k in range(n):
        v0, v1 = vstart[k], vstart[k + 1]
        t = tri_g[tstart[k]:tstart[k + 1]] - v0
        if z1[k] - z0[k] <= 1e-9 or not len(t):
            out.append(None)
            continue
        m = v1 - v0
        a, b = ea[estart[k]:estart[k + 1]] - v0, eb[estart[k]:estart[k + 1]] - v0
        v = np.empty((2 * m, 3))
        v[:m, :2] = v[m:, :2] = uxy[v0:v1]
        v[:m, 2], v[m:, 2] = z1[k], z0[k]
        out.append((v, np.vstack([t, t[:, ::-1] + m, np.column_stack([a, b + m, b]),
                                  np.column_stack([a, a + m, b + m])])))
    return out


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
