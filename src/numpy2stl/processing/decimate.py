"""Error-bounded mesh reduction.

* :func:`heightfield_tin` - adaptive triangulation of a raster heightfield whose
  linear interpolation stays within ``max_error`` of every pixel (exact bound;
  vertices stay on pixel centres). Used for terrain meshes.
* :func:`decimate_to_tolerance` - quadric decimation of an arbitrary mesh, as
  aggressive as a sampled symmetric Hausdorff bound allows. Used for imported
  building / city meshes.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)

__all__ = ["heightfield_tin", "decimate_to_tolerance"]


def _triangulate_pixels(ij: np.ndarray) -> np.ndarray:
    """Delaunay triangles over integer pixel coordinates (Shewchuk's Triangle)."""
    import triangle

    return triangle.triangulate({"vertices": ij.astype(np.float64)}, "Q")["triangles"]


def _rasterize_tin(xy: np.ndarray, zv: np.ndarray, tris: np.ndarray,
                   shape: tuple[int, int]) -> np.ndarray:
    """Linear interpolation of a TIN whose vertices lie on pixel centres, per pixel.

    Scan-converts every triangle's bounding box at once (vectorised) instead of
    locating each pixel in the triangulation, which is ~50x slower.
    """
    h, w = shape
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    lo = np.minimum(np.minimum(a, b), c).astype(np.int64)
    hi = np.maximum(np.maximum(a, b), c).astype(np.int64)
    bw, bh = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    counts = bw * bh
    tid = np.repeat(np.arange(len(tris)), counts)
    off = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    px = lo[tid, 0] + off % bw[tid]
    py = lo[tid, 1] + off // bw[tid]
    A, B, C = a[tid], b[tid], c[tid]
    den = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    l1 = ((B[:, 1] - C[:, 1]) * (px - C[:, 0]) + (C[:, 0] - B[:, 0]) * (py - C[:, 1])) / den
    l2 = ((C[:, 1] - A[:, 1]) * (px - C[:, 0]) + (A[:, 0] - C[:, 0]) * (py - C[:, 1])) / den
    l3 = 1.0 - l1 - l2
    inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
    zt = zv[tris][tid]
    out = np.full(shape, np.nan)
    out[py[inside], px[inside]] = (l1 * zt[:, 0] + l2 * zt[:, 1] + l3 * zt[:, 2])[inside]
    return out


def heightfield_tin(z: np.ndarray, max_error: float,
                seed_step: int = 8, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive triangulation of a heightfield within ``max_error`` at every pixel.

    Unlike surface decimation (:func:`decimate_to_tolerance`, sampled Hausdorff),
    the bound is exact and every vertex stays on a pixel centre.

    Start from every border pixel plus a lattice every ``seed_step`` pixels (about
    the source DEM's own spacing, where its information actually is), triangulate,
    then add every pixel that is a local peak of the remaining error and exceeds
    ``max_error``; repeat until no pixel does. Border pixels are all kept, so the
    model edge (and its side walls) is exact.
    Returns (pixel indices into z.ravel(), triangles over those indices).
    """
    h, w = z.shape
    ii, jj = np.divmod(np.arange(h * w), w)
    zf = z.ravel()
    chosen = (ii == 0) | (ii == h - 1) | (jj == 0) | (jj == w - 1)
    chosen |= (ii % seed_step == 0) & (jj % seed_step == 0)
    for _ in range(max_iter):
        idx = np.flatnonzero(chosen)
        xy = np.column_stack([jj[idx], ii[idx]])
        tris = _triangulate_pixels(xy)
        err = np.abs(_rasterize_tin(xy, zf[idx], tris, (h, w)) - z)
        err[~np.isfinite(err)] = 0.0
        if err.max() <= max_error:
            return idx, idx[tris]
        chosen |= ((err > max_error) & (err >= maximum_filter(err, size=3))).ravel()
    logger.warning("heightfield_tin stopped at %d iterations above %.3f mm", max_iter, max_error)
    idx = np.flatnonzero(chosen)
    return idx, idx[_triangulate_pixels(np.column_stack([jj[idx], ii[idx]]))]


def _symmetric_hausdorff(mesh_a, mesh_b, n_samples: int = 20000) -> float:
    """Symmetric Hausdorff distance (max of the two directed surface distances).

    Sampled on each surface and measured to the other with trimesh's nearest-
    surface query — robust and version-independent (pymeshlab's own Hausdorff
    sampling is finicky), in the mesh's native units.
    """
    def _directed(src, dst) -> float:
        try:
            pts = src.sample(n_samples)
        except Exception:
            pts = np.asarray(src.vertices)
        if len(pts) == 0:
            return 0.0
        _, dist, _ = dst.nearest.on_surface(pts)
        return float(np.max(dist)) if len(dist) else 0.0

    return max(_directed(mesh_a, mesh_b), _directed(mesh_b, mesh_a))


def decimate_to_tolerance(
    mesh,
    deviation_tol: float,
    min_ratio: float = 0.02,
    n_iter: int = 7,
    n_samples: int = 20000,
):
    """Most aggressive quadric decimation whose symmetric Hausdorff to `mesh`
    stays ≤ `deviation_tol` (mesh units).

    Binary-searches the kept-face ratio in [min_ratio, 1.0]: smaller ratio = more
    decimation.  Returns ``(decimated_mesh, kept_ratio, achieved_hausdorff)``.
    """
    from ..stl2numpy.reduction import decimate_trimesh

    f0 = len(mesh.faces)
    # lo = most aggressive (smallest ratio), hi = safest (no decimation).
    lo, hi = float(min_ratio), 1.0
    best_mesh, best_ratio, best_h = mesh, 1.0, 0.0

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        cand = decimate_trimesh(mesh, int(mid * f0), preserve=True)
        h = _symmetric_hausdorff(mesh, cand, n_samples=n_samples)
        if h <= deviation_tol:
            # within budget — accept and push for MORE decimation (lower ratio)
            best_mesh, best_ratio, best_h = cand, len(cand.faces) / f0, h
            hi = mid
        else:
            # exceeded budget — back off (keep more faces)
            lo = mid
    logger.info("decimate_to_tolerance: %d -> %d faces (%.1f%%), hausdorff=%.3f (tol %.3f)",
                f0, len(best_mesh.faces), 100.0 * best_ratio, best_h, deviation_tol)
    return best_mesh, best_ratio, best_h
