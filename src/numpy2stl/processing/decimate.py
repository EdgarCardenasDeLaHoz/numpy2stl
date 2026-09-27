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

__all__ = ["heightfield_tin", "heightfield_tin_budget", "decimate_to_tolerance"]


def _triangulate_pixels(ij: np.ndarray) -> np.ndarray:
    """Delaunay triangles over integer pixel coordinates (Shewchuk's Triangle)."""
    import triangle

    return triangle.triangulate({"vertices": ij.astype(np.float64)}, "Q")["triangles"]


def _rasterize_tin(xy: np.ndarray, zv: np.ndarray, tris: np.ndarray,
                   shape: tuple[int, int], out: np.ndarray | None = None,
                   chunk: int = 1 << 21) -> np.ndarray:
    """Linear interpolation of a TIN whose vertices lie on pixel centres, per pixel.

    Scan-converts every triangle's bounding box at once (vectorised) instead of
    locating each pixel in the triangulation, which is ~50x slower. Each
    triangle is reduced to three affine forms of (x, y) - two barycentrics and
    its plane - so a pixel costs a handful of multiply-adds; triangles are done
    in chunks of about ``chunk`` bounding-box pixels to bound memory. Only the
    pixels the given triangles cover are written into ``out`` (a new NaN grid
    by default), which lets a caller re-rasterise just the triangles that changed.
    """
    h, w = shape
    if out is None:
        out = np.full(shape, np.nan)
    if not len(tris):
        return out
    a, b, c = xy[tris[:, 0]], xy[tris[:, 1]], xy[tris[:, 2]]
    za, zb, zc = zv[tris[:, 0]], zv[tris[:, 1]], zv[tris[:, 2]]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = den != 0
    a, b, c, za, zb, zc, den = a[ok], b[ok], c[ok], za[ok], zb[ok], zc[ok], den[ok]
    # l1 = p1x*x + p1y*y + p1c, l2 likewise; z = zc + l1*(za-zc) + l2*(zb-zc).
    p1x, p1y = (b[:, 1] - c[:, 1]) / den, (c[:, 0] - b[:, 0]) / den
    p2x, p2y = (c[:, 1] - a[:, 1]) / den, (a[:, 0] - c[:, 0]) / den
    p1c = -(p1x * c[:, 0] + p1y * c[:, 1])
    p2c = -(p2x * c[:, 0] + p2y * c[:, 1])
    dza, dzb = za - zc, zb - zc
    coef = np.column_stack([p1x, p1y, p1c, p2x, p2y, p2c,
                            dza * p1x + dzb * p2x, dza * p1y + dzb * p2y,
                            zc + dza * p1c + dzb * p2c])
    lo = np.minimum(np.minimum(a, b), c).astype(np.int64)
    hi = np.maximum(np.maximum(a, b), c).astype(np.int64)
    bw, bh = hi[:, 0] - lo[:, 0] + 1, hi[:, 1] - lo[:, 1] + 1
    counts = bw * bh
    ends = np.cumsum(counts)
    start = 0
    while start < len(counts):
        base = ends[start - 1] if start else 0
        stop = max(int(np.searchsorted(ends, base + chunk, side="right")), start + 1)
        cnt = counts[start:stop]
        n = int(cnt.sum())
        rep = np.repeat(np.arange(start, stop), cnt)
        off = np.arange(n) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        bwr = bw[rep]
        px = (lo[rep, 0] + off % bwr).astype(np.float64)
        py = (lo[rep, 1] + off // bwr).astype(np.float64)
        k = coef[rep]
        l1 = k[:, 0] * px + k[:, 1] * py + k[:, 2]
        l2 = k[:, 3] * px + k[:, 4] * py + k[:, 5]
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1.0 + 1e-9)
        pxi, pyi = px[inside].astype(np.int64), py[inside].astype(np.int64)
        out[pyi, pxi] = (k[inside, 6] * px[inside] + k[inside, 7] * py[inside] + k[inside, 8])
        start = stop
    return out


def _tri_keys(tris: np.ndarray, n: int) -> np.ndarray:
    """One key per triangle over vertex ids < ``n``, independent of vertex order."""
    s = np.sort(tris, axis=1).astype(np.int64)
    if n ** 3 < 2 ** 63:
        return (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    s = np.ascontiguousarray(s)
    return s.view(np.dtype((np.void, s.dtype.itemsize * 3))).ravel()


def heightfield_tin(z: np.ndarray, max_error: float,
                seed_step: int = 8, max_iter: int = 80,
                max_vertices: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Adaptive triangulation of a heightfield within ``max_error`` at every pixel.

    Unlike surface decimation (:func:`decimate_to_tolerance`, sampled Hausdorff),
    the bound is exact and every vertex stays on a pixel centre.

    Start from every border pixel plus a lattice every ``seed_step`` pixels (about
    the source DEM's own spacing, where its information actually is), triangulate,
    then add every pixel that is a local peak of the remaining error and exceeds
    ``max_error``; repeat until no pixel does. Border pixels are all kept, so the
    model edge (and its side walls) is exact. Each pass re-rasterises only the
    triangles the new vertices changed (a Delaunay insertion re-triangulates its
    cavity and nothing else), so later passes cost a fraction of the first.

    ``max_vertices`` caps the vertex count (a preview budget): the bound is then
    raised until the mesh fits (see :func:`heightfield_tin_budget`, which also
    returns the bound reached).
    Returns (pixel indices into z.ravel(), triangles over those indices).
    """
    idx, tris, _ = heightfield_tin_budget(z, max_error, seed_step=seed_step,
                                          max_iter=max_iter, max_vertices=max_vertices)
    return idx, tris


def heightfield_tin_budget(z: np.ndarray, max_error: float, seed_step: int = 8,
                           max_iter: int = 80, max_vertices: int | None = None,
                           ladder: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    """:func:`heightfield_tin` plus the largest remaining error (the bound reached).

    With ``max_vertices`` the bound is tightened in stages - ``ladder``^k x
    ``max_error`` down to ``max_error``, each stage refining the previous one's vertices - and
    the last stage that fits the budget is returned. That is "raise the
    tolerance until the mesh fits" at the cost of about one refinement.
    """
    h, w = z.shape
    ii, jj = np.divmod(np.arange(h * w), w)
    zf = z.ravel()
    chosen = (ii == 0) | (ii == h - 1) | (jj == 0) | (jj == w - 1)
    chosen |= (ii % seed_step == 0) & (jj % seed_step == 0)
    interp = np.full((h, w), np.nan)
    old_keys = None
    tau = None
    best = None
    for _ in range(max_iter):
        idx = np.flatnonzero(chosen)
        if max_vertices is not None and len(idx) > max_vertices and best is not None:
            return best
        xy = np.column_stack([jj[idx], ii[idx]])
        tris = _triangulate_pixels(xy)
        g = idx[tris]
        keys = _tri_keys(g, h * w)
        changed = (tris if old_keys is None
                   else tris[~np.isin(keys, old_keys, assume_unique=True)])
        old_keys = keys
        _rasterize_tin(xy.astype(np.float64), zf[idx], changed, (h, w), out=interp)
        err = np.abs(interp - z)
        err[~np.isfinite(err)] = 0.0
        err_max = float(err.max())
        if max_vertices is None:
            if err_max <= max_error:
                return idx, g, err_max
            thresh = max_error
        else:
            if tau is None:   # first pass: the ladder starts just above err_max / ladder
                steps = int(np.ceil(np.log(err_max / max_error) / np.log(ladder))) - 1
                tau = max_error * ladder ** max(0, steps)
            while err_max <= tau:
                best = (idx, g, err_max)
                if tau <= max_error:
                    return best
                tau = max(max_error, tau / ladder)
            thresh = tau
        chosen |= ((err > thresh) & (err >= maximum_filter(err, size=3))).ravel()
    if max_vertices is None:
        logger.warning("heightfield_tin stopped at %d iterations above %.3f mm",
                       max_iter, max_error)
    elif best is not None:
        return best
    idx = np.flatnonzero(chosen)
    tris = _triangulate_pixels(np.column_stack([jj[idx], ii[idx]]))
    return idx, idx[tris], float("nan")


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
