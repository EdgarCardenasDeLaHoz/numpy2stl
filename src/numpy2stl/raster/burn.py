"""Polygon → raster ("burn") primitive.

``burn_polygons(polys, shape, transform=... | bounds=..., values, mode)`` draws
polygons (holes respected) into a grid:

- ``mode="max"``  the highest value of the polygons covering a cell
- ``mode="sum"``  overlapping polygons add
- ``mode="set"``  the last polygon drawn wins (input order)

Cells no polygon covers keep ``fill``.  ``bounds=(west, south, east, north)``
gives a north-up grid (row 0 = north, the image convention); callers that keep
row 0 = south flip the result at their boundary (``np.flipud``).

Uses rasterio when installed; otherwise a shapely cell-centre test (same result
as rasterio with ``all_touched=False``; ``all_touched=True`` is then ignored).
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rasterio.enums import MergeAlg
    from rasterio.features import rasterize as _rio_rasterize
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

_MODES = ("max", "sum", "set")


def _as_geometry(p):
    """shapely geometry from a shapely geometry, GeoJSON-like mapping or (K, 2) ring."""
    from shapely.geometry import Polygon, shape
    if hasattr(p, "geom_type"):
        return p
    if isinstance(p, dict) or hasattr(p, "__geo_interface__"):
        return shape(p)
    return Polygon(np.asarray(p, dtype=np.float64))


def _as_affine(transform, bounds, shape):
    """(a, b, c, d, e, f) of an affine x = a*col + b*row + c, y = d*col + e*row + f."""
    if (transform is None) == (bounds is None):
        raise ValueError("pass exactly one of transform= or bounds=")
    rows, cols = shape
    if bounds is not None:
        west, south, east, north = (float(v) for v in bounds)
        return ((east - west) / cols, 0.0, west, 0.0, -(north - south) / rows, north)
    return tuple(float(v) for v in tuple(transform)[:6])


def burn_polygons(
    polys: Iterable,
    shape: tuple[int, int],
    transform=None,
    bounds: Sequence[float] | None = None,
    values: float | Sequence[float] = 1.0,
    mode: str = "max",
    all_touched: bool = False,
    fill: float = 0.0,
    dtype=np.float64,
) -> np.ndarray:
    """Rasterise polygons into a ``shape`` grid.

    Parameters
    ----------
    polys : iterable of shapely (Multi)Polygons, GeoJSON-like mappings or
        (K, 2) vertex rings, in the coordinates of ``transform`` / ``bounds``.
        Interior rings (holes) are left unburnt.  Empty / None entries are skipped.
    shape : (rows, cols).
    transform : affine (``rasterio``/``affine.Affine`` or its first six
        coefficients ``a, b, c, d, e, f``) mapping (col, row) → (x, y).
    bounds : ``(west, south, east, north)`` — alternative to ``transform``;
        north-up, row 0 = north.
    values : one value for every polygon, or one per polygon.
    mode : {'max', 'sum', 'set'} — how overlapping polygons combine.
    all_touched : burn every cell a polygon touches (rasterio only), not just
        the cells whose centre it covers.
    fill : value of cells no polygon covers.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    rows, cols = int(shape[0]), int(shape[1])
    polys = list(polys)
    vals = (np.full(len(polys), float(values)) if np.ndim(values) == 0
            else np.asarray(values, dtype=np.float64))
    if len(vals) != len(polys):
        raise ValueError(f"{len(vals)} values for {len(polys)} polygons")
    pairs = [(g, float(v)) for g, v in zip((None if p is None else _as_geometry(p) for p in polys),
                                           vals, strict=True)
             if g is not None and not g.is_empty]
    out = np.full((rows, cols), fill, dtype=dtype)
    if not pairs:
        return out
    a, b, c, d, e, f = _as_affine(transform, bounds, (rows, cols))

    if mode == "max":
        pairs.sort(key=lambda gv: gv[1])     # drawn in ascending order → the max wins

    if HAS_RASTERIO:
        from affine import Affine
        aff = Affine(a, b, c, d, e, f)
        kw = dict(out_shape=(rows, cols), transform=aff, fill=0.0,
                  all_touched=all_touched, dtype=np.float64)
        burnt = _rio_rasterize(pairs, merge_alg=MergeAlg.add if mode == "sum"
                               else MergeAlg.replace, **kw)
        if fill == 0.0:              # uncovered cells are already 0: skip the coverage pass
            return burnt.astype(dtype, copy=False)
        covered = _rio_rasterize([(g, 1.0) for g, _ in pairs], **kw) > 0
    else:
        if all_touched:
            logger.debug("burn_polygons: all_touched ignored without rasterio")
        burnt, covered = _burn_shapely(pairs, (rows, cols), (a, b, c, d, e, f), mode)
    out[covered] = burnt[covered]
    return out


def _burn_shapely(pairs, shape, aff, mode):
    """Cell-centre burn with shapely (rasterio fallback)."""
    import shapely
    rows, cols = shape
    a, b, c, d, e, f = aff
    cc, rr = np.meshgrid(np.arange(cols) + 0.5, np.arange(rows) + 0.5)
    xs = a * cc + b * rr + c
    ys = d * cc + e * rr + f
    burnt = np.zeros(shape, dtype=np.float64)
    covered = np.zeros(shape, dtype=bool)
    for geom, v in pairs:
        hit = shapely.contains_xy(geom, xs, ys)
        if mode == "sum":
            burnt[hit] += v
        else:                       # 'set' and 'max' (pairs pre-sorted ascending)
            burnt[hit] = v
        covered |= hit
    return burnt, covered
