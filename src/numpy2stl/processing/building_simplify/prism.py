"""Prism decomposition — per-building flat/sloped prism LOD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


from ._io import _save_prism_models


class PrismStats(NamedTuple):
    n_buildings: int
    n_prisms: int
    mean_layers: float          # mean prism layers per building
    sloped_caps: int            # how many top caps used a fitted (slanted) plane
    hausdorff_m: float          # prism-model heightmap deviation vs original (metres)
    deviation_tol_m: float
    backend: str


def _fit_plane(xs, ys, zs):
    """Least-squares plane z = a*x + b*y + c; returns ((a,b,c), rms)."""
    A = np.column_stack([xs, ys, np.ones(len(zs))])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    rms = float(np.sqrt(np.mean((zs - A @ coef) ** 2)))
    return (float(coef[0]), float(coef[1]), float(coef[2])), rms


def prism_decompose(
    file_path: str,
    deviation_tol: float = 3.5,
    z_axis: int = 2,
    resolution: int = 512,
    m_per_unit: float = 1.0,
    max_layers: int = 8,
    min_area_px: int = 20,
    save_path: str | Path | None = None,
):
    """Decompose a city STL into a **sum of extruded prisms** (the reverse of how
    OSM footprints are rendered into a model).

    Each building becomes a wedding-cake stack of nested prisms (footprint at each
    budget-spaced height level), and the topmost roof is capped with a fitted
    plane (slanted) where it is planar within budget.  ``deviation_tol`` is in
    MESH UNITS (the caller converts a metres budget via the model scale); levels
    are spaced by it, so the prism model stays within budget of the original.

    Returns ``(vertices, faces, PrismStats, models)`` where ``models`` is a
    ``{name: (verts, faces)}`` soup (one entry per prism, like an OSM extrude
    set), saved to ``save_path`` (STL = merged, 3MF = soup) when given.
    """
    import trimesh
    from ...stl2numpy.heightmap import mesh_to_heightmap
    from ...registration.align import (building_mask, vectorize_buildings,
                                       terrain_residual)
    from ..extrusion import make_sloped_prism_solid

    # Non-isotropic render → simple pixel↔world mapping (no NaN padding).
    r = mesh_to_heightmap(file_path, resolution=resolution, projection="max",
                          z_axis=z_axis, isotropic=False, cache=False)
    hm = r["heightmap"]
    (x_min, x_max), (y_min, y_max) = r["bounds"]["x"], r["bounds"]["y"]
    cell_x, cell_y = r["cell_size"]
    cell_m = 0.5 * (abs(cell_x) + abs(cell_y)) * m_per_unit

    residual, _ = terrain_residual(hm, cell_size_m=cell_m)
    # Use the TRIANGLE threshold, not the default p50: the STL fills the whole frame
    # (base plate everywhere), so the residual median (p50) is low and ~50% of cells
    # exceed it — flooding streets/base as "buildings" and fusing the whole downtown
    # into one component.  Triangle finds the valley between the ground peak and the
    # building tail (~20% of frame), giving SEPARATED individual building footprints.
    mask = building_mask(hm, source="stl", cell_size_m=cell_m,
                         threshold_method="triangle", split_watershed=True)
    try:
        from scipy import ndimage as ndi
        labels, n_lab = ndi.label(mask)
    except Exception:
        labels, n_lab = mask.astype(int), 1

    def _px_to_world(poly_px):
        w = np.empty_like(poly_px, dtype=np.float64)
        w[:, 0] = x_min + (poly_px[:, 0] + 0.5) * cell_x   # col → world x
        w[:, 1] = y_min + (poly_px[:, 1] + 0.5) * cell_y   # row → world y
        return w

    step = float(deviation_tol)
    models: dict = {}
    n_prisms = 0
    layer_counts = []
    sloped = 0
    # Prism-model surface on the SAME grid as `residual`, for a grid-consistent
    # deviation measure (height-above-ground, so no base-plate offset).
    prism_hm = np.full_like(residual, np.nan, dtype=np.float64)

    base_polys_frac = []   # per-building footprint outlines, as [0,1] bounds-fractions
    for lid in range(1, n_lab + 1):
        comp = labels == lid
        rv = residual[comp]
        rv = rv[np.isfinite(rv)]
        if rv.size < min_area_px:
            continue
        top = float(np.percentile(rv, 99))
        # Space layers by the deviation budget so the quantization error stays ≤
        # the budget; cap the count for runaway-tall buildings (a smooth tower
        # keeps the same footprint at every level, so the extra prisms are nested
        # and the max-projected height is unaffected — only the file grows).
        if top <= step:
            levels = [0.0, top]                       # single prism
        else:
            n = max(1, int(np.ceil(top / step)))
            if n > max_layers:
                n = max_layers
            levels = list(np.linspace(0.0, top, n + 1))
        n_layers_here = 0
        for k in range(len(levels) - 1):
            z_lo, z_hi = levels[k], levels[k + 1]
            sub = comp & (residual >= z_lo)
            polys = vectorize_buildings(sub.astype(np.uint8), regularize=True,
                                        min_area_px=min_area_px)
            if k == 0:   # base layer = the building's full (separated) footprint
                for _p in polys:
                    base_polys_frac.append(_p.astype(np.float64) / float(resolution))
            is_top = (k == len(levels) - 2)
            plane = None
            if is_top:
                # Fit a plane to the original roof over the top region; slant the
                # cap if it is planar within the deviation budget.
                ys, xs = np.where(sub)
                if xs.size >= 8:
                    wx = x_min + (xs + 0.5) * cell_x
                    wy = y_min + (ys + 0.5) * cell_y
                    (a, b, c), rms = _fit_plane(wx, wy, residual[ys, xs])
                    if rms <= step:
                        plane = (a, b, c)
            # Record this layer's contribution to the prism surface (max-projection).
            cap = z_hi if plane is None else None
            sub_pix = sub if cap is not None else None
            for poly_px in polys:
                world = _px_to_world(poly_px.astype(np.float64))
                try:
                    if plane is not None:
                        v, f = make_sloped_prism_solid(world, z0=float(z_lo), plane=plane)
                        if len(f):
                            sloped += 1
                    else:
                        v, f = make_sloped_prism_solid(world, z0=float(z_lo),
                                                       plane=None, z1=float(z_hi))
                except Exception:
                    continue
                if len(f):
                    models[f"b{lid}_l{k}_{len(models)}"] = (v, f)
                    n_prisms += 1
                    n_layers_here += 1
            # Update the prism surface for this layer's footprint.
            if plane is None:
                prism_hm[sub] = z_hi
            else:
                ys, xs = np.where(sub)
                prism_hm[ys, xs] = plane[0] * (x_min + (xs + 0.5) * cell_x) + \
                                   plane[1] * (y_min + (ys + 0.5) * cell_y) + plane[2]
        if n_layers_here:
            layer_counts.append(n_layers_here)

    # Merge into one mesh for stats / return + save.
    if models:
        merged = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in models.values()])
        verts, faces = np.asarray(merged.vertices), np.asarray(merged.faces)
    else:
        merged, verts, faces = None, np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    # Deviation: prism surface vs original residual on the SAME grid (both are
    # height-above-ground, so no base-plate offset).  p99 of |prism − original|
    # over the building footprints — the honest "how well do the stacked prisms
    # approximate the massing" number.
    haus_m = float("nan")
    both = np.isfinite(prism_hm) & np.isfinite(residual) & (residual > 0)
    if both.any():
        haus_m = float(np.percentile(np.abs(prism_hm[both] - residual[both]), 99) * m_per_unit)

    stats = PrismStats(
        n_buildings=len(layer_counts), n_prisms=n_prisms,
        mean_layers=float(np.mean(layer_counts)) if layer_counts else 0.0,
        sloped_caps=sloped, hausdorff_m=haus_m,
        deviation_tol_m=float(deviation_tol * m_per_unit), backend="prism",
    )
    logger.info("prism_decompose: %d buildings → %d prisms (mean %.1f layers, %d sloped caps), "
                "deviation≈%.2f m", stats.n_buildings, stats.n_prisms, stats.mean_layers,
                stats.sloped_caps, stats.hausdorff_m)

    if save_path is not None and models:
        _save_prism_models(models, merged, save_path)
    # base_polys_frac: per-building footprint outlines as [0,1] bounds-fractions,
    # already watershed-separated + regularized — for deriving footprint LINES
    # downstream WITHOUT a (blob-prone) working-resolution re-segmentation.
    return verts, faces, stats, models, base_polys_frac
