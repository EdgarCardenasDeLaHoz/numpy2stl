"""buildings — a building table read off a surface heightmap and its ground.

F-STL2NUMPY step "building table" (map2stl/docs/plans/active/F-STL2NUMPY-decompose.md).
Input: the surface (DSM) and the ground (DTM, ``numpy2stl.raster.estimate_dtm``) on one grid,
in metres. Output: one row per building - footprint polygon, area, base, height, top, roof
shape - plus the label raster the rows were measured on.

Buildings are the cells standing more than ``min_height_m`` above the ground, split into
separate buildings wherever a wall separates two roofs (a low and a high building) - the
same wall test as ``raster.ground_mask_steps`` (``raster.terrain.neighbour_edges``). Touching
buildings of one height stay one row: a print model merges them too.
"""
from __future__ import annotations

import numpy as np

__all__ = ["building_table", "roof_shape"]

#: Roof shape from one least-squares plane over the roof cells: flat below this slope (deg).
FLAT_ROOF_DEG = 5.0
#: A sloped roof is one plane (shed) when the plane's RMS residual stays under this (m).
PLANE_RMS_M = 0.5


def roof_shape(xs_m: np.ndarray, ys_m: np.ndarray, zs_m: np.ndarray) -> dict:
    """``{"shape": flat | sloped | complex, "slope_deg", "rms_m"}`` from one plane fit."""
    if len(zs_m) < 3:
        return {"shape": "flat", "slope_deg": 0.0, "rms_m": 0.0}
    a = np.column_stack([xs_m, ys_m, np.ones(len(zs_m))])
    coef, *_ = np.linalg.lstsq(a, zs_m, rcond=None)
    rms = float(np.sqrt(np.mean((zs_m - a @ coef) ** 2)))
    slope = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    if rms > PLANE_RMS_M:
        shape = "complex"
    elif slope < FLAT_ROOF_DEG:
        shape = "flat"
    else:
        shape = "sloped"
    return {"shape": shape, "slope_deg": round(slope, 1), "rms_m": round(rms, 2)}


def building_table(dsm: np.ndarray, dtm: np.ndarray, cell_size_m: float, *,
                   min_height_m: float = 2.5, min_area_m2: float = 20.0,
                   jump_m: float | None = None, simplify_frac: float = 0.01) -> dict:
    """Buildings of a surface over its ground, as a table.

    Returns ``{"buildings": [row, ...], "labels": int32 grid (0 = none, k = row k-1's id)}``.
    Each row: ``id``, ``polygon`` (list of [x, y] in cells: x = column, y = row),
    ``area_m2``, ``base_m`` (median ground under it), ``height_m`` (90th percentile of each
    cell's height over the ground under it), ``top_m`` (90th percentile of the surface),
    ``roof`` (:func:`roof_shape`), ``cells``.
    *jump_m* (default ``raster.terrain.WALL_JUMP_M``) splits buildings at walls between
    roofs (``raster.terrain.neighbour_edges``; a steep roof is not split).
    """
    import cv2
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from ..raster.terrain import WALL_JUMP_M, neighbour_edges

    dsm = np.asarray(dsm, dtype=np.float64)
    dtm = np.asarray(dtm, dtype=np.float64)
    ndsm = dsm - dtm
    built = np.isfinite(ndsm) & (ndsm > min_height_m)
    h, w = dsm.shape

    # Built cells joined to their 4-neighbours unless a wall separates them.
    ea, eb, _, wall = neighbour_edges(dsm, WALL_JUMP_M if jump_m is None else jump_m)
    keep = ~wall & built.ravel()[ea] & built.ravel()[eb]
    ea, eb = ea[keep], eb[keep]
    graph = coo_matrix((np.ones(len(ea), np.int8), (ea, eb)), shape=(h * w, h * w))
    _, comp = connected_components(graph, directed=False)
    comp = np.where(built.ravel(), comp, -1).reshape(h, w)

    ids, inverse, counts = np.unique(comp[built], return_inverse=True, return_counts=True)
    min_cells = max(1, int(np.ceil(min_area_m2 / cell_size_m ** 2)))
    keep_ids = counts >= min_cells
    labels = np.zeros((h, w), dtype=np.int32)
    new_id = np.zeros(len(ids), dtype=np.int32)
    new_id[keep_ids] = np.arange(1, int(keep_ids.sum()) + 1)
    labels[built] = new_id[inverse]

    rows = []
    objs = ndimage.find_objects(labels)
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sub = labels[sl] == k
        r0, c0 = sl[0].start, sl[1].start
        rr, cc = np.nonzero(sub)
        z_top = dsm[sl][sub]
        z_base = dtm[sl][sub]
        base = float(np.nanmedian(z_base))
        # Per cell against the ground right under it: on a slope the median ground is
        # below the uphill cells and above the downhill ones.
        height = float(np.nanpercentile(z_top - z_base, 90))
        top = float(np.nanpercentile(z_top, 90))
        contours, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        outline = max(contours, key=cv2.contourArea)
        eps = simplify_frac * cv2.arcLength(outline, True)
        poly = cv2.approxPolyDP(outline, eps, True).reshape(-1, 2) + [c0, r0]
        rows.append({
            "id": k,
            "polygon": poly.tolist(),
            "cells": int(sub.sum()),
            "area_m2": round(float(sub.sum()) * cell_size_m ** 2, 1),
            "base_m": round(base, 2),
            "height_m": round(height, 2),
            "top_m": round(top, 2),
            "roof": roof_shape((cc + c0) * cell_size_m, (rr + r0) * cell_size_m, z_top),
        })
    return {"buildings": rows, "labels": labels}
