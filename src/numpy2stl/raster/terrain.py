"""terrain — the ground (DTM) under a surface heightmap (DSM).

Layer: ``raster`` (numpy / scipy / cv2 only; no geo). F-STL2NUMPY step "DTM"
(map2stl/docs/plans/active/F-STL2NUMPY-decompose.md).

- ``ground_mask_steps``  which cells are bare ground: smooth regions split at wall-height
                       steps; a region that mostly steps down to its neighbours is raised
                       (a roof), the rest is ground
- ``ground_mask_pmf``  the same from a progressive morphological filter
- ``estimate_dtm``     the ground everywhere (default ground: ``ground_mask_steps``): ground
                       cells as they are, interpolated under
                       buildings from a block-median grid of the ground cells
- ``estimate_terrain`` / ``terrain_to_grid`` / ``grid_to_terrain``  that coarse grid
                       (moved from map2stl/tools/align_tool/plate_vectors.py, which re-exports
                       them)

Why a progressive filter: a single opening wider than the largest building
(``segment.terrain_residual``) flattens hills - it treats a 30 m rise across 80 m as a
building and cuts hillside buildings into the ground (the 2026-05 terrain segmentation
finding; a frequency detrend failed the same way). The progressive filter (Zhang et al.
2003, "A progressive morphological filter for removing nonground measurements from
airborne LIDAR data") opens with growing windows and lets the height threshold grow with
the window by the terrain slope, so ground that rises steadily stays ground while a
structure that steps up from it does not.
"""
from __future__ import annotations

import warnings

import numpy as np

# Terrain samples this far apart, in metres.  A hillside bends over hundreds of metres, so
# sampling it every twenty loses almost nothing while cutting the stored grid by two orders of
# magnitude at the resolutions these plates use.
DEM_STEP_M = 20.0


def terrain_to_grid(terrain: np.ndarray, *, cell_size_m: float,
                    step_m: float = DEM_STEP_M) -> dict:
    """A terrain raster as a coarse grid that can be expanded back.

    NaN outside the plate is a problem for any resampling, because a coarse cell straddling the
    edge would average real ground with nothing.  The surface is therefore filled outward from
    its own edge before sampling -- the value carried out is meaningless but it is only ever read
    back in places the plate does not cover.
    """
    import cv2

    terrain = np.asarray(terrain, dtype=np.float64)
    valid = np.isfinite(terrain)
    if not valid.any():
        return {"values": [], "shape": list(terrain.shape), "grid": [0, 0],
                "step_m": float(step_m)}

    # Nearest-neighbour fill: every empty cell takes the value of the closest real one. OpenCV
    # labels each cell with the index of its nearest zero-distance pixel in one pass, so the fill
    # costs a distance transform rather than an iteration. A coarse cell straddling the plate
    # edge then averages ground with more ground instead of ground with nothing.
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.float64)
    lookup[labels[valid]] = terrain[valid]
    filled = lookup[labels]

    step_px = max(int(round(step_m / cell_size_m)), 1)
    rows = max(int(np.ceil(terrain.shape[0] / step_px)) + 1, 2)
    cols = max(int(np.ceil(terrain.shape[1] / step_px)) + 1, 2)
    coarse = cv2.resize(filled, (cols, rows), interpolation=cv2.INTER_AREA)
    return {"values": np.round(coarse, 2).ravel().tolist(),
            "shape": [int(terrain.shape[0]), int(terrain.shape[1])],
            "grid": [rows, cols], "step_m": float(step_m)}


def estimate_terrain(rendered: np.ndarray, built: np.ndarray, *, cell_size_m: float,
                     step_m: float = DEM_STEP_M, heights: np.ndarray | None = None) -> dict:
    """A terrain grid read off the render itself, at the cells no building covers.

    The obvious ground surface is the one the segmentation already produces -- the render minus
    its own white top-hat residual -- but that is a morphological envelope rather than a
    landscape.  An opening knocks the peaks off, which leaves a step wherever a structure begins
    and ends, and steps are the one thing a coarse grid cannot carry.

    Sampling the bare cells instead avoids the problem rather than fighting it.  Nothing has been
    subtracted from those cells, so the surface they describe is smooth; the cells under
    buildings are simply absent, and a gap between known ground is what interpolation is for.
    On the Alhambra this halves the error at every spacing worth using -- twenty-metre samples
    match what the top-hat ground needed five-metre samples to reach, with a sixteenth as many.

    A block median rather than a mean, because a block that is mostly roof still has a few open
    cells around the edges and the median reports those rather than averaging them with the roof.

    ``heights`` closes the remaining gap.  A built cell is not silent about its ground either --
    the segmented height is measured from that ground upward, so ``rendered - height`` reads it,
    weakly, wherever a building stands.  Bare cells still win where both exist; a block entirely
    under roof now has samples of its own rather than a neighbouring block's value.
    """
    import cv2

    rendered = np.asarray(rendered, dtype=np.float64)
    plate = np.isfinite(rendered)
    built = np.asarray(built, dtype=bool)
    bare = np.where(plate & ~built, rendered, np.nan)
    if heights is not None:
        base = rendered - np.asarray(heights, dtype=np.float64)
        bare = np.where(np.isfinite(bare), bare, np.where(plate & built, base, np.nan))
    if not np.isfinite(bare).any():
        bare = np.where(plate, rendered, np.nan)

    step_px = max(int(round(step_m / cell_size_m)), 1)
    rows = int(np.ceil(bare.shape[0] / step_px))
    cols = int(np.ceil(bare.shape[1] / step_px))
    pad = np.full((rows * step_px, cols * step_px), np.nan)
    pad[:bare.shape[0], :bare.shape[1]] = bare
    blocks = pad.reshape(rows, step_px, cols, step_px).transpose(0, 2, 1, 3)
    with warnings.catch_warnings():
        # Blocks wholly off the plate are empty by construction; they are filled just below.
        warnings.simplefilter("ignore", RuntimeWarning)
        coarse = np.nanmedian(blocks.reshape(rows, cols, -1), axis=2)

    # Blocks that saw no bare ground at all -- a courtyard-less block entirely under roof, or a
    # block off the plate -- take the nearest block that did.
    empty = ~np.isfinite(coarse)
    if empty.any() and not empty.all():
        _, labels = cv2.distanceTransformWithLabels(empty.astype(np.uint8), cv2.DIST_L2, 3,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        lookup = np.zeros(int(labels.max()) + 1, dtype=np.float64)
        lookup[labels[~empty]] = coarse[~empty]
        coarse = lookup[labels]
    coarse = np.nan_to_num(coarse, nan=0.0)

    return {"values": np.round(coarse, 2).ravel().tolist(),
            "shape": [int(rendered.shape[0]), int(rendered.shape[1])],
            "grid": [int(rows), int(cols)], "step_m": float(step_m)}


def grid_to_terrain(dem: dict, shape=None) -> np.ndarray:
    """The coarse grid expanded back to a full raster, bilinearly."""
    import cv2

    rows, cols = dem["grid"]
    if rows == 0 or cols == 0:
        return np.full(tuple(shape or dem["shape"]), np.nan, dtype=np.float32)
    coarse = np.asarray(dem["values"], dtype=np.float32).reshape(rows, cols)
    out_rows, out_cols = tuple(shape or dem["shape"])
    return cv2.resize(coarse, (out_cols, out_rows), interpolation=cv2.INTER_LINEAR)


def ground_mask_steps(dsm: np.ndarray, cell_size_m: float, *, jump_m: float | None = None,
                      max_slope: float = 0.7, down_frac: float = 0.3,
                      max_roof_m2: float = 40_000.0) -> np.ndarray:
    """Bare-ground cells of a surface heightmap (metres, NaN off the model), by walls.

    Neighbouring cells (4-connected) belong to one smooth region unless their heights
    differ by more than *jump_m* (default ``max(1.0, max_slope * cell_size_m)``: steeper
    than *max_slope* is a wall, not terrain). A region is raised - a roof - when at least
    *down_frac* of the wall steps on its boundary go down out of it and it covers at most
    *max_roof_m2* (a larger raised region is a terrace or plateau of the terrain). All other
    regions are ground: the street network, courtyards, terraces.

    Width does not matter, unlike an opening: a 100 m block is as raised as a 10 m one.
    The progressive filter (:func:`ground_mask_pmf`) needs a window wider than the block,
    where its slope allowance (``slope * window``) already exceeds a storey or three.
    A low roof enclosed only by taller ones reads as ground (it steps up everywhere).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    z = np.asarray(dsm, dtype=np.float64)
    valid = np.isfinite(z)
    h, w = z.shape
    jump = jump_m if jump_m is not None else max(1.0, max_slope * cell_size_m)
    idx = np.arange(h * w).reshape(h, w)
    # Edges between 4-neighbours that are both on the model.
    pairs = []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        za, zb = z.ravel()[a], z.ravel()[b]
        ok = np.isfinite(za) & np.isfinite(zb)
        pairs.append((a[ok], b[ok], za[ok] - zb[ok]))
    ea = np.concatenate([p[0] for p in pairs])
    eb = np.concatenate([p[1] for p in pairs])
    dz = np.concatenate([p[2] for p in pairs])
    smooth = np.abs(dz) <= jump
    n = h * w
    graph = coo_matrix((np.ones(int(smooth.sum()), np.int8), (ea[smooth], eb[smooth])),
                       shape=(n, n))
    _, labels = connected_components(graph, directed=False)

    # Wall steps on each region's boundary: "down" when the region is the higher side.
    wall = ~smooth
    la, lb, d = labels[ea[wall]], labels[eb[wall]], dz[wall]
    n_lab = int(labels.max()) + 1
    down = (np.bincount(la, weights=(d > 0), minlength=n_lab)
            + np.bincount(lb, weights=(d < 0), minlength=n_lab))
    total = np.bincount(la, minlength=n_lab) + np.bincount(lb, minlength=n_lab)
    area = np.bincount(labels, weights=valid.ravel(), minlength=n_lab) * cell_size_m ** 2
    raised = (total > 0) & (down >= down_frac * np.maximum(total, 1)) & (area <= max_roof_m2)
    return valid & ~raised[labels].reshape(h, w)


def ground_mask_pmf(dsm: np.ndarray, cell_size_m: float, *, max_window_m: float = 120.0,
                    slope: float = 0.3, dh0_m: float = 0.5, dh_max_m: float = 40.0) -> np.ndarray:
    """Bare-ground cells of a surface heightmap (metres, NaN off the model).

    Openings with square windows of 3, 5, 9, 17, ... cells up to *max_window_m*; at step k a
    cell becomes non-ground when it stands more than
    ``dh_k = min(dh0_m + slope * (w_k - w_k-1) * cell_size_m, dh_max_m)`` above the
    opened surface. *slope* is the steepest terrain (rise over run) still read as ground;
    *max_window_m* must exceed the widest building, or its middle survives as ground.
    """
    import cv2

    from .fill import fill_nan

    dsm = np.asarray(dsm, dtype=np.float64)
    valid = np.isfinite(dsm)
    if not valid.any():
        return valid
    # Off-model cells take their nearest value, so the model's edge is not a cliff.
    surface = fill_nan(dsm, method="nearest").astype(np.float32)
    nonground = np.zeros(dsm.shape, dtype=bool)
    w_prev, w, k = 1, 3, 0
    while (w - 1) * cell_size_m <= max_window_m or k == 0:
        kernel = np.ones((w, w), np.uint8)
        opened = cv2.dilate(cv2.erode(surface, kernel), kernel)
        dh = dh0_m if k == 0 else min(dh0_m + slope * (w - w_prev) * cell_size_m, dh_max_m)
        nonground |= (surface - opened) > dh
        surface = opened
        w_prev, w, k = w, 2 * w - 1, k + 1
    return valid & ~nonground


def estimate_dtm(dsm: np.ndarray, cell_size_m: float, *, ground: np.ndarray | None = None,
                 step_m: float | None = None, **steps) -> np.ndarray:
    """The ground under a surface heightmap (metres; NaN where the DSM is NaN).

    Ground cells (*ground*, default :func:`ground_mask_steps` with *steps*) keep their own height;
    every other cell is read off a block-median grid of the ground cells
    (:func:`estimate_terrain`, *step_m* default four cells but at least 5 m), expanded
    bilinearly. The result never stands above the surface.
    """
    dsm = np.asarray(dsm, dtype=np.float64)
    valid = np.isfinite(dsm)
    if ground is None:
        ground = ground_mask_steps(dsm, cell_size_m, **steps)
    step = step_m if step_m is not None else max(4 * cell_size_m, 5.0)
    grid = estimate_terrain(dsm, ~np.asarray(ground, dtype=bool), cell_size_m=cell_size_m,
                            step_m=step)
    under = grid_to_terrain(grid, dsm.shape).astype(np.float64)
    dtm = np.where(ground, dsm, np.minimum(under, dsm))
    return np.where(valid, dtm, np.nan)
