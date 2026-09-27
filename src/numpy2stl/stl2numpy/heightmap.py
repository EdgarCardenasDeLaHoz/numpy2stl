"""mesh_to_heightmap: convert a 3D mesh to a 2D elevation array.

The one mesh → heightmap implementation in numpy2stl / map2stl:

- ``method="bin"``      sample the surface, bin the samples per cell (fast, any mesh)
- ``method="raycast"``  one vertical ray per cell centre (exact top surface; needs
                        trimesh's ray backend — rtree or embree)

Row 0 is the mesh's max-y edge by default (``row0="north"``, the image convention
the project uses for rasters); ``row0="south"`` puts the min-y edge first, which is
what the registration code and the building simplifier work in (they pass it).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from .._paths import CACHE_ROOT
from ..io.readers import load_trimesh

logger = logging.getLogger(__name__)

_MAX_RESOLUTION = 1000
_OVERSAMPLING = 32  # sample this many points per output cell (16 causes 0.01% empty; 32 → near-zero)

# Cache directory for computed heightmaps (the 3MF/STL load + sampling is slow).
_STL_CACHE_DIR = CACHE_ROOT / "stl_cache"

_METHODS = ("bin", "raycast")
_ROW0 = ("south", "north")


def _stl_cache_path(file_path: str, resolution, projection, z_axis, allow_large,
                    isotropic=False, extra: str = "") -> Path:
    p = Path(file_path)
    try:
        mtime = int(os.path.getmtime(file_path))
    except OSError:
        mtime = 0
    key = (f"{p.resolve()}|{mtime}|r{resolution}|{projection}|z{z_axis}"
           f"|a{int(allow_large)}|i{int(isotropic)}{extra}")
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _STL_CACHE_DIR / f"stl_{h}.npz"


def mesh_to_heightmap(
    mesh_or_path,
    resolution: int | tuple[int, int] | None = None,
    projection: str = "max",
    z_axis: int = 2,
    allow_large: bool = False,
    cache: bool = True,
    isotropic: bool = False,
    *,
    method: str = "bin",
    cell_size: float | tuple[float, float] | None = None,
    row0: str = "north",
    oversampling: int = _OVERSAMPLING,
) -> dict:
    """
    Convert a 3D mesh (file or ``trimesh.Trimesh``) to a 2D heightmap (elevation) array.

    Parameters
    ----------
    mesh_or_path : str, PathLike or trimesh.Trimesh
        Path to an STL, OBJ, or 3MF file, or an in-memory mesh (never cached).
    resolution : int or (rows, cols) or None
        Output grid size.  None = auto-detect from mesh density, capped at
        1000×1000.  Pass a single int for a square grid.
    projection : {'max', 'min', 'mean'}
        How to resolve cells hit by multiple z-values (overhangs, caves).
        'max' (default) returns the highest surface.  For ``method='raycast'``
        'mean' averages every surface a cell's ray crosses.
    z_axis : int
        Which mesh axis (0=X, 1=Y, 2=Z) represents elevation.  Default 2.
    allow_large : bool
        If True, skip the 1000×1000 safety cap.
    cache : bool
        Cache the result on disk (file inputs only).
    isotropic : bool
        If True and `resolution` is a single int R, render with **square pixels**:
        the longer horizontal axis gets R bins and the shorter axis gets
        proportionally fewer, then the result is centre-padded with NaN to an
        R×R canvas.  This keeps a world-square footprint square in the image
        (no aspect stretch) — required for registration against an isotropic
        OSM raster.  Ignored when resolution is a (rows, cols) tuple.
    method : {'bin', 'raycast'}
        'bin' samples the surface (``oversampling`` points per cell plus the
        vertices) and bins them; 'raycast' casts one vertical ray through each
        cell centre, so it never leaves an empty cell over the mesh and reads
        the exact surface there.
    cell_size : float or (x_size, y_size), optional
        Cell size in mesh units, instead of ``resolution``: the grid is
        ``round(extent / cell_size)`` cells per axis (the returned ``cell_size``
        is the exact extent / count).
    row0 : {'north', 'south'}
        'north' (default): image convention, row 0 is the mesh's max-y edge.
        'south': row 0 is the min-y edge, exactly ``np.flipud`` of 'north'.
    oversampling : int
        Surface samples per output cell for ``method='bin'``.

    Returns
    -------
    dict with keys:
        'heightmap' : ndarray, shape (rows, cols), float64, NaN for empty cells
        'bounds'    : {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        'resolution': (rows, cols)
        'cell_size' : (x_size, y_size) in mesh units
        'projection': projection used
        'method'    : method used
        'row0'      : row-0 convention of 'heightmap'
    """
    try:
        import trimesh  # noqa: F401
    except ImportError as err:
        raise ImportError("trimesh is required. Install with: pip install trimesh") from err

    if projection not in ("max", "min", "mean"):
        raise ValueError(f"projection must be 'max', 'min', or 'mean', got {projection!r}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if row0 not in _ROW0:
        raise ValueError(f"row0 must be one of {_ROW0}, got {row0!r}")
    if cell_size is not None and resolution is not None:
        raise ValueError("pass resolution or cell_size, not both")

    is_path = isinstance(mesh_or_path, (str, os.PathLike))
    cache = cache and is_path
    if cache:
        # Cache check — the 3MF/STL load + surface sampling is the slow step.  The
        # key only grows for options other than the original ones (bin, south), so
        # existing cache files stay valid.
        extra = ""
        if (method, cell_size, row0, oversampling) != ("bin", None, "south", _OVERSAMPLING):
            extra = f"|m{method}|c{cell_size}|{row0}|o{oversampling}"
        cache_path = _stl_cache_path(str(mesh_or_path), resolution, projection, z_axis,
                                     allow_large, isotropic=isotropic, extra=extra)
        if cache_path.exists():
            logger.info("Loading STL heightmap from cache: %s", cache_path.name)
            d = np.load(cache_path, allow_pickle=True)
            return {
                "heightmap": d["heightmap"],
                "bounds": d["bounds"].item(),
                "resolution": tuple(d["resolution"]),
                "cell_size": tuple(d["cell_size"]),
                "projection": str(d["projection"]),
                "method": method,
                "row0": row0,
            }

    mesh = load_trimesh(str(mesh_or_path)) if is_path else mesh_or_path

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh {mesh_or_path!r} has no faces." if is_path
                         else "Mesh has no faces.")

    # Determine horizontal axes
    h_axes = [i for i in range(3) if i != z_axis]
    bounds = mesh.bounds  # shape (2, 3): [min, max]

    x_min, x_max = float(bounds[0, h_axes[0]]), float(bounds[1, h_axes[0]])
    y_min, y_max = float(bounds[0, h_axes[1]]), float(bounds[1, h_axes[1]])
    z_min, z_max = float(bounds[0, z_axis]), float(bounds[1, z_axis])

    if x_min == x_max or y_min == y_max:
        raise ValueError("Mesh has zero extent in one or both horizontal dimensions.")

    # Resolve output resolution.
    # `res` is the final (possibly padded) output shape; `render_res` is the
    # aspect-correct shape we actually render into.  They differ only for isotropic.
    if cell_size is not None:
        cs_x, cs_y = (cell_size, cell_size) if np.ndim(cell_size) == 0 else cell_size
        resolution = (max(1, int(round((y_max - y_min) / float(cs_y)))),
                      max(1, int(round((x_max - x_min) / float(cs_x)))))
    res = _resolve_resolution(mesh, resolution, h_axes, allow_large)
    pad_to_square = False
    render_res = res
    if isotropic and isinstance(resolution, (int, float)):
        R = int(res[0])  # square cap already applied by _resolve_resolution
        x_ext = x_max - x_min
        y_ext = y_max - y_min
        if x_ext >= y_ext:
            n_cols_r = R
            n_rows_r = max(1, int(round(R * y_ext / x_ext)))
        else:
            n_rows_r = R
            n_cols_r = max(1, int(round(R * x_ext / y_ext)))
        render_res = (n_rows_r, n_cols_r)
        pad_to_square = (render_res != (R, R))
        logger.debug("isotropic render: %s padded to (%d,%d)", render_res, R, R)
    logger.debug("heightmap resolution: %s", res)

    n_rows, n_cols = render_res  # rows=y-cells, cols=x-cells; row 0 = y_min
    extent = (x_min, x_max, y_min, y_max)
    if method == "raycast":
        heightmap = _raycast_grid(mesh, h_axes, z_axis, extent, z_min, z_max,
                                  n_rows, n_cols, projection)
    else:
        heightmap = _bin_grid(mesh, h_axes, z_axis, extent, n_rows, n_cols, projection,
                              oversampling)

    # Cell size (model units per pixel) of the rendered grid.  For isotropic
    # renders dx == dy by construction (square pixels).
    cell_x = (x_max - x_min) / n_cols
    cell_y = (y_max - y_min) / n_rows

    # Centre-pad the shorter axis with NaN so the output is square (R×R) while
    # the model keeps its true aspect (square pixels, no stretch).
    if pad_to_square:
        R = int(res[0])
        full = np.full((R, R), np.nan, dtype=np.float64)
        r0 = (R - n_rows) // 2
        c0 = (R - n_cols) // 2
        full[r0:r0 + n_rows, c0:c0 + n_cols] = heightmap
        heightmap = full

    if row0 == "north":
        heightmap = np.ascontiguousarray(np.flipud(heightmap))

    result = {
        "heightmap": heightmap,
        "bounds": {
            "x": (x_min, x_max),
            "y": (y_min, y_max),
            "z": (z_min, z_max),
        },
        "resolution": (int(heightmap.shape[0]), int(heightmap.shape[1])),
        "cell_size": (cell_x, cell_y),
        "projection": projection,
        "method": method,
        "row0": row0,
    }

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            heightmap=result["heightmap"],
            bounds=np.array(result["bounds"], dtype=object),
            resolution=np.array(result["resolution"]),
            cell_size=np.array(result["cell_size"]),
            projection=np.array(result["projection"]),
        )
        logger.info("Cached STL heightmap: %s", cache_path.name)

    return result


def _bin_grid(mesh, h_axes, z_axis, extent, n_rows, n_cols, projection, oversampling):
    """Sampled-surface binning → (n_rows, n_cols) float64, row 0 = min y."""
    try:
        from scipy.stats import binned_statistic_2d
    except ImportError as err:
        raise ImportError("scipy is required. Install with: pip install scipy") from err
    import trimesh

    x_min, x_max, y_min, y_max = extent
    # Sample points from the mesh surface for good cell coverage.
    # More points = fewer NaN gaps for coarse meshes.
    n_samples = max(n_rows * n_cols * oversampling, len(mesh.vertices))
    try:
        sampled, _ = trimesh.sample.sample_surface(mesh, n_samples)
    except Exception:
        sampled = np.empty((0, 3))

    # Combine sampled points with actual vertices
    all_pts = np.vstack([mesh.vertices, sampled]) if len(sampled) else mesh.vertices

    # binned_statistic_2d(x, y, bins=[nx, ny]) → shape (nx, ny); .T → (rows, cols)
    heightmap, _, _, _ = binned_statistic_2d(
        all_pts[:, h_axes[0]], all_pts[:, h_axes[1]], all_pts[:, z_axis],
        statistic=projection,  # 'max' | 'min' | 'mean' — scipy supports all three
        bins=[n_cols, n_rows],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    return heightmap.T.astype(np.float64)


def _raycast_grid(mesh, h_axes, z_axis, extent, z_min, z_max, n_rows, n_cols, projection):
    """One downward ray per cell centre → (n_rows, n_cols) float64, row 0 = min y."""
    x_min, x_max, y_min, y_max = extent
    xs = x_min + (np.arange(n_cols) + 0.5) * ((x_max - x_min) / n_cols)
    ys = y_min + (np.arange(n_rows) + 0.5) * ((y_max - y_min) / n_rows)
    xx, yy = np.meshgrid(xs, ys)
    n_rays = n_rows * n_cols
    origins = np.empty((n_rays, 3), dtype=np.float64)
    origins[:, h_axes[0]] = xx.ravel()
    origins[:, h_axes[1]] = yy.ravel()
    origins[:, z_axis] = z_max + max(1.0, z_max - z_min)   # start above the mesh
    dirs = np.zeros((n_rays, 3), dtype=np.float64)
    dirs[:, z_axis] = -1.0

    logger.debug("Ray-casting %d rays against %d faces", n_rays, len(mesh.faces))
    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=dirs, multiple_hits=True)
    z = np.asarray(locations, dtype=np.float64).reshape(-1, 3)[:, z_axis]
    index_ray = np.asarray(index_ray, dtype=np.intp)

    # np.maximum.at cannot start from NaN (nan wins every comparison): start from ∓inf.
    if projection == "mean":
        counts = np.bincount(index_ray, minlength=n_rays)
        sums = np.bincount(index_ray, weights=z, minlength=n_rays)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    else:
        fill = -np.inf if projection == "max" else np.inf
        out = np.full(n_rays, fill, dtype=np.float64)
        (np.maximum if projection == "max" else np.minimum).at(out, index_ray, z)
        out[np.isinf(out)] = np.nan
    return out.reshape(n_rows, n_cols)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_resolution(
    mesh,
    resolution: int | tuple[int, int] | None,
    h_axes: list[int],
    allow_large: bool,
) -> tuple[int, int]:
    if resolution is None:
        res = _auto_resolution(mesh, h_axes)
    elif isinstance(resolution, (int, float)):
        res = (int(resolution), int(resolution))
    else:
        res = (int(resolution[0]), int(resolution[1]))

    if not allow_large:
        res = (min(res[0], _MAX_RESOLUTION), min(res[1], _MAX_RESOLUTION))

    return res


def _auto_resolution(mesh, h_axes: list[int]) -> tuple[int, int]:
    """Estimate grid resolution from mesh face density, capped at MAX_RESOLUTION."""
    bounds = mesh.bounds
    x_size = bounds[1, h_axes[0]] - bounds[0, h_axes[0]]
    y_size = bounds[1, h_axes[1]] - bounds[0, h_axes[1]]
    xy_area = x_size * y_size

    if xy_area <= 0:
        return (_MAX_RESOLUTION, _MAX_RESOLUTION)

    face_density = len(mesh.faces) / xy_area  # faces per unit²
    base = int(np.sqrt(face_density) * 10)
    base = max(base, 32)  # floor

    aspect = x_size / y_size if y_size > 0 else 1.0
    if aspect >= 1:
        nx = min(base, _MAX_RESOLUTION)
        ny = min(int(base / aspect), _MAX_RESOLUTION)
    else:
        ny = min(base, _MAX_RESOLUTION)
        nx = min(int(base * aspect), _MAX_RESOLUTION)

    return (max(nx, 1), max(ny, 1))
