"""mesh_to_heightmap: convert a 3D mesh to a 2D elevation array."""

from __future__ import annotations

import hashlib
import logging
import os
import warnings
from pathlib import Path

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)

_MAX_RESOLUTION = 1000
_OVERSAMPLING = 32  # sample this many points per output cell (16 causes 0.01% empty; 32 → near-zero)

# Cache directory for computed heightmaps (the 3MF/STL load + sampling is slow).
_STL_CACHE_DIR = Path(__file__).parent.parent / "registration" / "runs" / "stl_cache"


def _stl_cache_path(file_path: str, resolution, projection, z_axis, allow_large,
                    isotropic=False) -> Path:
    p = Path(file_path)
    try:
        mtime = int(os.path.getmtime(file_path))
    except OSError:
        mtime = 0
    key = (f"{p.resolve()}|{mtime}|r{resolution}|{projection}|z{z_axis}"
           f"|a{int(allow_large)}|i{int(isotropic)}")
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _STL_CACHE_DIR / f"stl_{h}.npz"


def mesh_to_heightmap(
    file_path: str,
    resolution: int | tuple[int, int] | None = None,
    projection: str = "max",
    z_axis: int = 2,
    allow_large: bool = False,
    cache: bool = True,
    isotropic: bool = False,
) -> dict:
    """
    Convert a 3D mesh file to a 2D heightmap (elevation) array.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    resolution : int or (rows, cols) or None
        Output grid size.  None = auto-detect from mesh density, capped at
        1000×1000.  Pass a single int for a square grid.
    projection : {'max', 'min', 'mean'}
        How to resolve cells hit by multiple z-values (overhangs, caves).
        'max' (default) returns the highest surface.
    z_axis : int
        Which mesh axis (0=X, 1=Y, 2=Z) represents elevation.  Default 2.
    allow_large : bool
        If True, skip the 1000×1000 safety cap.
    isotropic : bool
        If True and `resolution` is a single int R, render with **square pixels**:
        the longer horizontal axis gets R bins and the shorter axis gets
        proportionally fewer, then the result is centre-padded with NaN to an
        R×R canvas.  This keeps a world-square footprint square in the image
        (no aspect stretch) — required for registration against an isotropic
        OSM raster.  Ignored when resolution is a (rows, cols) tuple.

    Returns
    -------
    dict with keys:
        'heightmap' : ndarray, shape (rows, cols), float64, NaN for empty cells
        'bounds'    : {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        'resolution': (rows, cols)
        'cell_size' : (x_size, y_size) in mesh units
        'projection': projection used
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    try:
        from scipy.stats import binned_statistic_2d
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

    # Cache check — the 3MF/STL load + surface sampling is the slow step.
    cache_path = _stl_cache_path(file_path, resolution, projection, z_axis, allow_large,
                                 isotropic=isotropic)
    if cache and cache_path.exists():
        logger.info("Loading STL heightmap from cache: %s", cache_path.name)
        d = np.load(cache_path, allow_pickle=True)
        return {
            "heightmap": d["heightmap"],
            "bounds": d["bounds"].item(),
            "resolution": tuple(d["resolution"]),
            "cell_size": tuple(d["cell_size"]),
            "projection": str(d["projection"]),
        }

    mesh = _load_trimesh_mesh(file_path)

    if len(mesh.faces) == 0:
        raise ValueError(f"Mesh at {file_path!r} has no faces.")

    if projection not in ("max", "min", "mean"):
        raise ValueError(f"projection must be 'max', 'min', or 'mean', got {projection!r}")

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
    # aspect-correct shape we actually bin into.  They differ only for isotropic.
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

    # Sample points from the mesh surface for good cell coverage.
    # More points = fewer NaN gaps for coarse meshes.
    n_samples = max(res[0] * res[1] * _OVERSAMPLING, len(mesh.vertices))
    try:
        sampled, _ = trimesh.sample.sample_surface(mesh, n_samples)
    except Exception:
        sampled = np.empty((0, 3))

    # Combine sampled points with actual vertices
    all_pts = np.vstack([mesh.vertices, sampled]) if len(sampled) else mesh.vertices

    x_pts = all_pts[:, h_axes[0]]
    y_pts = all_pts[:, h_axes[1]]
    z_pts = all_pts[:, z_axis]

    # render_res = (rows, cols) = (n_y_bins, n_x_bins)
    # binned_statistic_2d(x, y, bins=[nx, ny]) → shape (nx, ny)
    # After .T → (ny, nx) = (rows, cols)  ✓
    n_rows, n_cols = render_res  # rows=y-bins, cols=x-bins
    stat = projection  # 'max' | 'min' | 'mean' — scipy supports all three
    heightmap, _, _, _ = binned_statistic_2d(
        x_pts, y_pts, z_pts,
        statistic=stat,
        bins=[n_cols, n_rows],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    # shape (n_cols, n_rows) → after .T: (n_rows, n_cols)
    heightmap = heightmap.T.astype(np.float64)

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
