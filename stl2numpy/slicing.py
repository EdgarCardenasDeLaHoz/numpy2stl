"""slice_mesh / rasterize_slice: cross-sectional analysis of a mesh."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)


def slice_mesh(
    file_path: str,
    z_levels: list[float] | np.ndarray | None = None,
    n_slices: int = 10,
    z_axis: int = 2,
) -> list[dict]:
    """
    Extract 2D cross-sections of a mesh at specified heights.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    z_levels : list of float or None
        Heights at which to slice.  None = evenly spaced *n_slices* values
        between the mesh's min and max along z_axis.
    n_slices : int
        Number of evenly-spaced slices when z_levels is None.  Default 10.
    z_axis : int
        Axis index used as the slice direction (default 2 = Z).

    Returns
    -------
    list of dict, one per slice level, each with:
        'z'         : float  slice height
        'polygons'  : list of (N, 2) ndarray  closed 2D contours (may be empty)
        'section'   : trimesh Path3D or None  raw trimesh section object
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    mesh = _load_trimesh_mesh(file_path)
    bounds = mesh.bounds  # (2, 3)

    z_min = float(bounds[0, z_axis])
    z_max = float(bounds[1, z_axis])

    if z_levels is None:
        z_levels = np.linspace(z_min, z_max, n_slices + 2)[1:-1]
    else:
        z_levels = np.asarray(z_levels, dtype=float)

    normal = np.zeros(3)
    normal[z_axis] = 1.0

    results = []
    for z in z_levels:
        origin = np.zeros(3)
        origin[z_axis] = float(z)

        try:
            section = mesh.section(plane_origin=origin, plane_normal=normal)
        except Exception as e:
            logger.debug("slice at z=%.4f failed: %s", z, e)
            section = None

        polygons = _section_to_polygons(section)
        results.append({"z": float(z), "polygons": polygons, "section": section})

    return results


def rasterize_slice(
    file_path: str,
    z: float,
    resolution: int = 512,
    z_axis: int = 2,
) -> np.ndarray:
    """
    Rasterize a single cross-section into a 2D boolean mask.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    z : float
        Height at which to slice.
    resolution : int
        Output mask size (square).  Default 512.
    z_axis : int
        Axis used as the slice direction (default 2 = Z).

    Returns
    -------
    ndarray, shape (resolution, resolution), dtype bool
        True = inside the cross-section polygon.
    """
    slices = slice_mesh(file_path, z_levels=[z], z_axis=z_axis)
    polygons = slices[0]["polygons"]

    mask = np.zeros((resolution, resolution), dtype=bool)
    if not polygons:
        return mask

    # Find 2D bounding box across all polygons
    all_pts = np.vstack(polygons)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    if x_min == x_max or y_min == y_max:
        return mask

    try:
        from skimage.draw import polygon as draw_polygon

        for poly_pts in polygons:
            col = (poly_pts[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)
            row = (poly_pts[:, 1] - y_min) / (y_max - y_min) * (resolution - 1)
            rr, cc = draw_polygon(row.astype(int), col.astype(int), shape=mask.shape)
            mask[rr, cc] = True

    except ImportError:
        # Fallback: scanline fill without skimage
        mask = _scanline_fill(polygons, resolution, x_min, x_max, y_min, y_max)

    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_to_polygons(section) -> list[np.ndarray]:
    """Convert a trimesh Path3D section to a list of (N, 2) coordinate arrays."""
    if section is None:
        return []
    try:
        path_2d, _ = section.to_2D()
        polygons = []
        for poly in path_2d.polygons_closed:
            coords = np.array(poly.exterior.coords, dtype=np.float64)
            if len(coords) >= 3:
                polygons.append(coords[:, :2])
        return polygons
    except Exception as e:
        logger.debug("section → polygon conversion failed: %s", e)
        return []


def _scanline_fill(
    polygons: list[np.ndarray],
    resolution: int,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> np.ndarray:
    """Minimal scanline polygon fill (fallback when skimage unavailable)."""
    mask = np.zeros((resolution, resolution), dtype=bool)
    for poly_pts in polygons:
        col = ((poly_pts[:, 0] - x_min) / (x_max - x_min) * (resolution - 1)).astype(int)
        row = ((poly_pts[:, 1] - y_min) / (y_max - y_min) * (resolution - 1)).astype(int)
        # Bounding-box scan
        r_min, r_max = row.min(), row.max()
        c_min, c_max = col.min(), col.max()
        for r in range(max(r_min, 0), min(r_max + 1, resolution)):
            intersections = []
            n = len(row)
            for i in range(n):
                j = (i + 1) % n
                r0, r1 = row[i], row[j]
                c0, c1 = col[i], col[j]
                if (r0 <= r < r1) or (r1 <= r < r0):
                    if r1 != r0:
                        cx = c0 + (r - r0) * (c1 - c0) / (r1 - r0)
                        intersections.append(cx)
            intersections.sort()
            for k in range(0, len(intersections) - 1, 2):
                c_start = max(int(intersections[k]), 0)
                c_end = min(int(intersections[k + 1]) + 1, resolution)
                mask[r, c_start:c_end] = True
    return mask
