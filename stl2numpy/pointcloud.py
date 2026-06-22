"""mesh_to_pointcloud: sample a mesh surface as a point cloud."""

from __future__ import annotations

import logging

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)


def mesh_to_pointcloud(
    file_path: str,
    n_points: int = 10_000,
    include_normals: bool = False,
    method: str = "surface",
) -> np.ndarray:
    """
    Sample a mesh and return a point cloud.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    n_points : int
        Number of points to sample.  Default 10 000.
    include_normals : bool
        If True, return an (N, 6) array [x, y, z, nx, ny, nz].
        Default False (returns (N, 3)).
    method : {'surface', 'vertices'}
        'surface' (default) — uniformly sample from face surfaces.
        'vertices' — use the raw mesh vertices (ignores n_points).

    Returns
    -------
    ndarray, shape (N, 3) or (N, 6)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh is required. Install with: pip install trimesh")

    mesh = _load_trimesh_mesh(file_path)

    if method == "vertices":
        points = np.array(mesh.vertices, dtype=np.float64)
        if include_normals:
            normals = np.array(mesh.vertex_normals, dtype=np.float64)
            return np.hstack([points, normals])
        return points

    # method == 'surface'
    points, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    points = np.array(points, dtype=np.float64)

    if include_normals:
        normals = np.array(mesh.face_normals[face_idx], dtype=np.float64)
        return np.hstack([points, normals])

    return points
