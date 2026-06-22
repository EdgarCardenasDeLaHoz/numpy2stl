"""mesh_to_voxels: convert a mesh to a 3D boolean voxel grid."""

from __future__ import annotations

import logging

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)


def mesh_to_voxels(
    file_path: str,
    resolution: int = 64,
    pitch: float | None = None,
) -> dict:
    """
    Convert a 3D mesh to a boolean voxel grid.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    resolution : int
        Number of voxels along the longest dimension.  Used to compute
        *pitch* when pitch is None.  Default 64.
    pitch : float or None
        Explicit voxel size in mesh units.  Overrides *resolution* when given.

    Returns
    -------
    dict with keys:
        'voxels'     : ndarray, shape (nx, ny, nz), bool  (True = inside)
        'pitch'      : float  voxel edge length
        'origin'     : ndarray (3,)  world coords of voxel [0,0,0]
        'bounds'     : {'x': (min, max), 'y': ..., 'z': ...}
        'resolution' : (nx, ny, nz)
    """
    mesh = _load_trimesh_mesh(file_path)

    if not mesh.is_watertight:
        logger.warning(
            "Mesh is not watertight — voxel interior/exterior may be unreliable."
        )

    if pitch is None:
        pitch = float(mesh.extents.max() / resolution)

    vox = mesh.voxelized(pitch=pitch)

    matrix = vox.matrix  # (nx, ny, nz) bool
    # trimesh VoxelGrid stores position in a 4×4 transform; extract translation
    origin = np.array(vox.transform[:3, 3], dtype=np.float64)
    bounds = mesh.bounds

    return {
        "voxels": matrix,
        "pitch": float(pitch),
        "origin": origin,
        "bounds": {
            "x": (float(bounds[0, 0]), float(bounds[1, 0])),
            "y": (float(bounds[0, 1]), float(bounds[1, 1])),
            "z": (float(bounds[0, 2]), float(bounds[1, 2])),
        },
        "resolution": tuple(int(d) for d in matrix.shape),
    }
