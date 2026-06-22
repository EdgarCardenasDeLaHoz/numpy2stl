"""Mesh analysis: get_mesh_properties, detect_orientation."""

from __future__ import annotations

import logging

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)


def get_mesh_properties(file_path: str) -> dict:
    """
    Extract geometric properties from a mesh file.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.

    Returns
    -------
    dict with keys:
        'volume'              : float or None (None if mesh is not watertight)
        'surface_area'        : float
        'bounds'              : {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        'extents'             : [x_size, y_size, z_size]
        'center_of_mass'      : [x, y, z]
        'num_vertices'        : int
        'num_faces'           : int
        'is_watertight'       : bool
        'is_winding_consistent': bool
        'euler_number'        : int
    """
    mesh = _load_trimesh_mesh(file_path)

    bounds = mesh.bounds  # (2, 3)

    volume: float | None
    try:
        volume = float(mesh.volume) if mesh.is_watertight else None
    except Exception:
        volume = None

    return {
        "volume": volume,
        "surface_area": float(mesh.area),
        "bounds": {
            "x": (float(bounds[0, 0]), float(bounds[1, 0])),
            "y": (float(bounds[0, 1]), float(bounds[1, 1])),
            "z": (float(bounds[0, 2]), float(bounds[1, 2])),
        },
        "extents": mesh.extents.tolist(),
        "center_of_mass": mesh.center_mass.tolist(),
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
    }


def detect_orientation(file_path: str) -> dict:
    """
    Heuristically detect which mesh axis represents "up".

    Strategy:
    1. The axis with the smallest extent is often the height axis for flat
       objects (e.g. terrain tiles).
    2. The axis most aligned with the dominant face normal is a strong signal
       for solid objects.
    3. Falls back to Z if both methods are ambiguous.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.

    Returns
    -------
    dict with keys:
        'up_axis'       : int  (0=X, 1=Y, 2=Z)
        'up_axis_name'  : str  ('x', 'y', or 'z')
        'confidence'    : float  (0–1, higher = more certain)
        'method'        : str  explanation of which heuristic was used
    """
    mesh = _load_trimesh_mesh(file_path)

    extents = mesh.extents  # [dx, dy, dz]

    # --- Heuristic 1: extent ratio ---
    # The "flat" axis (smallest extent) is a strong candidate for the height
    # axis when the object is a terrain slab.
    flat_axis = int(np.argmin(extents))
    flat_ratio = float(extents[flat_axis] / extents.max()) if extents.max() > 0 else 1.0

    # --- Heuristic 2: dominant normal direction ---
    normals = mesh.face_normals  # (N, 3)
    areas = mesh.area_faces      # (N,)
    # Area-weighted mean normal
    if areas.sum() > 0:
        weighted = np.abs(normals) * areas[:, None]
        dominant = weighted.sum(axis=0) / areas.sum()
    else:
        dominant = np.abs(normals).mean(axis=0)

    normal_axis = int(np.argmax(dominant))
    normal_confidence = float(dominant[normal_axis] / dominant.sum()) if dominant.sum() > 0 else 0.33

    # --- Combine heuristics ---
    # Strong normal signal wins; weak normal → fall back to extent
    if normal_confidence > 0.6:
        up_axis = normal_axis
        confidence = normal_confidence
        method = "dominant face normal"
    elif flat_ratio < 0.3:
        up_axis = flat_axis
        confidence = float(1.0 - flat_ratio)
        method = "smallest extent (flat slab)"
    else:
        # Both ambiguous — default to Z
        up_axis = 2
        confidence = 0.5
        method = "default (Z axis)"

    return {
        "up_axis": up_axis,
        "up_axis_name": ["x", "y", "z"][up_axis],
        "confidence": confidence,
        "method": method,
    }
