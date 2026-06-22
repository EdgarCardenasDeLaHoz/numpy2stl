"""stl2numpy — convert STL/OBJ/3MF meshes back into NumPy arrays.

Usage::

    from numpy2stl.stl2numpy import mesh_to_heightmap, get_mesh_properties

Phase 1 — core conversions:
    mesh_to_heightmap     STL/OBJ → 2D elevation array
    get_mesh_properties   Geometry statistics (volume, area, bounds …)

Phase 2 — advanced conversions:
    mesh_to_voxels        STL/OBJ → 3D boolean voxel grid
    mesh_to_pointcloud    STL/OBJ → (N, 3) or (N, 6) point cloud
    slice_mesh            Cross-sections at given Z levels
    rasterize_slice       Single cross-section → 2D boolean mask

Phase 3 — utilities:
    detect_orientation    Heuristic "which axis is up?" detection
    decimate_mesh         Reduce triangle count (quadric decimation)
"""

from .analysis import detect_orientation, get_mesh_properties
from .heightmap import mesh_to_heightmap
from .pointcloud import mesh_to_pointcloud
from .reduction import decimate_mesh
from .slicing import rasterize_slice, slice_mesh
from .voxels import mesh_to_voxels

__all__ = [
    # Phase 1
    "mesh_to_heightmap",
    "get_mesh_properties",
    # Phase 2
    "mesh_to_voxels",
    "mesh_to_pointcloud",
    "slice_mesh",
    "rasterize_slice",
    # Phase 3
    "detect_orientation",
    "decimate_mesh",
]
