"""numpy2stl — convert NumPy arrays and geometry into STL/OBJ/3MF meshes.

Core functions are available directly::

    from numpy2stl import array_to_mesh, perimeter_to_walls, Solid, writeSTL

Submodules with heavier dependencies are imported explicitly::

    import numpy2stl.processing.simplify as simp   # needs scipy/shapely
    import numpy2stl.applications.puzzle   as puzzle  # needs trimesh, shapely
    import numpy2stl.processing.boolean    as boolean # needs pymeshlab
    import numpy2stl.utils.visualization   as view    # needs matplotlib
"""

import logging

# Core generation functions
from .core.generate import (
    array2faces,
    array_to_mesh,
    perimeter_to_walls,
    polygon_to_complex,
    polygon_to_prism,
)

# Polygon utilities
from .core.polygon import (
    get_ordered_perimeter,
    get_perimeter_angles,
    rotate_3D,
    triangulate_polygon,
)

# Solid class and utilities
from .core.solid import (
    Solid,
    calculate_normals,
    get_face_area,
    get_open_edges,
    get_surfaces,
    triangles_to_facets,
    validate_object,
    vertices_to_index,
)

# Save/export functions
from .io.writers import (
    write3MF,
    writeOBJ,
    writeSTL,
)

# Tools functions
from .utils.image import (
    rescale,
    resize_max,
)

logging.getLogger(__name__).addHandler(logging.NullHandler())


def get_logger():
    """Get the package logger. Users can configure as needed."""
    return logging.getLogger(__name__)

__all__ = [
    # Core mesh generation
    "array_to_mesh",
    "array2faces",
    "polygon_to_complex",
    "polygon_to_prism",
    "perimeter_to_walls",
    # Tools
    "resize_max",
    "rescale",
    # Export functions
    "writeSTL",
    "write3MF",
    "writeOBJ",
    # Polygon utilities
    "get_ordered_perimeter",
    "triangulate_polygon",
    "get_perimeter_angles",
    "rotate_3D",
    # Solid class and utilities
    "Solid",
    "calculate_normals",
    "vertices_to_index",
    "get_face_area",
    "get_surfaces",
    "triangles_to_facets",
    "get_open_edges",
    "validate_object",
]
