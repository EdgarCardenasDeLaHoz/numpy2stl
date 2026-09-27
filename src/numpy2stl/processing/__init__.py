from ..core.solid import simplify_surface
from .boolean import from_manifold, to_manifold, union
from .decimate import decimate_to_tolerance, heightfield_tin
from .extrusion import (
    extrude_solid_polygon,
    make_hollow_cap,
    make_hollow_prism_solid,
    make_prism_solid,
    prism_wall_vertices,
    robust_triangulate,
)
from .simplify import simplify_mesh_surfaces
from .verify import check_model_status

__all__ = [
    # boolean
    "to_manifold",
    "from_manifold",
    "union",
    # decimate
    "heightfield_tin",
    "decimate_to_tolerance",
    # simplify
    "simplify_mesh_surfaces",
    "simplify_surface",
    # verify
    "check_model_status",
    # extrusion
    "make_hollow_prism_solid",
    "make_hollow_cap",
    "extrude_solid_polygon",
    "make_prism_solid",
    "prism_wall_vertices",
    "robust_triangulate",
]
