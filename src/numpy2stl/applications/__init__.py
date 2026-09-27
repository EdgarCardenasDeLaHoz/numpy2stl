"""numpy2stl.applications — ready-made models built from the core primitives (puzzles).

The OSM city helpers that used to be re-exported here moved to map2stl's
``city2stl.osm_raster``; asking for them raises an ImportError saying so.
"""
from .puzzle import make_base_border, make_puzzle_model, make_puzzle_piece, make_puzzle_pts

__all__ = [
    "make_puzzle_model",
    "make_base_border",
    "make_puzzle_pts",
    "make_puzzle_piece",
]

_MOVED = {"get_city_bbox", "get_osm_building_heightmap", "get_philadelphia_heightmap"}


def __getattr__(name):
    if name in _MOVED:
        raise ImportError(
            f"numpy2stl.applications.{name} moved to map2stl's city2stl.osm_raster "
            "(numpy2stl is geo-free).")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
