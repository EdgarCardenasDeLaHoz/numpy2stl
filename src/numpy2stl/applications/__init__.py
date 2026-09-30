"""numpy2stl.applications — ready-made models built from the core primitives (puzzles).

The OSM city helpers that used to live here moved to map2stl's ``city2stl.osm_raster``
(numpy2stl is geo-free).
"""
from .puzzle import make_base_border, make_puzzle_model, make_puzzle_piece, make_puzzle_pts

__all__ = [
    "make_puzzle_model",
    "make_base_border",
    "make_puzzle_pts",
    "make_puzzle_piece",
]
