"""raster — generic 2-D raster helpers (no geo, no registration).

Layer: below ``registration`` / ``applications``, beside ``processing`` and
``stl2numpy``; imports only numpy/scipy/cv2/shapely/rasterio.  Row 0 = north
by default (image convention); mesh-derived rasters that keep row 0 = south
are handled as given (every function here is orientation-agnostic except
``burn_polygons(bounds=...)``, which is north-up).

- ``segment``    ``terrain_residual``, ``hill_relief_mask``, ``building_mask``,
                 ``split_touching_buildings``, ``building_edges``
- ``vectorize``  ``vectorize_buildings`` (mask → polygons)
- ``burn``       ``burn_polygons`` (polygons → raster; max / sum / set, holes kept)
- ``fill``       ``fill_nan`` (nearest / median / constant)
"""
from __future__ import annotations

from .burn import burn_polygons
from .fill import fill_nan
from .segment import (
    building_edges,
    building_mask,
    hill_relief_mask,
    split_touching_buildings,
    terrain_residual,
)
from .vectorize import vectorize_buildings

__all__ = [
    "burn_polygons",
    "fill_nan",
    "terrain_residual",
    "hill_relief_mask",
    "building_mask",
    "split_touching_buildings",
    "building_edges",
    "vectorize_buildings",
]
