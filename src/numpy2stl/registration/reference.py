"""Reference rasters for ``register_city_stl``: what the STL is registered against.

numpy2stl is geo-free: it never fetches map data and never converts lon/lat to
metres.  The caller passes ``register_city_stl`` a *reference source* that
produces the building heightmap (and optional exclusion masks / measured
heights) on demand, at whatever resolution a stage asks for.  strm2stl's
``city2stl.registration.OSMReference`` fetches these from OpenStreetMap;
``StaticReference`` serves arrays you already have.

A source implements ``ReferenceSource``.  ``target`` values are opaque to
numpy2stl (for OSM they are an (N, S, E, W) bbox or a city name); they are only
handed back to the source.

Raster conventions: ``heightmap`` is float, NaN where there is no building,
row 0 = south (the ``mesh_to_heightmap(row0="south")`` convention the
registration code works in); masks are bool on the same grid.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ReferenceSource(Protocol):
    """What ``register_city_stl`` needs from the reference (map) side.

    Methods
    -------
    resolve_target(stl_z_max, stl_xy_extent, margin) -> (target, anchored)
        The frame to fetch.  ``anchored=True`` means the target is sized to
        ``margin`` x the STL footprint, so the STL fills ``1/margin`` of it (the
        geometric scale anchor); ``False`` means an arbitrary frame.
    building_heightmap(target, resolution) -> dict
        ``heightmap`` (resolution x resolution), ``bounds`` ({"x": (x0, x1),
        "y": (y0, y1), ...}, passed through to the report) and ``cell_size_m``
        (metres per pixel).
    semantic_masks(target, resolution) -> dict | None
        Optional bool masks ``vegetation`` / ``water`` / ``elevated_roadway``
        excluded from the STL building mask.
    ndsm(target, resolution) -> ndarray | None
        Optional measured height above ground on the heightmap grid, used when
        ``height_source="lidar"``.
    m_per_unit(stl_z_max) -> float | None
        STL model units -> metres, when known (mesh simplification budget).
    candidate_targets(stl_z_max, stl_xy_extent, margin) -> list[(label, target)]
        Alternative frames for the centre search (``center_search != "never"``).
    """

    name: str

    def resolve_target(self, stl_z_max: float, stl_xy_extent: float,
                       margin: float) -> tuple[Any, bool]: ...

    def building_heightmap(self, target: Any, resolution: int) -> dict: ...

    def semantic_masks(self, target: Any, resolution: int) -> dict | None: ...

    def ndsm(self, target: Any, resolution: int) -> np.ndarray | None: ...

    def m_per_unit(self, stl_z_max: float) -> float | None: ...

    def candidate_targets(self, stl_z_max: float, stl_xy_extent: float,
                          margin: float) -> list[tuple[Any, Any]]: ...


def _resize_nearest(arr: np.ndarray, resolution: int) -> np.ndarray:
    rows = (np.arange(resolution) * arr.shape[0] / resolution).astype(int)
    cols = (np.arange(resolution) * arr.shape[1] / resolution).astype(int)
    return arr[np.ix_(rows, cols)]


class StaticReference:
    """A ``ReferenceSource`` over arrays already in memory.

    ``heightmap`` (row 0 = south, NaN = no building) covers ``bounds`` at
    ``cell_size_m`` metres per pixel.  Requests at another resolution are
    nearest-neighbour resampled (cell size scaled to match).  ``anchored`` and
    ``margin`` say whether the frame is sized to ``margin`` x the STL footprint.
    """

    def __init__(self, heightmap: np.ndarray, cell_size_m: float, *,
                 bounds: dict | None = None, masks: dict | None = None,
                 ndsm: np.ndarray | None = None, m_per_unit: float | None = None,
                 anchored: bool = False, name: str = "reference"):
        self._hm = np.asarray(heightmap, dtype=np.float64)
        self._cell = float(cell_size_m)
        rows, cols = self._hm.shape
        self._bounds = bounds or {"x": (0.0, cols * self._cell), "y": (0.0, rows * self._cell)}
        self._masks = masks
        self._ndsm = ndsm
        self._m_per_unit = m_per_unit
        self._anchored = anchored
        self.name = name

    def _at(self, arr: np.ndarray, resolution: int) -> np.ndarray:
        return arr if arr.shape == (resolution, resolution) else _resize_nearest(arr, resolution)

    def resolve_target(self, stl_z_max, stl_xy_extent, margin):
        return None, self._anchored

    def building_heightmap(self, target, resolution):
        hm = self._at(self._hm, resolution)
        valid = hm[~np.isnan(hm)]
        return {
            "heightmap": hm,
            "bounds": {**self._bounds, "z": (0.0, float(valid.max()) if valid.size else 0.0)},
            "resolution": hm.shape,
            "cell_size_m": self._cell * self._hm.shape[0] / resolution,
            "projection": "max",
        }

    def semantic_masks(self, target, resolution):
        if not self._masks:
            return None
        return {k: (self._at(np.asarray(v, dtype=bool), resolution) if v is not None else None)
                for k, v in self._masks.items()}

    def ndsm(self, target, resolution):
        return None if self._ndsm is None else self._at(self._ndsm, resolution)

    def m_per_unit(self, stl_z_max):
        return self._m_per_unit

    def candidate_targets(self, stl_z_max, stl_xy_extent, margin):
        return []
