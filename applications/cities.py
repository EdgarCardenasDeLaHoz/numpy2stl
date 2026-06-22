"""OSM building-height raster generation for named cities.

Mirrors the oceans.py pattern: named region wrappers with hardcoded bboxes
call a generic get_osm_building_heightmap() core function.

Return format matches stl2numpy/heightmap.py::mesh_to_heightmap() exactly so
both can be passed directly to registration.align.register().
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

# Per-city config used for scale estimation and tight bbox generation.
# 'center'  : (lat, lon) of the downtown/model area the STL represents
# 'tallest_m': height of the tallest building in the city centre (metres)
#              — used as a scale anchor: stl_max_z × scale = tallest_m
# 'full_bbox': (N, S, E, W) full-city fallback when scale is unknown
_CITY_CONFIG: dict[str, dict] = {
    "philadelphia": {
        "center": (39.952, -75.164),   # City Hall / Center City
        "tallest_m": 342.0,            # Comcast Technology Center
        "full_bbox": (40.060, 39.860, -74.950, -75.280),
    },
    "new york": {
        "center": (40.758, -73.985),   # Midtown Manhattan
        "tallest_m": 541.0,            # One World Trade Center
        "full_bbox": (40.920, 40.490, -73.700, -74.260),
    },
    "chicago": {
        "center": (41.882, -87.629),   # The Loop
        "tallest_m": 442.0,            # Willis Tower
        "full_bbox": (42.020, 41.640, -87.520, -87.940),
    },
    "boston": {
        "center": (42.356, -71.062),   # Downtown / Financial District
        "tallest_m": 240.0,            # 200 Clarendon (Hancock Tower)
        "full_bbox": (42.400, 42.220, -70.990, -71.190),
    },
}

# Aliases → canonical key in _CITY_CONFIG
_CITY_ALIASES: dict[str, str] = {
    k.lower(): v for k, v in {
        "philadelphia": "philadelphia",
        "philadelphia, pa": "philadelphia",
        "philadelphia, pa, usa": "philadelphia",
        "new york": "new york",
        "new york, ny": "new york",
        "new york, ny, usa": "new york",
        "nyc": "new york",
        "chicago": "chicago",
        "chicago, il": "chicago",
        "chicago, il, usa": "chicago",
        "boston": "boston",
        "boston, ma": "boston",
        "boston, ma, usa": "boston",
    }.items()
}


def tight_bbox_from_extent(
    center_lat: float,
    center_lon: float,
    extent_m: float,
    margin: float = 1.5,
) -> tuple[float, float, float, float]:
    """
    Return (N, S, E, W) bbox centered on (lat, lon) with side length
    extent_m * margin (metres).
    """
    import math
    half = extent_m * margin / 2.0
    d_lat = half / 111_000.0
    d_lon = half / (111_000.0 * math.cos(math.radians(center_lat)))
    return (
        center_lat + d_lat,
        center_lat - d_lat,
        center_lon + d_lon,
        center_lon - d_lon,
    )


def derive_scale_m_per_unit(
    city_name,
    stl_z_max: float,
    tallest_m: float | None = None,
    scale_m_per_unit: float | None = None,
) -> float | None:
    """Resolve the model's metres-per-unit scale anchor.

    Priority: explicit `scale_m_per_unit` → `tallest_m`/stl_z_max → city-config
    tallest/stl_z_max.  Returns None when no anchor is available.  Shared by
    `estimate_bbox_from_stl` and the mesh simplifier (which needs to convert a
    metres deviation budget into the mesh's own units).
    """
    if scale_m_per_unit is not None:
        return float(scale_m_per_unit)
    if stl_z_max <= 0:
        return None
    if tallest_m is not None:
        return float(tallest_m) / stl_z_max
    key = _CITY_ALIASES.get(city_name.lower().strip()) if isinstance(city_name, str) else None
    cfg = _CITY_CONFIG.get(key) if key else None
    if cfg is not None:
        return cfg["tallest_m"] / stl_z_max
    return None


def estimate_bbox_from_stl(
    city_name: str,
    stl_z_max: float,
    stl_xy_extent: float,
    osm_margin: float = 1.5,
    center: tuple[float, float] | None = None,
    tallest_m: float | None = None,
    scale_m_per_unit: float | None = None,
) -> tuple[float, float, float, float] | None:
    """
    Estimate a tight (N, S, E, W) OSM bbox from the STL's scale anchor:
        scale (m/unit) = tallest_m / stl_z_max  (or `scale_m_per_unit` directly)
        geographic extent (m) = stl_xy_extent × scale
    centred on the city's downtown point.

    City-agnostic resolution (no longer limited to the 4 hardcoded cities):
      - **centre**: explicit `center` arg → `_CITY_CONFIG` entry → geocoded
        centroid of the city via `get_city_bbox()`.
      - **scale**: explicit `scale_m_per_unit` → `tallest_m`/stl_z_max →
        `_CITY_CONFIG` tallest/stl_z_max.

    Returns the tight bbox, or None when neither a scale anchor nor a centre
    can be determined (caller then falls back to the full-city fetch).
    """
    if stl_z_max <= 0:
        return None
    key = _CITY_ALIASES.get(city_name.lower().strip()) if isinstance(city_name, str) else None
    cfg = _CITY_CONFIG.get(key) if key else None

    # --- resolve metres-per-unit scale ---
    if scale_m_per_unit is not None:
        scale = float(scale_m_per_unit)
        scale_src = "override"
    elif tallest_m is not None:
        scale = float(tallest_m) / stl_z_max
        scale_src = f"tallest_m={tallest_m:.0f}"
    elif cfg is not None:
        scale = cfg["tallest_m"] / stl_z_max
        scale_src = f"config tallest_m={cfg['tallest_m']:.0f}"
    else:
        scale = None
        scale_src = "unknown"

    # --- resolve centre (lat, lon) ---
    if center is not None:
        clat, clon = float(center[0]), float(center[1])
    elif cfg is not None:
        clat, clon = cfg["center"]
    else:
        try:
            N, S, E, W = get_city_bbox(city_name)
            clat, clon = (N + S) / 2.0, (E + W) / 2.0
            logger.info("Geocoded centre for %r: (%.4f, %.4f)", city_name, clat, clon)
        except Exception as exc:
            logger.warning("Could not geocode centre for %r (%s); "
                           "no tight bbox.", city_name, exc)
            return None

    if scale is None:
        logger.warning("No scale anchor for %r (not in config; pass tallest_m or "
                       "scale_m_per_unit) — falling back to full-city fetch.", city_name)
        return None

    extent_m = stl_xy_extent * scale
    logger.info(
        "Scale estimate: %.2f m/unit (%s)  centre=(%.4f,%.4f)  footprint ~%.0f m",
        scale, scale_src, clat, clon, extent_m,
    )
    return tight_bbox_from_extent(clat, clon, extent_m, margin=osm_margin)

try:
    import osmnx as ox
    HAS_OSMNX = True
except ImportError:
    ox = None
    HAS_OSMNX = False

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    gpd = None
    HAS_GEOPANDAS = False

try:
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_city_bbox(city_name: str) -> tuple[float, float, float, float]:
    """
    Geocode a city name and return its bounding box.

    Parameters
    ----------
    city_name : str
        Human-readable name, e.g. "Philadelphia, PA, USA".

    Returns
    -------
    tuple (N, S, E, W) in decimal degrees — matches oceans.py convention.
    """
    if not HAS_OSMNX:
        raise ImportError("osmnx is required. Install with: pip install osmnx")

    gdf = ox.geocode_to_gdf(city_name)
    if gdf is None or len(gdf) == 0:
        raise ValueError(f"Could not geocode city: {city_name!r}")

    # total_bounds returns (minx, miny, maxx, maxy) = (W, S, E, N)
    W, S, E, N = gdf.total_bounds
    return (float(N), float(S), float(E), float(W))


_OSM_CACHE_DIR = Path(__file__).parent.parent / "registration" / "runs" / "osm_cache"


def _osm_cache_path(N, S, E, W, resolution, default_height, levels_to_meters) -> Path:
    import hashlib
    key = f"{N:.5f}_{S:.5f}_{E:.5f}_{W:.5f}_r{resolution}_h{default_height}_l{levels_to_meters}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _OSM_CACHE_DIR / f"osm_{h}.npz"


def _osm_semantic_cache_path(N, S, E, W, resolution) -> Path:
    import hashlib
    key = f"sem_{N:.5f}_{S:.5f}_{E:.5f}_{W:.5f}_r{resolution}"
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return _OSM_CACHE_DIR / f"osm_{h}.npz"


def get_osm_building_heightmap(
    bbox_or_city: Union[tuple, str],
    resolution: int = 512,
    default_height: float = 10.0,
    levels_to_meters: float = 3.5,
    cache: bool = True,
) -> dict:
    """
    Fetch OSM building footprints and rasterize their heights.

    Parameters
    ----------
    bbox_or_city : (N, S, E, W) tuple or city name string
        Geographic bounds. If a string, get_city_bbox() is called first.
    resolution : int
        Output grid size (square). Default 512.
    default_height : float
        Fallback height (metres) for buildings with no OSM height/levels tag.
    levels_to_meters : float
        Floors-to-metres conversion. Default 3.5 m/floor.

    Returns
    -------
    dict matching mesh_to_heightmap() format:
        'heightmap'  : ndarray (rows, cols) float64 — NaN where no building
        'bounds'     : {'x': (W, E), 'y': (S, N), 'z': (0.0, max_height_m)}
        'resolution' : (rows, cols)
        'cell_size'  : (dx_degrees, dy_degrees)
        'projection' : 'max'
    """
    if not HAS_OSMNX:
        raise ImportError("osmnx is required. Install with: pip install osmnx")

    # 1. Resolve bbox — check city config for full bbox, then geocode
    if isinstance(bbox_or_city, str):
        key = _CITY_ALIASES.get(bbox_or_city.lower().strip())
        if key is not None:
            N, S, E, W = _CITY_CONFIG[key]["full_bbox"]
            logger.info("Using hardcoded full-city bbox for %r", bbox_or_city)
        else:
            N, S, E, W = get_city_bbox(bbox_or_city)
    else:
        N, S, E, W = bbox_or_city

    # 1b. Cache check (OSM fetch is the expensive step)
    cache_path = _osm_cache_path(N, S, E, W, resolution, default_height, levels_to_meters)
    if cache and cache_path.exists():
        logger.info("Loading OSM heightmap from cache: %s", cache_path.name)
        data = np.load(cache_path)
        return _make_result(data["heightmap"], N, S, E, W, resolution)

    logger.info("Fetching OSM buildings for bbox N=%.4f S=%.4f E=%.4f W=%.4f", N, S, E, W)

    # 2. Fetch buildings via osmnx 2.x API
    # osmnx 2.x: features_from_bbox(bbox=(left, bottom, right, top)) = (W, S, E, N)
    try:
        gdf = ox.features_from_bbox(bbox=(W, S, E, N), tags={"building": True})
    except Exception as e:
        logger.warning("osmnx features_from_bbox failed (%s); trying alternate signature", e)
        # Some osmnx builds take (north, south, east, west) positional args
        gdf = ox.features_from_bbox(N, S, E, W, tags={"building": True})

    if gdf is None or len(gdf) == 0:
        logger.warning("No buildings found in bbox; returning all-NaN heightmap.")
        heightmap = np.full((resolution, resolution), np.nan, dtype=np.float64)
        return _make_result(heightmap, N, S, E, W, resolution)

    # 3. Filter to polygon geometries only
    mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[mask].copy()
    logger.info("Found %d building polygons", len(gdf))

    if len(gdf) == 0:
        heightmap = np.full((resolution, resolution), np.nan, dtype=np.float64)
        return _make_result(heightmap, N, S, E, W, resolution)

    # 4. Resolve per-building heights
    gdf["height_m"] = gdf.apply(
        lambda row: _resolve_building_height(row, default_height, levels_to_meters),
        axis=1,
    )

    # 5. Rasterize — sort ascending so tallest buildings write last (= max per cell)
    gdf = gdf.sort_values("height_m", ascending=True)
    heightmap = _rasterize_buildings(gdf, N, S, E, W, resolution)

    # 6. Save to cache
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, heightmap=heightmap)
        logger.info("Cached OSM heightmap: %s", cache_path.name)

    return _make_result(heightmap, N, S, E, W, resolution)


def get_osm_semantic_masks(
    bbox_or_city: Union[tuple, str],
    resolution: int = 512,
    cache: bool = True,
) -> dict:
    """
    Fetch OSM vegetation and water polygons and rasterize them to boolean masks
    on the same grid as get_osm_building_heightmap().

    These label regions that are NOT buildings (parks, forests, rivers, lakes),
    so STL height that falls under them — trees, riverbanks misread as structures
    — can be excluded from the building mask before comparison.

    Returns
    -------
    dict with keys:
        'vegetation' : (rows, cols) bool — True under woods/forest/grass/parks
        'water'      : (rows, cols) bool — True under water/waterways
        'bounds', 'resolution'  (matching _make_result conventions)
    """
    if not HAS_OSMNX:
        raise ImportError("osmnx is required. Install with: pip install osmnx")

    if isinstance(bbox_or_city, str):
        key = _CITY_ALIASES.get(bbox_or_city.lower().strip())
        if key is not None:
            N, S, E, W = _CITY_CONFIG[key]["full_bbox"]
        else:
            N, S, E, W = get_city_bbox(bbox_or_city)
    else:
        N, S, E, W = bbox_or_city

    cache_path = _osm_semantic_cache_path(N, S, E, W, resolution)
    if cache and cache_path.exists():
        logger.info("Loading OSM semantic masks from cache: %s", cache_path.name)
        data = np.load(cache_path)
        return {
            "vegetation": data["vegetation"].astype(bool),
            "water": data["water"].astype(bool),
            "bounds": {"x": (float(W), float(E)), "y": (float(S), float(N))},
            "resolution": (resolution, resolution),
        }

    veg_tags = {
        "natural": ["wood", "tree", "tree_row", "scrub", "grassland"],
        "landuse": ["forest", "grass", "meadow", "recreation_ground", "village_green"],
        "leisure": ["park", "garden", "nature_reserve"],
    }
    water_tags = {"natural": ["water", "wetland"], "waterway": True, "water": True}

    veg = _rasterize_semantic(veg_tags, N, S, E, W, resolution)
    water = _rasterize_semantic(water_tags, N, S, E, W, resolution)
    logger.info("OSM semantic masks: vegetation=%.1f%%  water=%.1f%% of frame",
                100.0 * veg.mean(), 100.0 * water.mean())

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vegetation=veg, water=water)
        logger.info("Cached OSM semantic masks: %s", cache_path.name)

    return {
        "vegetation": veg,
        "water": water,
        "bounds": {"x": (float(W), float(E)), "y": (float(S), float(N))},
        "resolution": (resolution, resolution),
    }


def _rasterize_semantic(tags: dict, N, S, E, W, resolution: int) -> np.ndarray:
    """Fetch polygons for `tags` and rasterize to a boolean presence mask."""
    empty = np.zeros((resolution, resolution), dtype=bool)
    try:
        gdf = ox.features_from_bbox(bbox=(W, S, E, N), tags=tags)
    except Exception as e:
        logger.warning("OSM semantic fetch failed for %s (%s); empty mask.",
                       list(tags), e)
        return empty

    if gdf is None or len(gdf) == 0:
        return empty
    poly = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(poly) == 0:
        return empty

    # Reuse the building rasterizer with a constant height of 1.0 → presence.
    poly["height_m"] = 1.0
    arr = _rasterize_buildings(poly, N, S, E, W, resolution)
    return ~np.isnan(arr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_building_height(
    row,
    default_height: float,
    levels_to_meters: float,
) -> float:
    """Height priority: OSM 'height' tag → building:levels × factor → default."""
    # 1. Try 'height' tag (may be "15", "15 m", "15.5m")
    h = row.get("height", None)
    if h is not None and h == h:  # not NaN
        try:
            return float(re.sub(r"[^\d.]", "", str(h)))
        except ValueError:
            pass

    # 2. Try 'building:levels' tag
    lvl = row.get("building:levels", None)
    if lvl is None:
        lvl = row.get("levels", None)
    if lvl is not None and lvl == lvl:
        try:
            return float(lvl) * levels_to_meters
        except ValueError:
            pass

    return default_height


def _rasterize_buildings(gdf, N, S, E, W, resolution: int) -> np.ndarray:
    """Rasterize building polygons with their heights into a (rows, cols) array."""
    if HAS_RASTERIO:
        return _rasterize_rasterio(gdf, N, S, E, W, resolution)
    return _rasterize_numpy(gdf, N, S, E, W, resolution)


def _rasterize_rasterio(gdf, N, S, E, W, resolution: int) -> np.ndarray:
    """Rasterize using rasterio — fast and handles edge cases well."""
    # from_bounds produces a north-up transform (row 0 = north)
    transform = from_bounds(west=W, south=S, east=E, north=N,
                            width=resolution, height=resolution)

    shapes = [
        (geom, float(h))
        for geom, h in zip(gdf.geometry, gdf["height_m"])
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return np.full((resolution, resolution), np.nan, dtype=np.float64)

    arr = rio_rasterize(
        shapes,
        out_shape=(resolution, resolution),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype=np.float64,
    )

    # Mark cells with no building as NaN (fill=0 means "no building touched here")
    # We distinguish NaN (no building) from 0 (a building with zero height, rare)
    # by using a separate rasterization of footprint presence.
    presence = rio_rasterize(
        [(geom, 1.0) for geom, _ in shapes],
        out_shape=(resolution, resolution),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype=np.float64,
    )
    arr[presence == 0] = np.nan

    # Flip to row 0 = south (matches mesh_to_heightmap convention)
    return np.flipud(arr)


def _rasterize_numpy(gdf, N, S, E, W, resolution: int) -> np.ndarray:
    """Pure-numpy fallback rasterization (slow but dependency-free)."""
    try:
        from shapely.geometry import mapping
    except ImportError:
        pass

    arr = np.full((resolution, resolution), np.nan, dtype=np.float64)
    dx = (E - W) / resolution
    dy = (N - S) / resolution

    for _, row in gdf.iterrows():
        geom = row.geometry
        h = float(row["height_m"])
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        # Pixel coords (row 0 = south in our output convention)
        col_min = max(0, int((bounds[0] - W) / dx))
        col_max = min(resolution - 1, int((bounds[2] - W) / dx))
        row_min = max(0, int((bounds[1] - S) / dy))
        row_max = min(resolution - 1, int((bounds[3] - S) / dy))

        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                lon = W + (c + 0.5) * dx
                lat = S + (r + 0.5) * dy
                from shapely.geometry import Point
                pt = Point(lon, lat)
                if geom.contains(pt):
                    if np.isnan(arr[r, c]) or arr[r, c] < h:
                        arr[r, c] = h

    return arr


def _make_result(heightmap: np.ndarray, N, S, E, W, resolution: int) -> dict:
    """Assemble the return dict matching mesh_to_heightmap() format."""
    valid = heightmap[~np.isnan(heightmap)]
    z_max = float(valid.max()) if len(valid) > 0 else 0.0

    return {
        "heightmap": heightmap,
        "bounds": {
            "x": (float(W), float(E)),
            "y": (float(S), float(N)),
            "z": (0.0, z_max),
        },
        "resolution": (int(heightmap.shape[0]), int(heightmap.shape[1])),
        "cell_size": (
            (E - W) / resolution,
            (N - S) / resolution,
        ),
        "projection": "max",
    }


# ---------------------------------------------------------------------------
# Named city wrappers (hardcoded bboxes for reproducibility, like oceans.py)
# ---------------------------------------------------------------------------

def get_philadelphia_heightmap(resolution: int = 512) -> dict:
    """Philadelphia, PA — covers the full city extent."""
    bbox = (40.060, 39.860, -74.950, -75.280)  # (N, S, E, W)
    return get_osm_building_heightmap(bbox, resolution=resolution)


def get_new_york_heightmap(resolution: int = 512) -> dict:
    """New York City, NY — Manhattan + outer boroughs."""
    bbox = (40.920, 40.490, -73.700, -74.260)
    return get_osm_building_heightmap(bbox, resolution=resolution)


def get_chicago_heightmap(resolution: int = 512) -> dict:
    """Chicago, IL."""
    bbox = (42.020, 41.640, -87.520, -87.940)
    return get_osm_building_heightmap(bbox, resolution=resolution)


def get_boston_heightmap(resolution: int = 512) -> dict:
    """Boston, MA."""
    bbox = (42.400, 42.220, -70.990, -71.190)
    return get_osm_building_heightmap(bbox, resolution=resolution)
