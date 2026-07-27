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
import pandas as pd

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


def get_city_center_point(city_name: str) -> tuple[float, float] | None:
    """Best-effort (lat, lon) for a city's downtown/CBD, for anchoring a tight
    OSM-fetch bbox around wherever a partial-coverage model (e.g. a downtown-only
    STL) actually is — NOT the same as get_city_bbox()'s administrative-boundary
    centroid, which for a sprawling metro (e.g. Miami) can land many km from
    downtown, off the edge of what a small model actually covers.

    Tries "Downtown {city_name}" first (a point geocode via Nominatim, which
    resolves recognizable "Downtown X" / "X Central Business District" queries
    for most sizeable cities); falls back to get_city_bbox()'s centroid, then
    None if geocoding fails entirely.

    Measured across 7 Micropolitan-pack cities (Barcelona, Bilbao, Lisbon,
    Paris, Prague, Salzburg, Valencia) using the tight-bbox
    scale+center anchor this function feeds: confidence/footprint_iou
    improved for 4 (Bilbao, Lisbon, Prague, Valencia) and regressed for 3
    (Barcelona, Paris, Salzburg) vs. the old full-city-fetch behavior — all 7
    got dramatically faster (100-200s vs 165-725s) since the OSM fetch area
    shrank either way. The "Downtown {city}" geocode landed at plausible,
    named landmarks for all 3 regressions (Plaça Catalunya, Île de la
    Cité/Notre-Dame, Salzburg old town) — not an obviously bad center — so
    the likely cause is that Micropolitan's own crop center for those
    specific models doesn't coincide with the conventional "downtown" point
    for that city, not a bug in the geocode itself. Not resolved; would need
    per-model ground truth (the STL's actual real-world crop center) to fix
    properly rather than guessing at a better heuristic.
    """
    if not HAS_OSMNX:
        return None
    try:
        lat, lon = ox.geocode(f"Downtown {city_name}")
        return (float(lat), float(lon))
    except Exception as exc:
        logger.info("Could not geocode 'Downtown %s' (%s); falling back to city centroid.",
                    city_name, exc)
    try:
        n, s, e, w = get_city_bbox(city_name)
        return ((n + s) / 2.0, (e + w) / 2.0)
    except Exception as exc:
        logger.warning("Could not determine a center point for %r (%s).", city_name, exc)
        return None


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
    Fetch OSM vegetation/water polygons and elevated-roadway ways, rasterized
    to boolean masks on the same grid as get_osm_building_heightmap().

    These label regions that are NOT buildings but can still rise above local
    terrain in an STL model — trees, riverbanks, and elevated highways/overpasses
    all read as "height above ground" to the top-hat segmentation the same way a
    building does, and OSM has no building footprint there to match against. STL
    height under these regions can be excluded from the building mask before
    registration/comparison.

    Elevated roadways specifically (bridge=yes highways — overpasses, viaducts,
    elevated highway sections) were a measured, real source of false-positive
    "buildings" in the STL segmentation (e.g. Miami's elevated highway ramps),
    distinct from ordinary at-grade roads which lie flush with the ground and
    don't trigger the top-hat filter in the first place — so only bridges are
    fetched here, not the full road network.

    Returns
    -------
    dict with keys:
        'vegetation'       : (rows, cols) bool — True under woods/forest/grass/parks
        'water'            : (rows, cols) bool — True under water/waterways
        'elevated_roadway' : (rows, cols) bool — True under bridges/overpasses/viaducts
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
        # Older cache entries predate the elevated_roadway mask — treat as
        # "none found" rather than a stale/incomplete cache hit that silently
        # skips the bridge exclusion.
        elevated = (data["elevated_roadway"].astype(bool)
                    if "elevated_roadway" in data.files
                    else np.zeros((resolution, resolution), dtype=bool))
        return {
            "vegetation": data["vegetation"].astype(bool),
            "water": data["water"].astype(bool),
            "elevated_roadway": elevated,
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
    elevated = _rasterize_elevated_roadways(N, S, E, W, resolution)
    logger.info("OSM semantic masks: vegetation=%.1f%%  water=%.1f%%  "
                "elevated_roadway=%.1f%% of frame",
                100.0 * veg.mean(), 100.0 * water.mean(), 100.0 * elevated.mean())

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vegetation=veg, water=water,
                            elevated_roadway=elevated)
        logger.info("Cached OSM semantic masks: %s", cache_path.name)

    return {
        "vegetation": veg,
        "water": water,
        "elevated_roadway": elevated,
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


# Fallback roadway width (metres) when OSM has no width/lanes tag, by highway
# class — bridges are almost always multi-lane arterial/motorway sections.
_HIGHWAY_DEFAULT_WIDTH_M = {
    "motorway": 15.0, "motorway_link": 8.0,
    "trunk": 12.0, "trunk_link": 7.0,
    "primary": 10.0, "primary_link": 6.0,
    "secondary": 9.0, "secondary_link": 6.0,
}
_HIGHWAY_FALLBACK_WIDTH_M = 7.0  # any other bridge-tagged highway class


def _rasterize_elevated_roadways(N, S, E, W, resolution: int) -> np.ndarray:
    """Fetch bridge/viaduct highway ways and rasterize a buffered presence mask.

    OSM roads are LineStrings with no inherent width, so real-world width is
    estimated from the `width` tag when present, else `lanes` x 3.5m, else a
    highway-class default — then buffered to a polygon before rasterizing.
    Only bridge=* ways are fetched (ordinary at-grade roads don't rise above
    local terrain and can't false-positive the top-hat building filter).
    """
    empty = np.zeros((resolution, resolution), dtype=bool)
    try:
        # NOTE: osmnx/Overpass tag dicts with >1 key are OR'd, not AND'd —
        # {"bridge": [...], "highway": True} would match every highway=* way
        # regardless of bridge tag (measured: 6182/6480 matches had no bridge
        # tag at all). Query on "bridge" alone, then keep only rows that also
        # carry a highway=* tag (i.e. exclude bridge=yes on footways/railways
        # if desired — kept permissive here since all measured cases were
        # legitimate roads).
        gdf = ox.features_from_bbox(
            bbox=(W, S, E, N),
            tags={"bridge": ["yes", "viaduct", "aqueduct"]},
        )
    except Exception as e:
        logger.warning("OSM elevated-roadway fetch failed (%s); empty mask.", e)
        return empty

    if gdf is None or len(gdf) == 0:
        return empty
    ways = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if len(ways) == 0:
        return empty
    if "highway" in ways.columns:
        ways = ways[ways["highway"].notna()]
    if len(ways) == 0:
        return empty

    # Degrees-per-metre at this latitude, for buffering a geographic LineString
    # by a real-world half-width (buffer() operates in the geometry's own units).
    lat_c = (N + S) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(0.1, np.cos(np.radians(lat_c)))

    def _is_missing(v) -> bool:
        # OSM tag columns come back as pandas NA/NaN (a float) for empty
        # cells, not None — pd.isna handles both, and str/list values (which
        # pd.isna can't take directly) are never "missing" here.
        return v is None or (not isinstance(v, (str, list)) and bool(pd.isna(v)))

    def _width_m(row) -> float:
        w = row.get("width")
        if not _is_missing(w):
            try:
                v = float(str(w).split(";")[0].split()[0])
                if np.isfinite(v) and v > 0:
                    return v
            except (ValueError, IndexError):
                pass
        lanes = row.get("lanes")
        if not _is_missing(lanes):
            try:
                v = float(str(lanes).split(";")[0]) * 3.5
                if np.isfinite(v) and v > 0:
                    return v
            except (ValueError, IndexError):
                pass
        hwy = row.get("highway")
        hwy = hwy[0] if isinstance(hwy, list) else hwy
        return _HIGHWAY_DEFAULT_WIDTH_M.get(hwy, _HIGHWAY_FALLBACK_WIDTH_M)

    def _buffer(row):
        half_w_m = _width_m(row) / 2.0
        # Anisotropic degree-buffer approximating a real-world circular buffer:
        # buffer in lon-degrees using the lon-per-metre scale, matching the
        # N/S extent via a simple x/y scale correction.
        half_w_deg_lon = half_w_m / m_per_deg_lon
        scale_y = m_per_deg_lon / m_per_deg_lat
        from shapely.affinity import scale as _shp_scale
        g = row.geometry.buffer(half_w_deg_lon)
        return _shp_scale(g, xfact=1.0, yfact=scale_y, origin=row.geometry.centroid)

    def _safe_buffer(row):
        try:
            return _buffer(row)
        except Exception as exc:
            logger.debug("Skipping one bridge way (buffer failed: %s)", exc)
            return None

    ways["geometry"] = [
        _safe_buffer(row) for _, row in ways.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    ways = ways[ways.geometry.notna() & ~ways.geometry.is_empty]
    if len(ways) == 0:
        return empty

    ways["height_m"] = 1.0
    arr = _rasterize_buildings(ways, N, S, E, W, resolution)
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
