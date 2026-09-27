"""Centre search: find the OSM fetch centre whose buildings match a city STL.

``find_best_city_center`` scores a ring of candidate centres with a coarse
``register_global`` pass and keeps the best one that locks.  It lives here, not
in ``applications.cities``, so ``applications`` never imports ``registration``;
the OSM fetch helpers are looked up on ``applications.cities`` at call time.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def find_best_city_center(
    city_name: str,
    stl_z_max: float,
    stl_xy_extent: float,
    stl_hm: np.ndarray,
    tallest_m: float | None = None,
    scale_m_per_unit: float | None = None,
    initial_center: tuple[float, float] | None = None,
    search_radius_km: float = 4.0,
    search_step_km: float = 1.0,
    osm_margin: float = 1.5,
    probe_resolution: int = 256,
) -> dict:
    """Search a ring grid of candidate (lat, lon) centres around `initial_center`
    (or `get_city_center_point(city_name)` when omitted) for the one whose OSM
    building fetch actually matches the STL — a fallback for when the single
    "Downtown {city}" geocode lands in the wrong part of a partial-coverage city
    model (see cities.get_city_center_point()'s docstring: measured wrong for Barcelona,
    Paris, Lisbon and likely Bilbao — a plausible-sounding landmark that isn't
    where THIS particular STL crop actually is).

    Candidate grid: concentric rings at radii {1, 2, ..., search_radius_km} km
    (step `search_step_km`) x 8 compass directions, PLUS `initial_center` itself
    -- up to 8 * (search_radius_km / search_step_km) + 1 candidates (33 at the
    defaults). km offsets are converted to lat/lon deltas using the ACTUAL
    initial_center's latitude for the longitude cosine correction (1 deg lat ~=
    111 km; 1 deg lon ~= 111 km * cos(lat)) -- not a fixed assumed latitude,
    since candidates span multiple cities across a wide latitude range.

    Each candidate is scored by a COARSE, cheap register_global() pass (probe
    resolution 256, osm_margin=1.5 -- matched to production's DEFAULT_OSM_MARGIN,
    NOT a wider fetch, to keep per-candidate cost low: measured ~5-11s/candidate
    for the OSM building fetch alone at this margin+resolution vs. 55-205s at a
    wider margin + semantic masks, which this probe does not need -- rotation/
    scale locking only reads the returned scale_sweep/rot_sweep, both independent
    of vegetation/water/elevated-roadway exclusion). A candidate is `is_locked`
    when BOTH its Dice-vs-scale and edge-IoU-vs-rotation sweeps show a sharp,
    non-boundary peak -- see stages._common._is_locked_registration(), which
    this function shares with the pipeline's own coarse-probe gate so "locked"
    means the same thing in both places.

    Among locked candidates, the winner is the one with the highest `edge_iou`
    (register_global()'s own best-available alignment-quality signal at that
    pose). When NO candidate locks, this returns `resolved=False` and does NOT
    guess a best-of-N winner from unlocked (noisy) candidates -- the caller
    should fall back to `initial_center`.

    A per-candidate OSM-fetch failure (network error, Overpass timeout) is
    logged and treated as "this candidate does not lock", not a hard crash --
    consistent with how cities.get_osm_building_heightmap()'s own callers already
    tolerate fetch failures.

    Returns
    -------
    dict: {
        "center": (lat, lon) | None,        # winning candidate, or None if unresolved
        "resolved": bool,
        "candidates_tried": int,
        "best_dice_sharpness": float,
        "best_rot_sharpness": float,
        "reg_dict": dict | None,            # winning candidate's register_global() dict
        "osm_fetch_target": tuple | None,   # winning candidate's (N, S, E, W) tight bbox
    }
    """
    import math

    from ..applications import cities
    from .align import register_global
    from .config import CENTER_SEARCH_DICE_MARGIN, CENTER_SEARCH_ROT_MARGIN
    from .stages import _is_locked_registration

    if initial_center is None:
        initial_center = cities.get_city_center_point(city_name)
    if initial_center is None:
        logger.warning("find_best_city_center(%r): no initial_center and geocoding "
                       "failed; cannot search.", city_name)
        return {
            "center": None, "resolved": False, "candidates_tried": 0,
            "best_dice_sharpness": 0.0, "best_rot_sharpness": 0.0,
            "reg_dict": None, "osm_fetch_target": None,
        }

    base_lat, base_lon = float(initial_center[0]), float(initial_center[1])
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * math.cos(math.radians(base_lat))

    # Candidate grid: initial_center + rings of 8 compass directions.
    candidates: list[tuple[float, float]] = [(base_lat, base_lon)]
    radii_km = np.arange(search_step_km, search_radius_km + 1e-9, search_step_km)
    n_dirs = 8
    for r_km in radii_km:
        r_m = float(r_km) * 1000.0
        for k in range(n_dirs):
            theta = 2.0 * math.pi * k / n_dirs
            d_lat = (r_m * math.cos(theta)) / m_per_deg_lat
            d_lon = (r_m * math.sin(theta)) / m_per_deg_lon
            candidates.append((base_lat + d_lat, base_lon + d_lon))

    logger.info("find_best_city_center(%r): searching %d candidates around "
                "(%.4f, %.4f) (radius %.0fkm, step %.0fkm)",
                city_name, len(candidates), base_lat, base_lon,
                search_radius_km, search_step_km)

    # Resize the STL heightmap once to the probe resolution (register_global
    # itself will also resize `source` if shapes mismatch, but doing it once up
    # front avoids repeating a possibly-large resize per candidate).
    stl_probe = stl_hm
    if stl_hm.shape != (probe_resolution, probe_resolution):
        import cv2 as _cv2
        _filled = np.nan_to_num(stl_hm.astype(np.float32), nan=0.0)
        stl_probe = _cv2.resize(_filled, (probe_resolution, probe_resolution),
                                interpolation=_cv2.INTER_LINEAR)

    best: dict | None = None
    best_locked: dict | None = None
    for idx, (clat, clon) in enumerate(candidates):
        offset_desc = "initial" if idx == 0 else f"#{idx} ({clat:.4f},{clon:.4f})"
        try:
            bbox = cities.estimate_bbox_from_stl(
                city_name, stl_z_max, stl_xy_extent, osm_margin=osm_margin,
                center=(clat, clon), tallest_m=tallest_m,
                scale_m_per_unit=scale_m_per_unit,
            )
            if bbox is None:
                logger.info("  candidate %s: no bbox (no scale anchor); skipping", offset_desc)
                continue
            osm = cities.get_osm_building_heightmap(bbox, resolution=probe_resolution, cache=True)
            geometric_anchor = 1.0 / osm_margin
            reg_dict = register_global(
                stl_probe, osm["heightmap"], scale_prior=geometric_anchor, scale_search=0.0,
            )
        except Exception as exc:
            logger.warning("  candidate %s: fetch/register failed (%s); treating as not locked",
                           offset_desc, exc)
            continue

        lock = _is_locked_registration(
            reg_dict, dice_margin=CENTER_SEARCH_DICE_MARGIN, rot_margin=CENTER_SEARCH_ROT_MARGIN,
        )
        logger.info("  candidate %s: dice_sharpness=%.3f rot_sharpness=%.3f "
                    "edge_iou=%.3f locked=%s", offset_desc, lock["dice_sharpness"],
                    lock["rot_sharpness"], reg_dict.get("edge_iou", 0.0), lock["is_locked"])

        entry = {
            "center": (clat, clon), "bbox": bbox, "reg_dict": reg_dict,
            "dice_sharpness": lock["dice_sharpness"], "rot_sharpness": lock["rot_sharpness"],
            "is_locked": lock["is_locked"], "edge_iou": float(reg_dict.get("edge_iou", 0.0)),
        }
        if best is None or entry["edge_iou"] > best["edge_iou"]:
            best = entry
        if lock["is_locked"] and (best_locked is None or entry["edge_iou"] > best_locked["edge_iou"]):
            best_locked = entry

    candidates_tried = len(candidates)
    if best_locked is None:
        logger.warning("find_best_city_center(%r): 0/%d candidates locked; falling back "
                       "to initial_center (%.4f, %.4f).", city_name, candidates_tried,
                       base_lat, base_lon)
        return {
            "center": initial_center, "resolved": False, "candidates_tried": candidates_tried,
            "best_dice_sharpness": float(best["dice_sharpness"]) if best else 0.0,
            "best_rot_sharpness": float(best["rot_sharpness"]) if best else 0.0,
            "reg_dict": None, "osm_fetch_target": None,
        }

    logger.info("find_best_city_center(%r): resolved to (%.4f, %.4f) "
                "(edge_iou=%.3f, dice_sharpness=%.3f, rot_sharpness=%.3f) after %d candidates",
                city_name, best_locked["center"][0], best_locked["center"][1],
                best_locked["edge_iou"], best_locked["dice_sharpness"],
                best_locked["rot_sharpness"], candidates_tried)
    return {
        "center": best_locked["center"], "resolved": True, "candidates_tried": candidates_tried,
        "best_dice_sharpness": float(best_locked["dice_sharpness"]),
        "best_rot_sharpness": float(best_locked["rot_sharpness"]),
        "reg_dict": best_locked["reg_dict"], "osm_fetch_target": best_locked["bbox"],
    }
