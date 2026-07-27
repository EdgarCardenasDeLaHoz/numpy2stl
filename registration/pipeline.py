"""numpy2stl.registration.pipeline — the city-STL→OSM registration orchestrator.

`register_city_stl` and its private stage helpers (`_simplify_stage`,
`_run_registration`, `_run_comparison`) live here; the package `__init__` is a thin
façade that re-exports the public entry points.  Import via the package::

    from numpy2stl.registration import register_city_stl
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default output root: a single gitignored repo-level _reports/ folder (NOT inside the
# package source tree).  _MODULE_ROOT = Code/numpy2stl/registration; parents[1] = Code/.
# Generated reports are regeneratable artifacts — never committed.  An explicit out_dir
# (e.g. --out into the data tree) overrides this.
_MODULE_ROOT = Path(__file__).parent
RUNS_DIR = _MODULE_ROOT.parents[1] / "_reports"


def _default_out_dir(region_name: str) -> Path:
    """Return the default output folder for a region: Code/_reports/<region-slug>/."""
    slug = re.sub(r"[^\w\-]+", "_", region_name).strip("_").lower()
    return RUNS_DIR / slug

from .align import apply_transform, register
from .compare import compare
from .html_report import write_registration_report
from .types import CityRegistrationReport, ComparisonResult, RegistrationResult

# Stage helpers extracted to the stages/ subpackage (B1 split).
from .stages import _simplify_stage, _run_registration, _run_comparison
from .stages._common import _inpaint_stl_nan, _landmark_check


def register_city_stl(
    stl_file: str,
    city_name: str | tuple,
    resolution: int = 1024,
    default_height: float = 10.0,
    levels_to_meters: float = 3.5,
    max_scale_ratio: float = 5.0,
    height_scale: float | None = None,
    stl_z_axis: int = 2,
    out_dir: str | Path | None = None,
    refine: bool = True,
    forced_scale: float | None = None,
    center: tuple[float, float] | None = None,
    tallest_m: float | None = None,
    scale_m_per_unit: float | None = None,
    detect_resolution_factor: int = 2,
    height_source: str = "osm",
    forced_rotation: float | None = None,
    simplify_mesh: bool = False,
    simplify_mode: str = "off",     # "off" | "decimate" | "prism"
    simplify_tol_m: float = 3.5,
    save_simplified: str | Path | None = None,
    regularize_footprints: bool = False,
    refine_polygons: bool = False,
    decimation_curve: bool = False,
    free_scale: bool = False,
    registration_method: str = "raster",   # "raster" | "polygon"
) -> CityRegistrationReport:
    """
    Full pipeline: STL → heightmap → OSM raster → register → compare → (report).

    Parameters
    ----------
    stl_file         : path to city STL (no geographic metadata required)
    city_name        : city name string (e.g. "Philadelphia, PA, USA") or
                       (N, S, E, W) bbox tuple for the OSM fetch
    resolution       : output grid size for the comparison/report heightmaps
                       (square, default 1024).  The registration SEARCH always
                       runs at REGISTER_RES (512) and is resolution-independent,
                       so raising this only sharpens the comparison/footprint
                       images (more agreement pixels) without changing the
                       transform or paying a larger search cost.
    default_height   : fallback building height for OSM (metres, default 10)
    levels_to_meters : OSM floors → metres conversion (default 3.5)
    max_scale_ratio  : maximum spatial scale search range (default 5.0)
    height_scale     : STL model units → metres. None = auto-estimate.
    stl_z_axis       : which mesh axis is elevation (default 2 = Z)
    out_dir          : output folder for HTML report + PNGs.
                       Defaults to the gitignored Code/_reports/{region}/
                       (mirrors the skyline runs/ convention).
                       Pass False to suppress all file output.

    Returns
    -------
    CityRegistrationReport dataclass with all intermediate results.
    HTML report is always written unless out_dir=False.
    """
    from ..stl2numpy.heightmap import mesh_to_heightmap
    from ..applications.cities import get_osm_building_heightmap

    step_timings: list[tuple[str, float]] = []

    def _timed(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        step_timings.append((name, time.perf_counter() - t0))
        logger.info("  %-28s %.2f s", name, step_timings[-1][1])
        return result

    logger.info("register_city_stl: %s  city=%s  resolution=%d", stl_file, city_name, resolution)

    # `eff_stl_file` is the mesh actually rasterized for segmentation/registration.
    # It becomes the simplified mesh once the model scale is known (see Stage 0
    # below — the metres deviation budget must be converted to mesh units first).
    eff_stl_file = stl_file      # mesh actually rasterized downstream (set by Stage 0 below)

    # 1. STL → heightmap (original mesh; gives bounds for the scale anchor)
    stl_result = _timed(
        "Load STL + mesh_to_heightmap",
        mesh_to_heightmap,
        stl_file,
        resolution=resolution,
        projection="max",
        z_axis=stl_z_axis,
        isotropic=True,   # square pixels → no aspect stretch vs isotropic OSM
    )
    stl_hm = _inpaint_stl_nan(stl_result["heightmap"])

    # 1b. Estimate a tight OSM bbox from the STL's scale.
    # The STL's tallest feature (z_max) anchors the model→metres scale:
    #   scale = city tallest_m / stl_z_max ;  footprint_m = stl_xy_extent * scale
    # This avoids fetching the entire 25 km city for a ~1 km model.
    from ..applications.cities import estimate_bbox_from_stl

    bx = stl_result["bounds"]["x"]
    by = stl_result["bounds"]["y"]
    bz = stl_result["bounds"]["z"]
    stl_z_max = float(bz[1] - bz[0])
    stl_xy_extent = max(float(bx[1] - bx[0]), float(by[1] - by[0]))

    # Fetch OSM at 1.5x the STL footprint. Larger frames shrink the STL within
    # the grid and weaken the base registration; 1.5x balances surrounding
    # context against keeping the STL large enough to register reliably.
    from .config import DEFAULT_OSM_MARGIN, M_PER_DEG_LAT
    osm_margin = DEFAULT_OSM_MARGIN
    osm_fetch_target = city_name
    known_scale = None
    geometric_anchor = None   # 1/osm_margin — kept for the report's anchor line
    if isinstance(city_name, str):
        tight_bbox = estimate_bbox_from_stl(
            city_name, stl_z_max, stl_xy_extent, osm_margin=osm_margin,
            center=center, tallest_m=tallest_m, scale_m_per_unit=scale_m_per_unit,
        )
        if tight_bbox is not None:
            osm_fetch_target = tight_bbox
            # OSM frame is osm_margin × the STL footprint, both rendered at the
            # same resolution → the STL fills 1/osm_margin of the OSM frame.
            geometric_anchor = 1.0 / osm_margin
            known_scale = geometric_anchor

    # Scale is finalized AFTER the OSM fetch from estimate_scale() (area + Fourier),
    # which is reliable now that the STL is rendered isotropically (square pixels).
    # The geometric anchor 1/osm_margin set above is only a provisional fallback.
    # A manual --scale always wins.
    forced_scale_val = float(forced_scale) if forced_scale is not None else None
    if forced_scale_val is not None:
        known_scale = forced_scale_val

    # 0. Mesh simplification (now that the model scale is known).  See _simplify_stage:
    # "decimate" replaces the mesh + re-renders stl_hm; "prism" leaves the registration
    # mesh and produces a prism heightmap + footprint polygons for the comparison.
    _simpl = _simplify_stage(
        stl_file, city_name, stl_z_max, tallest_m, scale_m_per_unit, stl_hm,
        simplify_mode=simplify_mode, simplify_mesh=simplify_mesh, simplify_tol_m=simplify_tol_m,
        save_simplified=save_simplified, decimation_curve=decimation_curve,
        resolution=resolution, stl_z_axis=stl_z_axis, mesh_to_heightmap=mesh_to_heightmap,
        timed=_timed)
    eff_stl_file = _simpl["eff_stl_file"]
    stl_hm = _simpl["stl_hm"]
    _stl_hm_original = _simpl["stl_hm_original"]
    _simplify_stats_dict = _simpl["simplify_stats"]
    _prism_stats_dict = _simpl["prism_stats"]
    _prism_polys = _simpl["prism_polys"]
    _prism_hm_render = _simpl["prism_hm_render"]
    _decimation_sweep = _simpl["decimation_sweep"]

    # 2. OSM building heights (over the tight bbox when available)
    osm_result = _timed(
        "Fetch OSM building heights",
        get_osm_building_heightmap,
        osm_fetch_target,
        resolution=resolution,
        default_height=default_height,
        levels_to_meters=levels_to_meters,
    )
    osm_hm = osm_result["heightmap"]

    # 2b. Metres-per-pixel of the OSM grid (the comparison grid). Used to size the
    # terrain top-hat kernel in physical units so the building-height residual is
    # consistent across resolutions.
    import math as _math
    (W_, E_), (S_, N_) = osm_result["bounds"]["x"], osm_result["bounds"]["y"]
    dlon = (E_ - W_) / osm_hm.shape[1]
    dlat = (N_ - S_) / osm_hm.shape[0]
    lat_c = (N_ + S_) / 2.0
    dx_m = abs(dlon) * M_PER_DEG_LAT * _math.cos(_math.radians(lat_c))
    dy_m = abs(dlat) * M_PER_DEG_LAT
    cell_size_m = float((dx_m + dy_m) / 2.0)
    logger.info("OSM grid cell size: %.2f m/px (%.2f x %.2f)", cell_size_m, dx_m, dy_m)

    # 2b1. Measured-height source: replace per-footprint OSM tag heights with the
    # median lidar nDSM inside each footprint (fixes the 10 m fill + levels guesses
    # + untagged buildings).  No-ops gracefully to OSM tags when deps/coverage are
    # missing.  Footprint geometry (the OSM mask) is unchanged — only the heights.
    if str(height_source).lower() == "lidar":
        try:
            from ..applications.lidar import get_ndsm
            ndsm = _timed("Fetch lidar nDSM", get_ndsm, (N_, S_, E_, W_), resolution=resolution)
            if ndsm is not None and ndsm.shape == osm_hm.shape:
                import cv2 as _cv2
                bm = (~np.isnan(osm_hm)).astype(np.uint8)
                ncc, lab = _cv2.connectedComponents(bm, connectivity=8)
                replaced = 0
                for i in range(1, ncc):
                    comp = lab == i
                    med = float(np.nanmedian(ndsm[comp]))
                    if np.isfinite(med) and med > 0:
                        osm_hm[comp] = med
                        replaced += 1
                logger.info("Height source = lidar nDSM: replaced %d/%d OSM footprint heights",
                            replaced, ncc - 1)
            else:
                logger.info("Height source = lidar requested but nDSM unavailable; using OSM tags.")
        except Exception as _exc:
            logger.warning("Lidar height source failed (%s); using OSM tags.", _exc)

    # 2b2. Fetch vegetation/water/elevated-roadway semantic masks on the same
    # grid — used to exclude trees/water/overpasses misread as STL buildings
    # before comparison. Elevated highways/bridges rise above local terrain in
    # the STL the same way a building does (measured false-positive source,
    # e.g. Miami's elevated ramps segmenting as "buildings"), but OSM has no
    # building footprint there to match against, so they're excluded rather
    # than compared.
    try:
        from ..applications.cities import get_osm_semantic_masks
        sem = _timed("Fetch OSM semantic masks", get_osm_semantic_masks,
                     osm_fetch_target, resolution=resolution)
        veg_mask = sem["vegetation"]
        water_mask = sem["water"]
        elevated_roadway_mask = sem.get("elevated_roadway")
    except Exception as _exc:
        logger.warning("OSM semantic masks unavailable (%s); skipping exclusion.", _exc)
        veg_mask = water_mask = elevated_roadway_mask = None

    # 2b3. Resolution-independent registration inputs.  Run the SEARCH at a fixed
    # canonical resolution (REGISTER_RES) so the transform is identical for any
    # output `resolution` and a 1k output doesn't pay a 4x search cost.  Comparison
    # and segmentation below still use the full-resolution stl_hm/osm_hm.
    from .config import REGISTER_RES
    if resolution == REGISTER_RES:
        stl_reg, osm_reg, cell_size_m_reg = stl_hm, osm_hm, cell_size_m
    else:
        stl_reg = _inpaint_stl_nan(mesh_to_heightmap(
            eff_stl_file, resolution=REGISTER_RES, projection="max",
            z_axis=stl_z_axis, isotropic=True)["heightmap"])
        osm_reg = get_osm_building_heightmap(
            osm_fetch_target, resolution=REGISTER_RES,
            default_height=default_height, levels_to_meters=levels_to_meters)["heightmap"]
        cell_size_m_reg = cell_size_m * (float(resolution) / REGISTER_RES)
        logger.info("Registration search at %dx%d (output %dx%d); transform scaled by %.3f",
                    REGISTER_RES, REGISTER_RES, resolution, resolution,
                    float(resolution) / REGISTER_RES)

    # 2c. Finalize the scale from estimate_scale() (area-ratio + Fourier profile).
    # With isotropic STL rendering the area ratio is trustworthy (a world-square
    # building is square in both rasters), so we use the data-driven estimate
    # rather than a hardcoded constant.  Manual --scale and the geometric anchor
    # (when no estimate is available) remain as overrides/fallback.
    scale_estimate = None
    if known_scale is not None:
        from .align import estimate_scale
        est = estimate_scale(stl_reg, osm_reg)   # report diagnostics only
        scale_estimate = est
        if forced_scale_val is None:
            # The scale is GEOMETRICALLY determined: the OSM frame is fetched at
            # osm_margin × the STL footprint and BOTH are rendered at the same
            # resolution, so the STL fills exactly 1/osm_margin of the frame.
            # Trust that anchor.  estimate_scale (area-ratio + Fourier) is kept for
            # the report only — it drifts on dense city grids (here area=1.25,
            # fourier=1.67) and was previously, wrongly, overriding the exact value.
            logger.info("Scale: geometric anchor=%.3f (1/osm_margin)  "
                        "[estimate area=%.3f fourier=%.3f corr %.2f — report only]",
                        known_scale, est["area_scale"], est["fourier_scale"],
                        est["fourier_corr"])
        else:
            logger.info("Scale: manual override=%.3f  [estimate area=%.3f fourier=%.3f]",
                        known_scale, est["area_scale"], est["fourier_scale"])

    # 3. Register (raster or polygon point-pattern), projection discovery + ECC
    # refine, then scale the transform to the output grid.  See _run_registration.
    _reg = _run_registration(
        stl_reg, osm_reg, prism_polys=_prism_polys, bx=bx, by=by,
        cell_size_m_reg=cell_size_m_reg, known_scale=known_scale,
        max_scale_ratio=max_scale_ratio, forced_rotation=forced_rotation,
        free_scale=free_scale, registration_method=registration_method,
        refine=refine, resolution=resolution, timed=_timed, step_timings=step_timings)
    reg_result        = _reg["reg_result"]
    transform         = _reg["transform"]
    chosen_projection = _reg["chosen_projection"]
    reg_dict          = _reg["reg_dict"]
    # Report diagnostics (sweeps / histograms / xcorr) come straight from reg_dict.
    scale_sweep    = reg_dict.get("scale_sweep",    [])
    rot_sweep      = reg_dict.get("rot_sweep",      [])
    hist_src       = reg_dict.get("hist_src",       None)
    hist_tgt       = reg_dict.get("hist_tgt",       None)
    hist_xcorr     = reg_dict.get("hist_xcorr",     None)
    hist_rot_deg   = reg_dict.get("hist_rot_deg",   0.0)
    xcorr_map      = reg_dict.get("xcorr_map",      None)
    best_dx        = reg_dict.get("best_dx",        0.0)
    best_dy        = reg_dict.get("best_dy",        0.0)
    rot_l1_xcorr   = reg_dict.get("rot_l1_xcorr",  None)
    rot_l2_xcorr   = reg_dict.get("rot_l2_xcorr",  None)
    dice_fine      = reg_dict.get("dice_fine",      None)

    # Prism mode: registration used the ORIGINAL heightmap; the COMPARISON uses the
    # prism-model heightmap (same world coords → same transform).
    if _prism_hm_render is not None:
        stl_hm = _prism_hm_render

    # 4–5. Warp + mask + height comparison, optional polygon-ICP, and footprint
    # polygons.  See _run_comparison (stl_hm is the prism render in prism mode).
    _cmp = _run_comparison(
        stl_hm, osm_hm, reg_result, cell_size_m=cell_size_m, height_scale=height_scale,
        veg_mask=veg_mask, water_mask=water_mask, elevated_roadway_mask=elevated_roadway_mask,
        refine_polygons=refine_polygons,
        reg_dict=reg_dict, chosen_projection=chosen_projection, prism_polys=_prism_polys,
        bx=bx, by=by, resolution=resolution, eff_stl_file=eff_stl_file,
        stl_z_axis=stl_z_axis, detect_resolution_factor=detect_resolution_factor,
        regularize_footprints=regularize_footprints,
        mesh_to_heightmap=mesh_to_heightmap, timed=_timed, step_timings=step_timings)
    reg_result        = _cmp["reg_result"]
    stl_aligned       = _cmp["stl_aligned"]
    stl_building_mask = _cmp["stl_building_mask"]
    stl_residual      = _cmp["stl_residual"]
    stl_for_compare   = _cmp["stl_for_compare"]
    comp_result       = _cmp["comp_result"]
    stl_polygons      = _cmp["stl_polygons"]

    # 5b. Landmark sanity check
    landmark = _landmark_check(osm_hm, transform)
    if landmark:
        logger.info("Landmark check: OSM landmark at (%.0f,%.0f) h=%.0fm → "
                    "STL (%.0f,%.0f) %.1fpx from STL centre",
                    landmark["osm_row"], landmark["osm_col"], landmark["osm_mean_h"],
                    landmark["stl_row"], landmark["stl_col"], landmark["stl_dist_from_center"])

    # 6. Assemble report
    region_name = city_name if isinstance(city_name, str) else f"bbox {city_name}"
    report = CityRegistrationReport(
        region_name=region_name,
        stl_file=str(stl_file),
        stl_heightmap=stl_hm,
        osm_heightmap=osm_hm,
        stl_aligned=stl_aligned,
        registration=reg_result,
        comparison=comp_result,
        step_timings=step_timings,
        osm_bbox=(N_, S_, E_, W_),
        cell_size_m=cell_size_m,
        stl_building_mask=stl_building_mask,
        scale_sweep=tuple(scale_sweep),
        rot_sweep=tuple(rot_sweep),
        _hist_src=hist_src,
        _hist_tgt=hist_tgt,
        _hist_xcorr=hist_xcorr,
        _hist_rot_deg=float(hist_rot_deg),
        _xcorr_map=xcorr_map,
        _best_dx=float(best_dx),
        _best_dy=float(best_dy),
        known_scale=geometric_anchor,
        landmark_check=landmark,
        _rot_l1_xcorr=rot_l1_xcorr,
        _rot_l2_xcorr=rot_l2_xcorr,
        _dice_fine=dice_fine,
        area_scale=float(scale_estimate["area_scale"]) if scale_estimate else None,
        fourier_scale=(float(scale_estimate["fourier_scale"])
                       if scale_estimate and np.isfinite(scale_estimate["fourier_scale"])
                       else None),
        _stl_polygons=stl_polygons,
        _stl_heightmap_original=_stl_hm_original,
        _simplify_stats=_simplify_stats_dict,
        _decimation_sweep=_decimation_sweep,
        _prism_stats=_prism_stats_dict,
    )

    # 7. Write HTML report (default ON, pass out_dir=False to suppress)
    if out_dir is not False:
        if out_dir is None:
            out_dir = _default_out_dir(region_name)
        _timed(
            "Write HTML report",
            write_registration_report,
            out_dir,
            report,
        )
        index = Path(out_dir) / "index.html"
        logger.info("Report written -> %s", index)
        print(f"\nReport -> {index}")

    return report
