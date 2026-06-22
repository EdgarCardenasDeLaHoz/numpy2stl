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


def _decompose_for_report(M: np.ndarray) -> tuple[float, float]:
    """Extract (scale, angle_deg) from a 2x3 affine or 3x3 homography matrix."""
    import math
    scale = float(math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    angle = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
    return scale, angle


def _crop_osm_to_stl_extent(
    stl_hm: np.ndarray,
    osm_hm: np.ndarray,
    margin: float = 1.5,
) -> np.ndarray:
    """
    Crop OSM to at most margin × STL's non-NaN pixel extent, centred.

    When the STL model covers a small portion of the full city OSM
    (e.g. a neighbourhood model vs whole-city raster), this prevents ECC
    from searching an irrelevantly large space.
    """
    valid = ~np.isnan(stl_hm)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return osm_hm

    stl_r = rows[-1] - rows[0] + 1
    stl_c = cols[-1] - cols[0] + 1

    osm_h, osm_w = osm_hm.shape
    crop_h = min(osm_h, max(64, int(stl_r * margin)))
    crop_w = min(osm_w, max(64, int(stl_c * margin)))

    if crop_h >= osm_h and crop_w >= osm_w:
        return osm_hm  # already within bounds — nothing to do

    r0 = max(0, (osm_h - crop_h) // 2)
    c0 = max(0, (osm_w - crop_w) // 2)
    r1 = min(osm_h, r0 + crop_h)
    c1 = min(osm_w, c0 + crop_w)

    logger.info(
        "Cropped OSM %dx%d -> %dx%d  (1.5x STL non-NaN extent %dx%d)",
        osm_h, osm_w, r1 - r0, c1 - c0, stl_r, stl_c,
    )
    return osm_hm[r0:r1, c0:c1]

from .align import apply_transform, register
from .compare import compare
from .html_report import write_registration_report
from .types import CityRegistrationReport, ComparisonResult, RegistrationResult


def _landmark_check(osm_hm: np.ndarray, transform: np.ndarray) -> dict | None:
    """
    Sanity-check registration against the largest tall building near the OSM
    bbox centre (City Hall for Philadelphia, or the dominant central landmark
    for any city whose bbox is centred on a known building).

    The OSM bbox is always centred on a known geographic point (e.g. City Hall),
    so the expected OSM landmark pixel = (cx, cy) = bbox centre ≈ (256, 256).
    We find the largest building component within 80px of the centre that has
    mean height > 40m, which is the best proxy for that landmark in the OSM data.

    After applying the inverse of the registration transform we can see where that
    landmark sits in the STL coordinate frame and report the residual error.
    """
    try:
        import cv2
        h, w = osm_hm.shape
        cy, cx = h // 2, w // 2

        from .align import building_mask
        osm_mask = building_mask(osm_hm, source="osm").astype(np.uint8)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(osm_mask, connectivity=8)

        best = None
        for i in range(1, n):
            ccx, ccy = centroids[i]
            dist = float(((ccx - cx) ** 2 + (ccy - cy) ** 2) ** 0.5)
            if dist > 80:
                continue
            area = int(stats[i, cv2.CC_STAT_AREA])
            mean_h = float(np.nanmean(osm_hm[labels == i]))
            if mean_h < 40 or area < 100:
                continue
            if best is None or area > best["area"]:
                best = {"area": area, "osm_col": float(ccx), "osm_row": float(ccy),
                        "mean_h": mean_h, "dist_from_center": dist}

        if best is None:
            return None

        # Map the OSM landmark pixel back into STL space via the inverse transform
        M3 = np.eye(3)
        M3[:2] = transform
        M_inv = np.linalg.inv(M3)
        ox, oy = best["osm_col"], best["osm_row"]
        stl_col = M_inv[0, 0] * ox + M_inv[0, 1] * oy + M_inv[0, 2]
        stl_row = M_inv[1, 0] * ox + M_inv[1, 1] * oy + M_inv[1, 2]

        # Pixel distance from STL centre (the model should be centred too)
        stl_cx, stl_cy = w / 2.0, h / 2.0
        stl_dist = float(((stl_col - stl_cx) ** 2 + (stl_row - stl_cy) ** 2) ** 0.5)

        return {
            "osm_row": best["osm_row"], "osm_col": best["osm_col"],
            "osm_dist_from_center": best["dist_from_center"],
            "osm_mean_h": best["mean_h"],
            "stl_row": stl_row, "stl_col": stl_col,
            "stl_dist_from_center": stl_dist,
        }
    except Exception as exc:
        logger.debug("landmark_check failed: %s", exc)
        return None


def _inpaint_stl_nan(hm: np.ndarray) -> np.ndarray:
    """
    Fill NaN holes in the STL heightmap by nearest-neighbour interpolation.

    The STL mesh doesn't cover every grid cell (gaps at boundary, thin walls,
    mesh holes). NaN pixels would propagate into the morphological top-hat and
    edge detection as false edges.  Nearest-neighbour fill is fast and correct
    for this use: we just need a plausible height so the terrain model is
    continuous, not a precise interpolated value.
    """
    nan_mask = np.isnan(hm)
    if not nan_mask.any():
        return hm
    try:
        from scipy.ndimage import distance_transform_edt, label

        # Only fill *interior* holes — NaN regions enclosed by the mesh.  NaN that
        # is connected to the image border is exterior padding (added when the
        # model is rendered isotropically into a square canvas) and must stay NaN,
        # otherwise nearest-neighbour fill would smear building heights into the
        # empty margin and re-introduce a stretch-like artefact.
        structure = np.ones((3, 3), dtype=int)
        lbl, n_lbl = label(nan_mask, structure=structure)
        border_ids = set(np.unique(np.concatenate([
            lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])).tolist())
        border_ids.discard(0)
        exterior = np.isin(lbl, list(border_ids)) if border_ids else np.zeros_like(nan_mask)
        interior = nan_mask & ~exterior

        filled = hm.copy()
        if interior.any():
            _, idx = distance_transform_edt(nan_mask, return_indices=True)
            interior_idx = (idx[0][interior], idx[1][interior])
            filled[interior] = hm[interior_idx]
        logger.debug(
            "STL inpaint: filled %d interior NaN px; left %d exterior padding px",
            int(interior.sum()), int(exterior.sum()))
        return filled
    except ImportError:
        # scipy not available — median fill as last resort
        median_val = float(np.nanmedian(hm))
        filled = hm.copy()
        filled[nan_mask] = median_val
        return filled

def _polygon_register_dict(stl_reg, osm_reg, known_scale, prism_polys, bx, by, cell_size_m_reg):
    """Polygon point-pattern registration → a reg_dict like register_global's.

    Builds STL footprint polygons (prism polygons mapped to REGISTER_RES, else a
    triangle-threshold vectorization of stl_reg) and OSM footprint polygons, then
    runs `register_polygons` with scale pinned to the geometric anchor.  Returns
    None (→ caller falls back to raster) when the match confidence is low.
    """
    import cv2 as _cv2
    from .align import (register_polygons, vectorize_buildings, building_mask)
    from .config import REGISTER_RES as _RR
    h, w = osm_reg.shape

    if prism_polys:                              # map [0,1] fractions → iso pixels
        xe = float(bx[1] - bx[0]); ye = float(by[1] - by[0])
        if xe >= ye:
            cols = _RR; rows = max(1, int(round(_RR * ye / xe)))
        else:
            rows = _RR; cols = max(1, int(round(_RR * xe / ye)))
        c0 = (_RR - cols) // 2; r0 = (_RR - rows) // 2
        stl_polys = [np.column_stack([c0 + p[:, 0] * cols, r0 + p[:, 1] * rows]) for p in prism_polys]
    else:
        stl_polys = vectorize_buildings(
            building_mask(stl_reg, source="stl", cell_size_m=cell_size_m_reg,
                          threshold_method="triangle", split_watershed=True), regularize=True)
    osm_polys = vectorize_buildings(building_mask(osm_reg, source="osm"))

    res = register_polygons(stl_polys, osm_polys, scale_prior=known_scale)
    logger.info("Polygon registration: %s (%d inliers, conf %.2f, scale=%.3f rot=%.2f°)",
                res["reason"], res["n_inliers"], res["confidence"], res["scale"], res["angle_deg"])
    if not res["applied"]:
        return None
    return {
        "transform": res["transform"], "confidence": float(res["confidence"]),
        "scale": float(res["scale"]), "angle_deg": float(res["angle_deg"]),
        "n_iterations": 0, "converged": True,
        "substep_timings": [], "scale_sweep": [], "rot_sweep": [],
    }


def _simplify_stage(stl_file, city_name, stl_z_max, tallest_m, scale_m_per_unit, stl_hm,
                    *, simplify_mode, simplify_mesh, simplify_tol_m, save_simplified,
                    decimation_curve, resolution, stl_z_axis, mesh_to_heightmap, timed):
    """Stage 0 — optional mesh simplification (decimate | prism).

    The deviation budget is in METRES but the mesh is in its own units, so it is
    converted via the model scale (tol_units = simplify_tol_m / m_per_unit).
    "decimate" replaces `eff_stl_file` + re-renders `stl_hm`; "prism" leaves the
    registration mesh alone (base-plate anchor) and produces a prism heightmap +
    footprint polygons for the COMPARISON.  Returns an effects dict; on any failure
    it returns the unchanged inputs.  `simplify_mesh=True` aliases mode "decimate".
    """
    out = {"eff_stl_file": stl_file, "stl_hm": stl_hm, "stl_hm_original": None,
           "simplify_stats": None, "prism_stats": None, "prism_polys": None,
           "prism_hm_render": None, "decimation_sweep": None}
    mode = simplify_mode if simplify_mode != "off" else ("decimate" if simplify_mesh else "off")
    if mode == "off":
        return out
    try:
        from ..applications.cities import derive_scale_m_per_unit
        import tempfile, os as _os
        m_per_unit = derive_scale_m_per_unit(
            city_name, stl_z_max, tallest_m=tallest_m, scale_m_per_unit=scale_m_per_unit)
        if not m_per_unit or m_per_unit <= 0:
            m_per_unit = 1.0
            logger.warning("Simplify: no scale anchor; treating deviation budget "
                           "as mesh units (%.2f).", simplify_tol_m)
        tol_units = float(simplify_tol_m) / float(m_per_unit)
        stem = Path(stl_file).stem

        if mode == "prism":
            from ..processing.building_simplify import prism_decompose
            out_simpl = (str(save_simplified) if save_simplified is not None
                         else _os.path.join(tempfile.gettempdir(), f"prism_{stem}.stl"))
            _, _, pstats, _, prism_polys = timed(
                "Prism decomposition", prism_decompose, str(stl_file),
                deviation_tol=tol_units, z_axis=stl_z_axis, resolution=resolution,
                m_per_unit=m_per_unit, save_path=out_simpl)
            if pstats and pstats.backend == "prism" and Path(out_simpl).exists():
                logger.info("Prism model: %s (%d buildings → %d prisms, mean %.1f layers, "
                            "%d sloped caps, deviation≈%.2f m)", out_simpl, pstats.n_buildings,
                            pstats.n_prisms, pstats.mean_layers, pstats.sloped_caps, pstats.hausdorff_m)
                out["stl_hm_original"] = stl_hm.copy()
                out["prism_polys"] = prism_polys
                pd = dict(pstats._asdict())
                pd["m_per_unit"] = float(m_per_unit)
                pd["deviation_tol_m_metres"] = float(simplify_tol_m)
                out["prism_stats"] = pd
                # Registration stays on the ORIGINAL mesh (base-plate anchor); the
                # prism model (ground filled flat, NaN→0) feeds the comparison only.
                _raw = timed("Re-render prism heightmap", mesh_to_heightmap, out_simpl,
                             resolution=resolution, projection="max", z_axis=stl_z_axis,
                             isotropic=True, cache=False)["heightmap"]
                out["prism_hm_render"] = np.nan_to_num(_raw, nan=0.0)
        else:  # "decimate"
            from ..processing.building_simplify import simplify_building_mesh
            out_simpl = (str(save_simplified) if save_simplified is not None
                         else _os.path.join(tempfile.gettempdir(), f"simplified_{stem}.stl"))
            _, _, sstats = timed("Simplify mesh (decimate)", simplify_building_mesh,
                                 str(stl_file), deviation_tol_m=tol_units,
                                 z_axis=stl_z_axis, save_path=out_simpl)
            if sstats and sstats.backend != "none" and Path(out_simpl).exists():
                out["eff_stl_file"] = out_simpl
                logger.info("Using simplified mesh: %s (%d→%d faces, %.1f%%, deviation "
                            "%.3f units = %.2f m of %.2f m budget)", out_simpl, sstats.orig_faces,
                            sstats.simplified_faces, 100.0 * sstats.face_ratio, sstats.hausdorff_m,
                            sstats.hausdorff_m * m_per_unit, simplify_tol_m)
                out["stl_hm_original"] = stl_hm.copy()
                sd = dict(sstats._asdict())
                sd["m_per_unit"] = float(m_per_unit)
                sd["hausdorff_units"] = sstats.hausdorff_m
                sd["hausdorff_m"] = sstats.hausdorff_m * m_per_unit
                sd["deviation_tol_m_metres"] = float(simplify_tol_m)
                out["simplify_stats"] = sd
                out["stl_hm"] = _inpaint_stl_nan(timed(
                    "Re-render simplified heightmap", mesh_to_heightmap, out_simpl,
                    resolution=resolution, projection="max", z_axis=stl_z_axis,
                    isotropic=True)["heightmap"])
                if decimation_curve:
                    try:
                        from ..processing.building_simplify import decimation_sweep
                        from ..io.readers import _load_trimesh_mesh
                        out["decimation_sweep"] = timed(
                            "Decimation sweep (curve)", decimation_sweep,
                            _load_trimesh_mesh(str(stl_file)), m_per_unit=m_per_unit)
                    except Exception as _exc:
                        logger.warning("Decimation sweep failed (%s).", _exc)
    except Exception as _exc:
        logger.warning("Mesh simplification failed (%s); using original mesh.", _exc)
    return out


def _run_registration(stl_reg, osm_reg, *, prism_polys, bx, by, cell_size_m_reg,
                      known_scale, max_scale_ratio, forced_rotation, free_scale,
                      registration_method, refine, resolution, timed, step_timings):
    """Stage 3 — recover the STL→OSM similarity transform (raster or polygon).

    In PRISM mode the clean separated prism-polygon mask drives rotation+scale from
    the polygon LINES (not the blob-prone heightmap segmentation).  Optionally runs
    projection discovery + an ECC SDF refine (shear-projected back to a similarity),
    then scales the REGISTER_RES transform to the output grid.  Returns reg_result,
    the output-grid transform, the chosen projection, and the raw reg_dict (whose
    sweep/histogram/xcorr diagnostics the caller unpacks for the report).
    """
    # Build the prism-polygon mask at REGISTER_RES from the [0,1] fractions.
    _reg_src_mask = None
    if prism_polys:
        try:
            from .config import REGISTER_RES as _RR
            xe = float(bx[1] - bx[0]); ye = float(by[1] - by[0])
            if xe >= ye:
                _cols = _RR; _rows = max(1, int(round(_RR * ye / xe)))
            else:
                _rows = _RR; _cols = max(1, int(round(_RR * xe / ye)))
            _c0 = (_RR - _cols) // 2; _r0 = (_RR - _rows) // 2
            import cv2 as _cv2
            _reg_src_mask = np.zeros((_RR, _RR), dtype=np.uint8)
            for pf in prism_polys:
                iso = np.column_stack([_c0 + pf[:, 0] * _cols, _r0 + pf[:, 1] * _rows])
                _cv2.fillPoly(_reg_src_mask, [np.rint(iso).astype(np.int32)], 1)
            logger.info("Registration signal: %d prism-polygon footprints "
                        "(rotation + scale from polygon lines, scale swept)", len(prism_polys))
        except Exception as _exc:
            logger.warning("Could not build prism-polygon registration mask (%s).", _exc)
            _reg_src_mask = None
    # Without a polygon mask: lock scale to the physical anchor (no drift basin).
    scale_search = 0.0 if known_scale is not None else 0.35
    reg_dict = None
    if registration_method == "polygon":
        # Polygon point-pattern matching (no gradient/xcorr); falls back to raster
        # when match confidence is low.
        reg_dict = timed("Register (polygon match)", _polygon_register_dict,
                         stl_reg, osm_reg, known_scale, prism_polys, bx, by, cell_size_m_reg)
        if reg_dict is None:
            logger.info("Polygon registration low-confidence → falling back to raster.")
    if reg_dict is None:
        reg_dict = timed(
            "Register (mask + ECC)", register, stl_reg, osm_reg,
            max_scale_ratio=max_scale_ratio, known_scale=known_scale,
            scale_search=scale_search, cell_size_m=cell_size_m_reg,
            forced_rotation=forced_rotation, source_mask=_reg_src_mask, free_scale=free_scale)
    for sub_name, sub_t in reg_dict.get("substep_timings", []):
        step_timings.append((f"↳ {sub_name}", sub_t))
    transform = reg_dict["transform"]

    # 3b. Optional second-stage projection discovery.  Skipped when scale is
    # physically anchored (ECC motion models reset scale to 1.0, discarding the warp).
    chosen_projection = "affine"
    if refine and known_scale is None:
        from .align import discover_projection

        def _refine_step():
            disc = discover_projection(stl_reg, osm_reg, transform)
            logger.info("Projection chosen: %s (val IoU=%.3f)",
                        disc["projection"], disc["val_iou"])
            return disc

        disc = timed("Discover projection", _refine_step)
        transform = disc["transform"]
        chosen_projection = disc["projection"]

    # 3b. ECC fine-alignment on SDFs of building masks.  ECC is affine (no similarity
    # mode), so its result can shear/anisotropically scale to overfit the periodic
    # grid; project it onto the nearest similarity (uniform scale+rotation+translation):
    # for [[a,b],[c,d]] the closest conformal matrix is [[p,-q],[q,p]], p=(a+d)/2,
    # q=(c-b)/2.  Keep only when edge IoU improves.
    def _project_to_similarity(M: np.ndarray, keep_scale: float | None = None) -> np.ndarray:
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        p = (a + d) / 2.0
        q = (c - b) / 2.0
        if keep_scale is not None:                 # force uniform scale back to target
            mag = (p * p + q * q) ** 0.5
            if mag > 1e-9:
                f = keep_scale / mag
                p *= f; q *= f
        S = M.copy()
        S[0, 0], S[0, 1] = p, -q
        S[1, 0], S[1, 1] = q, p
        return S

    import math as _m
    _peak_scale = _m.hypot(transform[0, 0], transform[1, 0])  # scale before ECC
    from .align import refine_transform as _refine_ecc, score_alignment as _score_align
    _pre_iou = _score_align(stl_reg, osm_reg, transform)["edge_iou"]
    t_ecc0 = time.perf_counter()
    try:
        _ecc = _refine_ecc(stl_reg, osm_reg, transform,
                           signal="sdf", motion="affine",
                           ecc_iterations=300, ecc_eps=1e-6, blur_sigma=1.5)
        _ecc_sim = _project_to_similarity(_ecc["transform"], keep_scale=_peak_scale)
        _post_iou = _score_align(stl_reg, osm_reg, _ecc_sim)["edge_iou"]
        if _post_iou > _pre_iou:
            transform = _ecc_sim
            logger.info("ECC SDF refine accepted (shear-projected): edge IoU "
                        "%.3f → %.3f (Δ+%.3f)", _pre_iou, _post_iou, _post_iou - _pre_iou)
        else:
            logger.info("ECC SDF refine rejected: %.3f ≤ %.3f (global kept)",
                        _post_iou, _pre_iou)
    except Exception as _exc:
        logger.warning("ECC SDF refine failed: %s", _exc)
    step_timings.append(("ECC SDF refinement", time.perf_counter() - t_ecc0))

    # Scale the (REGISTER_RES) transform to the output grid: the linear part is a pixel
    # ratio and is unchanged; only translation scales.
    from .config import REGISTER_RES
    if resolution != REGISTER_RES:
        _f = float(resolution) / REGISTER_RES
        transform = transform.copy()
        transform[0, 2] *= _f
        transform[1, 2] *= _f

    scale, angle_deg = _decompose_for_report(transform)
    reg_result = RegistrationResult(
        transform=transform, confidence=reg_dict["confidence"], scale=scale,
        angle_deg=angle_deg, n_iterations=reg_dict["n_iterations"],
        converged=reg_dict["converged"], projection=chosen_projection)
    return {"reg_result": reg_result, "transform": transform,
            "chosen_projection": chosen_projection, "reg_dict": reg_dict}


def _run_comparison(stl_hm, osm_hm, reg_result, *, cell_size_m, height_scale,
                    veg_mask, water_mask, refine_polygons, reg_dict, chosen_projection,
                    prism_polys, bx, by, resolution, eff_stl_file, stl_z_axis,
                    detect_resolution_factor, regularize_footprints,
                    mesh_to_heightmap, timed, step_timings):
    """Stages 4–5 — warp + mask + height comparison, optional polygon-ICP, footprints.

    `stl_hm` is the heightmap to compare (the prism-model render in prism mode; the
    caller swaps it in).  `_warp_mask_compare` is a closure so the polygon-ICP can
    re-run it on the corrected transform.  Returns the (possibly ICP-updated)
    reg_result, the aligned/masked/residual/compared arrays, comp_result, and the
    footprint polygons (prism polys mapped into OSM space, else hi-res detection).
    """
    from .align import building_mask as _building_mask, terrain_residual as _terrain_residual

    def _warp_mask_compare(T):
        s_al = apply_transform(stl_hm, T, osm_hm.shape)
        # Comparison mask.  PRISM mode: reuse the prism decomposition's separated +
        # regularized building footprints (rasterize the [0,1]-fraction polys → iso
        # grid → warp by T into OSM space) — one segmentation source shared with the
        # registration, and triangle+watershed-clean rather than the p50 re-threshold
        # which floods to ~50%.  Otherwise fall back to the p50 building_mask.
        if prism_polys:
            import cv2 as _cv2
            R = resolution
            xe = float(bx[1] - bx[0]); ye = float(by[1] - by[0])
            if xe >= ye:
                cols = R; rows = max(1, int(round(R * ye / xe)))
            else:
                rows = R; cols = max(1, int(round(R * xe / ye)))
            c0 = (R - cols) // 2; r0 = (R - rows) // 2
            _m = np.zeros(osm_hm.shape, dtype=np.uint8)
            for pf in prism_polys:
                iso = np.column_stack([c0 + pf[:, 0] * cols, r0 + pf[:, 1] * rows])
                osm = iso @ T[:, :2].T + T[:, 2]
                _cv2.fillPoly(_m, [np.rint(osm).astype(np.int32)], 1)
            bmask = _m.astype(bool)
        else:
            # working-res, dense (p50) split — many STL cells per OSM footprint.
            bmask = _building_mask(s_al, source="stl", cell_size_m=cell_size_m,
                                   segment_features=True, fill_holes_px=None)
        # Exclude STL cells OSM labels as vegetation or water (trees/rivers).
        if veg_mask is not None or water_mask is not None:
            exclude = np.zeros(bmask.shape, dtype=bool)
            if veg_mask is not None and veg_mask.shape == exclude.shape:
                exclude |= veg_mask
            if water_mask is not None and water_mask.shape == exclude.shape:
                exclude |= water_mask
            bmask &= ~exclude
        resid, _ = _terrain_residual(s_al, cell_size_m=cell_size_m)
        s_cmp = resid.copy()
        s_cmp[~bmask] = np.nan
        cmp = compare(s_cmp, osm_hm, height_scale)
        return s_al, bmask, resid, s_cmp, cmp

    stl_aligned, stl_building_mask, stl_residual, stl_for_compare, comp_result = \
        timed("Compare (height analysis)", _warp_mask_compare, reg_result.transform)

    # 5c. Polygon-matched ICP fine-tuning — ONLY when the coarse registration is
    # already decent (honest footprint IoU > 0.2; the ICP's own RMSE guard rejects
    # bad refinements).  STL footprints are already in OSM space (warped working-res
    # mask), so the ICP runs with an identity init and returns the OSM-space
    # correction to compose onto the registration transform.
    # Gate below the reference (Philadelphia ≈ 0.14 on this conservative comparison-mask
    # IoU); the ICP's own RMSE guard rejects refinements that don't actually help.
    _ICP_IOU_GATE = 0.10
    if refine_polygons and comp_result.footprint_iou > _ICP_IOU_GATE:
        try:
            from .align import (vectorize_buildings as _vec0,
                                 refine_registration_polygons as _poly_icp)
            _stl_polys = _vec0(_building_mask(stl_aligned, source="stl",
                               cell_size_m=cell_size_m, split_watershed=True),
                               regularize=True)
            _osm_polys = _vec0(_building_mask(osm_hm, source="osm"))
            _t_icp = time.perf_counter()
            _icp = _poly_icp(_stl_polys, _osm_polys,
                             np.array([[1.0, 0, 0], [0, 1.0, 0]]),
                             dice=comp_result.footprint_iou)
            step_timings.append(("Polygon ICP refinement", time.perf_counter() - _t_icp))
            if _icp["applied"]:
                _delta = _icp["transform"]
                _newT = (np.vstack([_delta, [0, 0, 1]]) @
                         np.vstack([reg_result.transform, [0, 0, 1]]))[:2]
                _sc, _ang = _decompose_for_report(_newT)
                reg_result = RegistrationResult(
                    transform=_newT, confidence=reg_dict["confidence"], scale=_sc,
                    angle_deg=_ang, n_iterations=reg_dict["n_iterations"],
                    converged=reg_dict["converged"], projection=chosen_projection)
                stl_aligned, stl_building_mask, stl_residual, stl_for_compare, comp_result = \
                    _warp_mask_compare(_newT)
                logger.info("Polygon ICP accepted: IoU=%.3f matched=%d rmse %.2f→%.2f px "
                            "(re-compared)", comp_result.footprint_iou, _icp["n_matched"],
                            _icp["rmse_before"], _icp["rmse_after"])
            else:
                logger.info("Polygon ICP rejected: %s", _icp["reason"])
        except Exception as _exc:
            logger.warning("Polygon ICP failed (%s); coarse transform kept.", _exc)
    elif refine_polygons:
        logger.info("Polygon ICP skipped: footprint IoU %.3f ≤ %.2f (registration "
                    "not confident enough).", comp_result.footprint_iou, _ICP_IOU_GATE)

    # Hi-res adaptive footprint POLYGONS: prefer the prism decomposition's separated,
    # regularized polygons (mapped [0,1]→iso grid→OSM space); else render the STL at
    # detect_factor× and segment with the adaptive (triangle) threshold (higher res
    # separates buildings that merge into blobs at 512).  Footprint figure only.
    stl_polygons = None
    detect_factor = max(1, int(detect_resolution_factor))
    if prism_polys:
        try:
            R = resolution
            xe = float(bx[1] - bx[0]); ye = float(by[1] - by[0])
            if xe >= ye:
                cols = R; rows = max(1, int(round(R * ye / xe)))
            else:
                rows = R; cols = max(1, int(round(R * xe / ye)))
            c0 = (R - cols) // 2; r0 = (R - rows) // 2
            T = reg_result.transform
            stl_polygons = []
            for pf in prism_polys:
                iso = np.column_stack([c0 + pf[:, 0] * cols, r0 + pf[:, 1] * rows])
                osm = iso @ T[:, :2].T + T[:, 2]
                stl_polygons.append(np.rint(osm).astype(np.int32))
            logger.info("Footprints: %d prism-decomposition polygons "
                        "(lines from polygons, no re-segmentation)", len(stl_polygons))
        except Exception as _exc:
            logger.warning("Prism polygon mapping failed (%s); using hi-res detect.", _exc)
            stl_polygons = None
    if stl_polygons is None and detect_factor > 1:
        try:
            from .align import vectorize_buildings as _vec
            det_res = int(resolution * detect_factor)
            stl_hi = _inpaint_stl_nan(mesh_to_heightmap(
                eff_stl_file, resolution=det_res, projection="max",
                z_axis=stl_z_axis, isotropic=True)["heightmap"])
            T_hi = reg_result.transform.copy()
            T_hi[0, 2] *= detect_factor
            T_hi[1, 2] *= detect_factor
            stl_hi_al = apply_transform(
                stl_hi, T_hi,
                (osm_hm.shape[0] * detect_factor, osm_hm.shape[1] * detect_factor))
            mask_hi = _building_mask(
                stl_hi_al, source="stl", cell_size_m=cell_size_m / detect_factor,
                threshold_method="triangle", segment_features=True, fill_holes_px=None,
                split_watershed=regularize_footprints)
            polys_hi = _vec(mask_hi, regularize=regularize_footprints)
            inv = 1.0 / detect_factor          # scale vertices back to the working grid
            stl_polygons = [(p.astype(np.float64) * inv).astype(np.int32) for p in polys_hi]
            logger.info("Hi-res footprints: %dx%d adaptive(triangle) -> %d polygons "
                        "(working-res had %d)", det_res, det_res, len(stl_polygons),
                        len(_vec(stl_building_mask)))
        except Exception as _exc:
            logger.warning("Hi-res footprint pass failed (%s); using working-res polygons.", _exc)
            stl_polygons = None

    return {"reg_result": reg_result, "stl_aligned": stl_aligned,
            "stl_building_mask": stl_building_mask, "stl_residual": stl_residual,
            "stl_for_compare": stl_for_compare, "comp_result": comp_result,
            "stl_polygons": stl_polygons}


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

    # 2b2. Fetch vegetation/water semantic masks on the same grid — used to
    # exclude trees/water misread as STL buildings before comparison.
    try:
        from ..applications.cities import get_osm_semantic_masks
        sem = _timed("Fetch OSM semantic masks", get_osm_semantic_masks,
                     osm_fetch_target, resolution=resolution)
        veg_mask = sem["vegetation"]
        water_mask = sem["water"]
    except Exception as _exc:
        logger.warning("OSM semantic masks unavailable (%s); skipping exclusion.", _exc)
        veg_mask = water_mask = None

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
        veg_mask=veg_mask, water_mask=water_mask, refine_polygons=refine_polygons,
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
