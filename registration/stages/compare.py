"""Stages 4–5 — warp + mask + height comparison, optional polygon-ICP, footprints."""

from __future__ import annotations

import logging
import time

import numpy as np

from ..align import apply_transform
from ..compare import compare
from ..types import RegistrationResult
from ._common import _decompose_for_report, _inpaint_stl_nan

logger = logging.getLogger(__name__)


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
    from ..align import building_mask as _building_mask, terrain_residual as _terrain_residual

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
            from ..align import (vectorize_buildings as _vec0,
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
            from ..align import vectorize_buildings as _vec
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
