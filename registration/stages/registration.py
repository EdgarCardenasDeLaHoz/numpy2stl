"""Stage 3 — recover the STL→OSM similarity transform (raster or polygon)."""

from __future__ import annotations

import logging
import time

import numpy as np

from ..align import register
from ..types import RegistrationResult
from ._common import _decompose_for_report

logger = logging.getLogger(__name__)


def _polygon_register_dict(stl_reg, osm_reg, known_scale, prism_polys, bx, by, cell_size_m_reg):
    """Polygon point-pattern registration → a reg_dict like register_global's.

    Builds STL footprint polygons (prism polygons mapped to REGISTER_RES, else a
    triangle-threshold vectorization of stl_reg) and OSM footprint polygons, then
    runs `register_polygons` with scale pinned to the geometric anchor.  Returns
    None (→ caller falls back to raster) when the match confidence is low.
    """
    import cv2 as _cv2
    from ..align import (register_polygons, vectorize_buildings, building_mask)
    from ..config import REGISTER_RES as _RR
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


def _run_registration(stl_reg, osm_reg, *, prism_polys, bx, by, cell_size_m_reg,
                      known_scale, max_scale_ratio, forced_rotation, free_scale,
                      registration_method, refine, resolution, timed, step_timings,
                      source_exclude_mask=None):
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
            from ..config import REGISTER_RES as _RR
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
            forced_rotation=forced_rotation, source_mask=_reg_src_mask, free_scale=free_scale,
            source_exclude_mask=source_exclude_mask)
    for sub_name, sub_t in reg_dict.get("substep_timings", []):
        step_timings.append((f"↳ {sub_name}", sub_t))
    transform = reg_dict["transform"]

    # 3b. Optional second-stage projection discovery.  Skipped when scale is
    # physically anchored (ECC motion models reset scale to 1.0, discarding the warp).
    chosen_projection = "affine"
    if refine and known_scale is None:
        from ..align import discover_projection

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
    from ..align import refine_transform as _refine_ecc, score_alignment as _score_align
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
    from ..config import REGISTER_RES
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
