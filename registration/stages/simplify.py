"""Stage 0 — optional mesh simplification (decimate | prism)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ._common import _inpaint_stl_nan

logger = logging.getLogger(__name__)


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
        from ...applications.cities import derive_scale_m_per_unit
        import tempfile
        import os as _os
        m_per_unit = derive_scale_m_per_unit(
            city_name, stl_z_max, tallest_m=tallest_m, scale_m_per_unit=scale_m_per_unit)
        if not m_per_unit or m_per_unit <= 0:
            m_per_unit = 1.0
            logger.warning("Simplify: no scale anchor; treating deviation budget "
                           "as mesh units (%.2f).", simplify_tol_m)
        tol_units = float(simplify_tol_m) / float(m_per_unit)
        stem = Path(stl_file).stem

        if mode == "prism":
            from ...processing.building_simplify import prism_decompose
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
            from ...processing.building_simplify import simplify_building_mesh
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
                        from ...processing.building_simplify import decimation_sweep
                        from ...io.readers import _load_trimesh_mesh
                        out["decimation_sweep"] = timed(
                            "Decimation sweep (curve)", decimation_sweep,
                            _load_trimesh_mesh(str(stl_file)), m_per_unit=m_per_unit)
                    except Exception as _exc:
                        logger.warning("Decimation sweep failed (%s).", _exc)
    except Exception as _exc:
        logger.warning("Mesh simplification failed (%s); using original mesh.", _exc)
    return out
