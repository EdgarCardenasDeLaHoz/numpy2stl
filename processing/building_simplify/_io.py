"""Shared mesh I/O + rasterization helpers for building simplification."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _save_prism_models(models: dict, merged, save_path: str | Path) -> None:
    """Save the prism soup: STL = merged mesh, 3MF = one object per prism."""
    from ...io.writers import write3MF
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _save_mesh(merged, save_path)                         # merged STL (or by ext)
    try:
        write3MF(str(save_path.with_suffix(".3mf")), models)   # prism-soup 3MF
    except Exception as exc:
        logger.warning("Could not write prism 3MF (%s).", exc)


def _save_mesh(mesh, save_path: str | Path) -> None:
    """Write a trimesh to STL/3MF/OBJ (via trimesh.export, which handles formats)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(str(save_path))
        logger.info("Saved simplified mesh: %s (%d faces)", save_path, len(mesh.faces))
    except Exception as exc:
        logger.warning("Could not export %s (%s); falling back to writeSTL.", save_path, exc)
        from ...io.writers import writeSTL
        tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        facets = np.zeros((len(tris), 12), dtype=np.float32)
        facets[:, 3:] = tris.reshape(len(tris), 9)
        writeSTL(facets, str(save_path.with_suffix(".stl")))


def _save_prism_lod(mesh, z_axis: int, save_path: Path, deviation_tol_m: float) -> None:
    """Aggressive blocky LOD: detect footprints on the (simplified) heightmap and
    re-extrude each as a flat-top prism at its plateau height."""
    from ...stl2numpy.heightmap import _load_trimesh_mesh  # noqa: F401
    from ...registration.align.segmentation import building_mask, vectorize_buildings
    from ..extrusion import make_prism_solid
    from ...io.writers import write3MF

    # Rasterise the in-memory simplified mesh to a heightmap.
    hm, cell = _rasterize_mesh(mesh, z_axis=z_axis, resolution=512)
    mask = building_mask(hm, source="stl")
    polys = vectorize_buildings(mask, simplify_frac=0.02)
    models = {}
    for i, poly in enumerate(polys):
        comp_vals = hm[mask][np.isfinite(hm[mask])]
        z1 = float(np.percentile(comp_vals, 90)) if comp_vals.size else 1.0
        try:
            v, f = make_prism_solid(poly.astype(np.float64), z0=0.0, z1=z1)
            if len(f):
                models[f"b{i}"] = (v, f)
        except Exception:
            continue
    if models:
        out = save_path.with_name(save_path.stem + "_prismLOD.3mf")
        write3MF(str(out), models)
        logger.info("Saved prism LOD: %s (%d buildings)", out, len(models))


def _rasterize_mesh(mesh, z_axis: int = 2, resolution: int = 512):
    """Quick in-memory mesh → heightmap (max projection), mirroring
    mesh_to_heightmap's binning but without the file/cache path."""
    from scipy.stats import binned_statistic_2d
    h_axes = [i for i in range(3) if i != z_axis]
    v = np.asarray(mesh.vertices)
    try:
        pts = mesh.sample(resolution * resolution * 8)
        pts = np.vstack([v, pts])
    except Exception:
        pts = v
    x, y, z = pts[:, h_axes[0]], pts[:, h_axes[1]], pts[:, z_axis]
    hm, _, _, _ = binned_statistic_2d(x, y, z, statistic="max",
                                      bins=[resolution, resolution])
    cell = (np.ptp(x) / resolution, np.ptp(y) / resolution)
    return hm.T.astype(np.float64), cell
