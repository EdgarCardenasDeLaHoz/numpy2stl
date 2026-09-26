"""Stage 1 — shape-preserving quadric decimation + budget-bounded roof flattening."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


from ._io import _save_mesh, _save_prism_lod


class SimplifyStats(NamedTuple):
    orig_faces: int
    simplified_faces: int
    face_ratio: float           # simplified / original
    hausdorff_m: float          # achieved symmetric surface deviation
    deviation_tol_m: float
    flattened_buildings: int    # roofs levelled in Stage 2 (0 if flatten skipped)
    backend: str                # "trimesh+pymeshlab" | "none" (libs missing)


def _symmetric_hausdorff(mesh_a, mesh_b, n_samples: int = 20000) -> float:
    """Symmetric Hausdorff distance (max of the two directed surface distances).

    Sampled on each surface and measured to the other with trimesh's nearest-
    surface query — robust and version-independent (pymeshlab's own Hausdorff
    sampling is finicky), in the mesh's native units.
    """
    def _directed(src, dst) -> float:
        try:
            pts = src.sample(n_samples)
        except Exception:
            pts = np.asarray(src.vertices)
        if len(pts) == 0:
            return 0.0
        _, dist, _ = dst.nearest.on_surface(pts)
        return float(np.max(dist)) if len(dist) else 0.0

    return max(_directed(mesh_a, mesh_b), _directed(mesh_b, mesh_a))


def decimate_to_tolerance(
    mesh,
    deviation_tol: float,
    min_ratio: float = 0.02,
    n_iter: int = 7,
    n_samples: int = 20000,
):
    """Most aggressive quadric decimation whose symmetric Hausdorff to `mesh`
    stays ≤ `deviation_tol` (mesh units).

    Binary-searches the kept-face ratio in [min_ratio, 1.0]: smaller ratio = more
    decimation.  Returns ``(decimated_mesh, kept_ratio, achieved_hausdorff)``.
    """
    from ...stl2numpy.reduction import decimate_trimesh

    f0 = len(mesh.faces)
    # lo = most aggressive (smallest ratio), hi = safest (no decimation).
    lo, hi = float(min_ratio), 1.0
    best_mesh, best_ratio, best_h = mesh, 1.0, 0.0

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        cand = decimate_trimesh(mesh, int(mid * f0), preserve=True)
        h = _symmetric_hausdorff(mesh, cand, n_samples=n_samples)
        if h <= deviation_tol:
            # within budget — accept and push for MORE decimation (lower ratio)
            best_mesh, best_ratio, best_h = cand, len(cand.faces) / f0, h
            hi = mid
        else:
            # exceeded budget — back off (keep more faces)
            lo = mid
    logger.info("decimate_to_tolerance: %d -> %d faces (%.1f%%), hausdorff=%.3f (tol %.3f)",
                f0, len(best_mesh.faces), 100.0 * best_ratio, best_h, deviation_tol)
    return best_mesh, best_ratio, best_h


def decimation_sweep(
    mesh,
    m_per_unit: float = 1.0,
    ratios=(0.2, 0.4, 0.6, 0.8),
    n_samples: int = 3000,
):
    """Characterise the decimation quality/size trade-off for evaluation.

    Decimates the mesh to each target face ratio and measures the resulting
    symmetric Hausdorff (surface deviation).  Returns a list of dicts
    ``{ratio, faces, hausdorff_units, hausdorff_m}`` — the "decimation curve"
    the report plots so the deviation budget can be chosen with eyes open.

    Extremely aggressive ratios (<~15 %) are intentionally excluded: with
    topology preservation the decimator locks up there and emits degenerate
    geometry (spurious giant triangles → non-physical Hausdorff), which is also
    where it is slowest.  Any point whose deviation exceeds the mesh bounding
    diagonal is dropped as degenerate.
    """
    from ...stl2numpy.reduction import decimate_trimesh
    f0 = len(mesh.faces)
    diag = float(np.linalg.norm(mesh.extents))   # sanity ceiling for Hausdorff
    out = []
    for r in ratios:
        cand = decimate_trimesh(mesh, int(r * f0), preserve=True)
        h = _symmetric_hausdorff(mesh, cand, n_samples=n_samples)
        if not np.isfinite(h) or h > diag:
            logger.info("decimation_sweep: ratio≈%.2f degenerate (dev=%.1f > diag=%.1f); skipped",
                        r, h, diag)
            continue
        out.append({"ratio": float(len(cand.faces) / f0), "faces": int(len(cand.faces)),
                    "hausdorff_units": float(h), "hausdorff_m": float(h * m_per_unit)})
        logger.info("decimation_sweep: ratio≈%.2f -> %d faces, dev=%.3f units (%.2f m)",
                    r, len(cand.faces), h, h * m_per_unit)
    return sorted(out, key=lambda d: d["ratio"])


def flatten_roof_clutter(
    heightmap: np.ndarray,
    labels: np.ndarray,
    deviation_tol_m: float,
) -> tuple[np.ndarray, int]:
    """Level per-building roof variation that is within the deviation budget.

    For each labelled building component, if its roof height spread (p95−p10)
    is ≤ ``deviation_tol_m`` the whole roof is set to a single representative
    plateau (p90) — removing AC units / parapets / small slopes.  Components with
    a spread larger than the budget (towers, spires, stepped massing) are left
    untouched so large structure survives.  The footprint (the labelled extent)
    is never modified, so overlap is preserved by construction.

    Returns ``(flattened_heightmap, n_flattened)``.
    """
    out = heightmap.copy()
    n_flat = 0
    ids = np.unique(labels)
    for lid in ids:
        if lid == 0:
            continue
        comp = labels == lid
        vals = heightmap[comp]
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            continue
        spread = float(np.percentile(vals, 95) - np.percentile(vals, 10))
        if spread <= deviation_tol_m:
            out[comp] = float(np.percentile(vals, 90))
            n_flat += 1
    logger.info("flatten_roof_clutter: levelled %d/%d building roofs (spread<=%.2fm)",
                n_flat, max(0, len(ids) - 1), deviation_tol_m)
    return out, n_flat


def simplify_building_mesh(
    file_path: str,
    deviation_tol_m: float = 3.5,
    z_axis: int = 2,
    save_path: str | Path | None = None,
    save_prism_lod: bool = False,
):
    """Footprint-preserving simplification of a city building mesh (Stage 1 + save).

    Returns ``(vertices, faces, SimplifyStats)``.  This entry point performs the
    mesh-level decimation (Stage 1) + saved deliverable; Stage-2 roof flattening is
    a heightmap-domain operation provided separately as `flatten_roof_clutter` for
    callers that have segmented building labels.  When mesh libs are unavailable,
    returns the original mesh unchanged with ``backend="none"``.
    """
    try:
        import trimesh  # noqa: F401
        from ...io.readers import _load_trimesh_mesh
    except Exception:
        logger.warning("simplify_building_mesh: trimesh unavailable; returning original mesh.")
        return None, None, SimplifyStats(0, 0, 1.0, 0.0, deviation_tol_m, 0, "none")

    mesh = _load_trimesh_mesh(file_path)
    f0 = len(mesh.faces)
    try:
        simp, ratio, haus = decimate_to_tolerance(mesh, deviation_tol_m)
        backend = "trimesh+pymeshlab"
    except Exception as exc:
        logger.warning("simplify_building_mesh: decimation failed (%s); using original.", exc)
        simp, ratio, haus, backend = mesh, 1.0, 0.0, "none"

    stats = SimplifyStats(
        orig_faces=f0, simplified_faces=len(simp.faces), face_ratio=float(ratio),
        hausdorff_m=float(haus), deviation_tol_m=float(deviation_tol_m),
        flattened_buildings=0, backend=backend,
    )

    if save_path is not None:
        _save_mesh(simp, save_path)
    if save_prism_lod and save_path is not None:
        _save_prism_lod(simp, z_axis, Path(save_path), deviation_tol_m)

    return np.asarray(simp.vertices), np.asarray(simp.faces), stats
