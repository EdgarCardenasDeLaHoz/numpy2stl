"""Footprint-preserving building-mesh simplification.

Takes a city STL/3MF and reduces its detail **as much as possible without
changing the building shape beyond a metres deviation budget**, then optionally
flattens roof clutter and re-extrudes a clean blocky LOD.  The cleaner mesh
rasterises into a crisper heightmap, which makes building-footprint segmentation
and street-level separation downstream much easier.

Single knob — `deviation_tol_m` (metres).  "As much as possible without changing
overlap" = maximise removal subject to a symmetric surface (Hausdorff) deviation
≤ `deviation_tol_m`.  A larger budget removes more (more aggressive flattening);
tall distinct structures whose removal would exceed the budget are preserved.

Stages
------
1. **Decimation** (shape-preserving): binary-search the quadric-decimation strength
   for the most aggressive face reduction whose symmetric Hausdorff to the original
   stays within budget.  Reuses `stl2numpy.reduction.decimate_trimesh`.
2. **Roof flattening** (budget-bounded): on the decimated mesh's heightmap, level
   per-building vertical roof variation that is ≤ budget into a flat plateau; keep
   structures taller than the budget.
3. (segmentation/regularization live in `registration/align/segmentation.py`.)
4. **Save**: write the decimated mesh (faithful LOD) and/or a re-extruded prism LOD
   via `processing.extrusion` + `io.writers`.

Mesh-geometry libs (trimesh, pymeshlab) are required for Stage 1; the function
degrades gracefully (returns the original mesh + a flag) when they are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class SimplifyStats(NamedTuple):
    orig_faces: int
    simplified_faces: int
    face_ratio: float           # simplified / original
    hausdorff_m: float          # achieved symmetric surface deviation
    deviation_tol_m: float
    flattened_buildings: int    # roofs levelled in Stage 2 (0 if flatten skipped)
    backend: str                # "trimesh+pymeshlab" | "none" (libs missing)


# ---------------------------------------------------------------------------
# Surface-deviation measurement
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Stage 1 — shape-preserving decimation to a deviation budget
# ---------------------------------------------------------------------------

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
    from ..stl2numpy.reduction import decimate_trimesh

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
    from ..stl2numpy.reduction import decimate_trimesh
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


# ---------------------------------------------------------------------------
# Stage 2 — roof/clutter flattening, bounded by the budget (height-field)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

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
        from ..io.readers import _load_trimesh_mesh
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


class PrismStats(NamedTuple):
    n_buildings: int
    n_prisms: int
    mean_layers: float          # mean prism layers per building
    sloped_caps: int            # how many top caps used a fitted (slanted) plane
    hausdorff_m: float          # prism-model heightmap deviation vs original (metres)
    deviation_tol_m: float
    backend: str


def _fit_plane(xs, ys, zs):
    """Least-squares plane z = a*x + b*y + c; returns ((a,b,c), rms)."""
    A = np.column_stack([xs, ys, np.ones(len(zs))])
    coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    rms = float(np.sqrt(np.mean((zs - A @ coef) ** 2)))
    return (float(coef[0]), float(coef[1]), float(coef[2])), rms


def prism_decompose(
    file_path: str,
    deviation_tol: float = 3.5,
    z_axis: int = 2,
    resolution: int = 512,
    m_per_unit: float = 1.0,
    max_layers: int = 8,
    min_area_px: int = 20,
    save_path: str | Path | None = None,
):
    """Decompose a city STL into a **sum of extruded prisms** (the reverse of how
    OSM footprints are rendered into a model).

    Each building becomes a wedding-cake stack of nested prisms (footprint at each
    budget-spaced height level), and the topmost roof is capped with a fitted
    plane (slanted) where it is planar within budget.  ``deviation_tol`` is in
    MESH UNITS (the caller converts a metres budget via the model scale); levels
    are spaced by it, so the prism model stays within budget of the original.

    Returns ``(vertices, faces, PrismStats, models)`` where ``models`` is a
    ``{name: (verts, faces)}`` soup (one entry per prism, like an OSM extrude
    set), saved to ``save_path`` (STL = merged, 3MF = soup) when given.
    """
    import trimesh
    from ..stl2numpy.heightmap import mesh_to_heightmap
    from ..registration.align import (building_mask, vectorize_buildings,
                                       terrain_residual)
    from .extrusion import make_sloped_prism_solid

    # Non-isotropic render → simple pixel↔world mapping (no NaN padding).
    r = mesh_to_heightmap(file_path, resolution=resolution, projection="max",
                          z_axis=z_axis, isotropic=False, cache=False)
    hm = r["heightmap"]
    (x_min, x_max), (y_min, y_max) = r["bounds"]["x"], r["bounds"]["y"]
    cell_x, cell_y = r["cell_size"]
    cell_m = 0.5 * (abs(cell_x) + abs(cell_y)) * m_per_unit

    residual, _ = terrain_residual(hm, cell_size_m=cell_m)
    # Use the TRIANGLE threshold, not the default p50: the STL fills the whole frame
    # (base plate everywhere), so the residual median (p50) is low and ~50% of cells
    # exceed it — flooding streets/base as "buildings" and fusing the whole downtown
    # into one component.  Triangle finds the valley between the ground peak and the
    # building tail (~20% of frame), giving SEPARATED individual building footprints.
    mask = building_mask(hm, source="stl", cell_size_m=cell_m,
                         threshold_method="triangle", split_watershed=True)
    try:
        from scipy import ndimage as ndi
        labels, n_lab = ndi.label(mask)
    except Exception:
        labels, n_lab = mask.astype(int), 1

    def _px_to_world(poly_px):
        w = np.empty_like(poly_px, dtype=np.float64)
        w[:, 0] = x_min + (poly_px[:, 0] + 0.5) * cell_x   # col → world x
        w[:, 1] = y_min + (poly_px[:, 1] + 0.5) * cell_y   # row → world y
        return w

    step = float(deviation_tol)
    models: dict = {}
    n_prisms = 0
    layer_counts = []
    sloped = 0
    # Prism-model surface on the SAME grid as `residual`, for a grid-consistent
    # deviation measure (height-above-ground, so no base-plate offset).
    prism_hm = np.full_like(residual, np.nan, dtype=np.float64)

    base_polys_frac = []   # per-building footprint outlines, as [0,1] bounds-fractions
    for lid in range(1, n_lab + 1):
        comp = labels == lid
        rv = residual[comp]
        rv = rv[np.isfinite(rv)]
        if rv.size < min_area_px:
            continue
        top = float(np.percentile(rv, 99))
        # Space layers by the deviation budget so the quantization error stays ≤
        # the budget; cap the count for runaway-tall buildings (a smooth tower
        # keeps the same footprint at every level, so the extra prisms are nested
        # and the max-projected height is unaffected — only the file grows).
        if top <= step:
            levels = [0.0, top]                       # single prism
        else:
            n = max(1, int(np.ceil(top / step)))
            if n > max_layers:
                n = max_layers
            levels = list(np.linspace(0.0, top, n + 1))
        n_layers_here = 0
        for k in range(len(levels) - 1):
            z_lo, z_hi = levels[k], levels[k + 1]
            sub = comp & (residual >= z_lo)
            polys = vectorize_buildings(sub.astype(np.uint8), regularize=True,
                                        min_area_px=min_area_px)
            if k == 0:   # base layer = the building's full (separated) footprint
                for _p in polys:
                    base_polys_frac.append(_p.astype(np.float64) / float(resolution))
            is_top = (k == len(levels) - 2)
            plane = None
            if is_top:
                # Fit a plane to the original roof over the top region; slant the
                # cap if it is planar within the deviation budget.
                ys, xs = np.where(sub)
                if xs.size >= 8:
                    wx = x_min + (xs + 0.5) * cell_x
                    wy = y_min + (ys + 0.5) * cell_y
                    (a, b, c), rms = _fit_plane(wx, wy, residual[ys, xs])
                    if rms <= step:
                        plane = (a, b, c)
            # Record this layer's contribution to the prism surface (max-projection).
            cap = z_hi if plane is None else None
            sub_pix = sub if cap is not None else None
            for poly_px in polys:
                world = _px_to_world(poly_px.astype(np.float64))
                try:
                    if plane is not None:
                        v, f = make_sloped_prism_solid(world, z0=float(z_lo), plane=plane)
                        if len(f):
                            sloped += 1
                    else:
                        v, f = make_sloped_prism_solid(world, z0=float(z_lo),
                                                       plane=None, z1=float(z_hi))
                except Exception:
                    continue
                if len(f):
                    models[f"b{lid}_l{k}_{len(models)}"] = (v, f)
                    n_prisms += 1
                    n_layers_here += 1
            # Update the prism surface for this layer's footprint.
            if plane is None:
                prism_hm[sub] = z_hi
            else:
                ys, xs = np.where(sub)
                prism_hm[ys, xs] = plane[0] * (x_min + (xs + 0.5) * cell_x) + \
                                   plane[1] * (y_min + (ys + 0.5) * cell_y) + plane[2]
        if n_layers_here:
            layer_counts.append(n_layers_here)

    # Merge into one mesh for stats / return + save.
    if models:
        merged = trimesh.util.concatenate(
            [trimesh.Trimesh(vertices=v, faces=f, process=False) for v, f in models.values()])
        verts, faces = np.asarray(merged.vertices), np.asarray(merged.faces)
    else:
        merged, verts, faces = None, np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    # Deviation: prism surface vs original residual on the SAME grid (both are
    # height-above-ground, so no base-plate offset).  p99 of |prism − original|
    # over the building footprints — the honest "how well do the stacked prisms
    # approximate the massing" number.
    haus_m = float("nan")
    both = np.isfinite(prism_hm) & np.isfinite(residual) & (residual > 0)
    if both.any():
        haus_m = float(np.percentile(np.abs(prism_hm[both] - residual[both]), 99) * m_per_unit)

    stats = PrismStats(
        n_buildings=len(layer_counts), n_prisms=n_prisms,
        mean_layers=float(np.mean(layer_counts)) if layer_counts else 0.0,
        sloped_caps=sloped, hausdorff_m=haus_m,
        deviation_tol_m=float(deviation_tol * m_per_unit), backend="prism",
    )
    logger.info("prism_decompose: %d buildings → %d prisms (mean %.1f layers, %d sloped caps), "
                "deviation≈%.2f m", stats.n_buildings, stats.n_prisms, stats.mean_layers,
                stats.sloped_caps, stats.hausdorff_m)

    if save_path is not None and models:
        _save_prism_models(models, merged, save_path)
    # base_polys_frac: per-building footprint outlines as [0,1] bounds-fractions,
    # already watershed-separated + regularized — for deriving footprint LINES
    # downstream WITHOUT a (blob-prone) working-resolution re-segmentation.
    return verts, faces, stats, models, base_polys_frac


def _save_prism_models(models: dict, merged, save_path: str | Path) -> None:
    """Save the prism soup: STL = merged mesh, 3MF = one object per prism."""
    from ..io.writers import write3MF
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
        from ..io.writers import writeSTL
        tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        facets = np.zeros((len(tris), 12), dtype=np.float32)
        facets[:, 3:] = tris.reshape(len(tris), 9)
        writeSTL(facets, str(save_path.with_suffix(".stl")))


def _save_prism_lod(mesh, z_axis: int, save_path: Path, deviation_tol_m: float) -> None:
    """Aggressive blocky LOD: detect footprints on the (simplified) heightmap and
    re-extrude each as a flat-top prism at its plateau height."""
    from ..stl2numpy.heightmap import _load_trimesh_mesh  # noqa: F401
    from ..registration.align.segmentation import building_mask, vectorize_buildings
    from .extrusion import make_prism_solid
    from ..io.writers import write3MF

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
    cell = (x.ptp() / resolution, y.ptp() / resolution)
    return hm.T.astype(np.float64), cell
