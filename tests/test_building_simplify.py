# Tests for footprint-preserving mesh simplification + improved segmentation.
# No network. Mesh-geometry tests skip when trimesh/pymeshlab are unavailable.
import numpy as np
import pytest

try:
    import trimesh  # noqa: F401
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def _footprint(mesh, z_axis=2, n=64, ref_bounds=None):
    """Filled XY footprint of a mesh on an n×n grid over a common bounds box.

    Splats densely sampled surface points then closes + fills so the occupancy is
    a solid footprint (not a sparse point spray).  Both meshes must use the SAME
    `ref_bounds` so the grids are comparable.
    """
    import scipy.ndimage as ndi
    h_axes = [i for i in range(3) if i != z_axis]
    pts = np.vstack([mesh.vertices, mesh.sample(80000)])
    b = ref_bounds if ref_bounds is not None else mesh.bounds
    x = (pts[:, h_axes[0]] - b[0, h_axes[0]]) / (b[1, h_axes[0]] - b[0, h_axes[0]] + 1e-9)
    y = (pts[:, h_axes[1]] - b[0, h_axes[1]]) / (b[1, h_axes[1]] - b[0, h_axes[1]] + 1e-9)
    grid = np.zeros((n, n), bool)
    ix = np.clip((x * (n - 1)).astype(int), 0, n - 1)
    iy = np.clip((y * (n - 1)).astype(int), 0, n - 1)
    grid[iy, ix] = True
    grid = ndi.binary_closing(grid, iterations=2)
    grid = ndi.binary_fill_holes(grid)
    return grid


def _dice(a, b):
    a, b = a > 0, b > 0
    return 2.0 * (a & b).sum() / max(1, a.sum() + b.sum())


def _building_with_clutter():
    """30 m tower + a 2 m roof bump (< budget) + a 12 m spire (> budget)."""
    box = trimesh.creation.box(extents=(10, 10, 30)); box.apply_translation([0, 0, 15])
    bump = trimesh.creation.box(extents=(3, 3, 2)); bump.apply_translation([0, 0, 31])
    spire = trimesh.creation.box(extents=(1, 1, 12)); spire.apply_translation([3, 3, 36])
    return trimesh.util.concatenate([box, bump, spire]).subdivide().subdivide()


# ---------------------------------------------------------------------------
# Stage 1 — Hausdorff-bounded decimation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
class TestDecimateToTolerance:

    def test_reduces_faces_within_budget_preserving_footprint(self):
        from numpy2stl.processing.building_simplify import (
            decimate_to_tolerance, _symmetric_hausdorff)
        mesh = _building_with_clutter()
        tol = 3.5
        simp, ratio, h = decimate_to_tolerance(mesh, deviation_tol=tol)
        assert len(simp.faces) < len(mesh.faces)          # actually decimated
        assert h <= tol + 1e-6                             # within deviation budget
        assert _symmetric_hausdorff(mesh, simp) <= tol + 1e-6
        # footprint (XY silhouette) essentially unchanged (common bounds box)
        rb = mesh.bounds
        assert _dice(_footprint(mesh, ref_bounds=rb), _footprint(simp, ref_bounds=rb)) > 0.97

    def test_tiny_budget_keeps_more_faces_than_large_budget(self):
        from numpy2stl.processing.building_simplify import decimate_to_tolerance
        mesh = _building_with_clutter()
        _, r_small, _ = decimate_to_tolerance(mesh, deviation_tol=0.2)
        _, r_large, _ = decimate_to_tolerance(mesh, deviation_tol=6.0)
        assert r_small >= r_large    # tighter budget → fewer removals (higher keep ratio)

    def test_decimation_sweep_monotonic_and_clean(self):
        from numpy2stl.processing.building_simplify import decimation_sweep
        mesh = _building_with_clutter()
        diag = float(np.linalg.norm(mesh.extents))
        sweep = decimation_sweep(mesh, m_per_unit=2.0, ratios=(0.3, 0.6, 0.9), n_samples=2000)
        assert len(sweep) >= 2
        # all points physical (no degenerate blow-ups) and metres = units × scale
        for d in sweep:
            assert d["hausdorff_units"] <= diag
            assert abs(d["hausdorff_m"] - d["hausdorff_units"] * 2.0) < 1e-6
        # keeping more faces → lower-or-equal deviation
        ratios = [d["ratio"] for d in sweep]; devs = [d["hausdorff_units"] for d in sweep]
        assert ratios == sorted(ratios)
        assert devs[-1] <= devs[0] + 1e-6


# ---------------------------------------------------------------------------
# Stage 2 — roof flattening
# ---------------------------------------------------------------------------

class TestFlattenRoofClutter:

    def test_levels_low_spread_keeps_tall(self):
        from numpy2stl.processing.building_simplify import flatten_roof_clutter
        hm = np.zeros((40, 40))
        # building A: flat-ish roof at 20 with a 2 m bump (spread < 3.5) → flatten
        hm[5:15, 5:15] = 20.0; hm[8:11, 8:11] = 22.0
        # building B: stepped 10..30 (spread 20 > 3.5) → keep
        hm[25:35, 25:35] = np.linspace(10, 30, 10)[None, :]
        labels = np.zeros((40, 40), int)
        labels[5:15, 5:15] = 1
        labels[25:35, 25:35] = 2
        out, n = flatten_roof_clutter(hm, labels, deviation_tol_m=3.5)
        assert n == 1                                   # only A flattened
        assert np.ptp(out[5:15, 5:15]) == 0.0           # A is now a single plateau
        assert np.ptp(out[25:35, 25:35]) > 3.5          # B untouched


# ---------------------------------------------------------------------------
# Stage 3 — segmentation improvements
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CV2, reason="opencv required")
class TestSegmentationImprovements:

    def test_watershed_splits_touching(self):
        from numpy2stl.registration.align import split_touching_buildings
        import scipy.ndimage as ndi
        m = np.zeros((80, 80), bool)
        m[20:40, 15:35] = True; m[20:40, 45:65] = True; m[28:32, 35:45] = True
        assert ndi.label(m)[1] == 1
        out = split_touching_buildings(m, cell_size_m=1.0, min_separation_m=12)
        assert ndi.label(out)[1] == 2

    def test_regularize_squares_jog(self):
        from numpy2stl.registration.align import _regularize_polygon
        poly = np.array([[10, 10], [30, 12], [50, 10], [51, 40], [10, 39]], np.int32)
        reg = _regularize_polygon(poly, max_snap_px=3)
        # top edge becomes a single y; near-vertical right edge a single x
        assert reg[0, 1] == reg[1, 1] == reg[2, 1]
        assert abs(int(reg[3, 0]) - int(reg[2, 0])) <= 1

    def test_regularize_preserves_clean_rect(self):
        from numpy2stl.registration.align import _regularize_polygon
        r = np.array([[0, 0], [40, 0], [40, 30], [0, 30]], np.int32)
        out = _regularize_polygon(r, max_snap_px=3)
        assert _dice(_poly_mask(r), _poly_mask(out)) > 0.99


def _poly_mask(poly, n=64):
    import cv2
    m = np.zeros((n, n), np.uint8)
    cv2.fillPoly(m, [np.asarray(poly, np.int32)], 1)
    return m.astype(bool)


# ---------------------------------------------------------------------------
# Stage 5 — polygon-matched ICP refinement
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CV2, reason="opencv required")
class TestPolygonICP:

    def _grid_polys(self):
        polys = []
        for gx in range(4):
            for gy in range(4):
                x, y = 20 + gx * 40, 20 + gy * 40
                polys.append(np.array([[x, y], [x + 20, y], [x + 20, y + 15], [x, y + 15]], float))
        return polys

    def test_recovers_known_misalignment(self):
        import cv2
        from numpy2stl.registration.align import refine_registration_polygons
        osm = self._grid_polys()
        M = cv2.getRotationMatrix2D((100, 100), 1.5, 1.02); M[0, 2] += 3; M[1, 2] += -2
        stl = [p @ M[:, :2].T + M[:, 2] for p in osm]   # residual error baked in
        res = refine_registration_polygons(stl, osm, np.eye(2, 3, dtype=float), dice=0.99)
        assert res["applied"]
        assert res["n_matched"] >= 12
        assert res["rmse_after"] < 0.5 * res["rmse_before"]

    def test_gate_rejects_low_dice(self):
        from numpy2stl.registration.align import refine_registration_polygons
        osm = self._grid_polys()
        res = refine_registration_polygons(osm, osm, np.eye(2, 3, dtype=float), dice=0.80)
        assert not res["applied"]
        assert "skipped" in res["reason"]


# ---------------------------------------------------------------------------
# Orchestrator + save round-trip
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
class TestSlopedPrism:

    def test_flat_slanted_and_badfit(self):
        from numpy2stl.processing.extrusion import make_sloped_prism_solid
        sq = np.array([[0, 0], [10, 0], [10, 8], [0, 8]], float)
        for plane, z1 in ((None, 7.0), ((0.3, 0.1, 8.0), None), ((0, 0, -5), None)):
            v, f = make_sloped_prism_solid(sq, z0=1.0, plane=plane,
                                           z1=(z1 if z1 else 5))
            m = trimesh.Trimesh(vertices=v, faces=f)
            assert len(f) > 0 and m.is_watertight
            assert m.bounds[0, 2] >= 1.0 - 1e-6     # never dips below z0


@pytest.mark.skipif(not (HAS_TRIMESH and HAS_CV2), reason="trimesh+opencv required")
class TestPrismDecompose:

    def test_stacked_layers_and_save(self, tmp_path):
        from numpy2stl.processing.building_simplify import prism_decompose
        # podium (40x40x10) + tower (15x15x40) → a multi-layer wedding cake
        podium = trimesh.creation.box(extents=(40, 40, 10)); podium.apply_translation([0, 0, 5])
        tower = trimesh.creation.box(extents=(15, 15, 40)); tower.apply_translation([0, 0, 30])
        stl = tmp_path / "step.stl"
        trimesh.util.concatenate([podium, tower]).export(str(stl))
        out = tmp_path / "prism.stl"
        v, f, st, models, base_polys = prism_decompose(str(stl), deviation_tol=8.0, resolution=128,
                                           m_per_unit=1.0, save_path=str(out))
        assert st.backend == "prism"
        assert st.n_buildings >= 1
        assert st.n_prisms >= 2          # stepped massing → multiple layers
        assert st.mean_layers >= 2.0
        assert len(f) > 0
        assert out.exists() and out.with_suffix(".3mf").exists()
        assert len(trimesh.load(str(out)).faces) > 0


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh required")
class TestSimplifyBuildingMesh:

    def test_simplify_and_save_roundtrip(self, tmp_path):
        from numpy2stl.processing.building_simplify import simplify_building_mesh
        stl_in = tmp_path / "in.stl"
        _building_with_clutter().export(str(stl_in))
        out = tmp_path / "simplified.stl"
        v, f, stats = simplify_building_mesh(str(stl_in), deviation_tol_m=3.5, save_path=str(out))
        assert stats.backend != "none"
        assert stats.simplified_faces <= stats.orig_faces
        assert stats.hausdorff_m <= stats.deviation_tol_m + 1e-6
        assert out.exists()
        reload = trimesh.load(str(out))
        assert len(reload.faces) > 0
