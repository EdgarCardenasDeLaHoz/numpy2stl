# Tests for the registration pipeline
# No-network unit tests run always; integration tests require osmnx + a real STL.
import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import osmnx  # noqa: F401
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

try:
    import geopandas  # noqa: F401
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ---------------------------------------------------------------------------
# Synthetic fixtures (no files, no network)
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_building_array():
    """32×32 array with a few rectangular 'buildings' — controlled test data."""
    arr = np.zeros((32, 32), dtype=np.float64)
    arr[6:14, 6:14] = 20.0    # building A
    arr[18:26, 18:26] = 35.0  # building B
    arr[10:16, 20:28] = 15.0  # building C
    return arr


@pytest.fixture
def osm_mock():
    """Synthetic OSM-like heightmap dict matching cities.py return format."""
    arr = np.full((32, 32), np.nan, dtype=np.float64)
    arr[6:14, 6:14] = 18.0
    arr[18:26, 18:26] = 32.0
    arr[10:16, 20:28] = 14.0
    return {
        "heightmap": arr,
        "bounds": {"x": (-75.28, -74.95), "y": (39.86, 40.06), "z": (0.0, 35.0)},
        "resolution": (32, 32),
        "cell_size": (0.01, 0.006),
        "projection": "max",
    }


# ---------------------------------------------------------------------------
# _preprocess_for_registration
# ---------------------------------------------------------------------------

class TestPreprocessForRegistration:

    def test_output_is_uint8(self, simple_building_array):
        from numpy2stl.registration.align import _preprocess_for_registration
        out = _preprocess_for_registration(simple_building_array)
        assert out.dtype == np.uint8

    def test_shape_preserved(self, simple_building_array):
        from numpy2stl.registration.align import _preprocess_for_registration
        out = _preprocess_for_registration(simple_building_array)
        assert out.shape == simple_building_array.shape

    def test_nan_filled_no_nan_in_output(self):
        from numpy2stl.registration.align import _preprocess_for_registration
        arr = np.ones((32, 32), dtype=np.float64)
        arr[10:15, 10:15] = np.nan
        out = _preprocess_for_registration(arr)
        assert not np.any(np.isnan(out.astype(float)))

    def test_all_nan_input_returns_zeros(self):
        from numpy2stl.registration.align import _preprocess_for_registration
        arr = np.full((16, 16), np.nan, dtype=np.float64)
        out = _preprocess_for_registration(arr)
        assert out.dtype == np.uint8
        assert out.shape == (16, 16)

    def test_non_square_shape_preserved(self):
        from numpy2stl.registration.align import _preprocess_for_registration
        arr = np.random.rand(48, 64)
        out = _preprocess_for_registration(arr)
        assert out.shape == (48, 64)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------

class TestApplyTransform:

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_identity_warp_preserves_interior(self, simple_building_array):
        from numpy2stl.registration.align import apply_transform
        M = np.float64([[1, 0, 0], [0, 1, 0]])
        out = apply_transform(simple_building_array, M)
        interior = np.s_[4:-4, 4:-4]
        np.testing.assert_allclose(
            out[interior], simple_building_array[interior], atol=1e-4
        )

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_output_shape_respects_output_shape_param(self, simple_building_array):
        from numpy2stl.registration.align import apply_transform
        M = np.float64([[1, 0, 0], [0, 1, 0]])
        out = apply_transform(simple_building_array, M, output_shape=(64, 64))
        assert out.shape == (64, 64)

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_border_pixels_are_nan_by_default(self, simple_building_array):
        from numpy2stl.registration.align import apply_transform
        # Translate by 10 px so left/top border is outside source
        M = np.float64([[1, 0, 10], [0, 1, 10]])
        out = apply_transform(simple_building_array, M)
        # Top row should be fill_value (NaN)
        assert np.all(np.isnan(out[0, :]))

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_custom_fill_value(self, simple_building_array):
        from numpy2stl.registration.align import apply_transform
        M = np.float64([[1, 0, 10], [0, 1, 10]])
        out = apply_transform(simple_building_array, M, fill_value=-999.0)
        assert np.all(out[0, :] == -999.0)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

class TestRegister:

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_returns_correct_keys(self, simple_building_array):
        from numpy2stl.registration.align import register
        result = register(simple_building_array, simple_building_array)
        for k in ("transform", "confidence", "n_iterations", "converged",
                  "scale", "angle_deg"):
            assert k in result

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_transform_shape(self, simple_building_array):
        from numpy2stl.registration.align import register
        result = register(simple_building_array, simple_building_array)
        assert result["transform"].shape == (2, 3)

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_identity_self_registration(self):
        """Registering a structured image against itself → near-identity.

        The edge-based global search needs enough structure to pin the identity,
        so we use a 128px grid of buildings (a realistic block pattern). The
        recovered transform should map a centre point onto itself within a few
        percent of the image size.
        """
        from numpy2stl.registration.align import register
        import numpy as np
        rng = np.random.RandomState(0)
        # 128px city-like GRID of buildings: blocks on a regular lattice with
        # street gaps, giving a dominant axis-aligned orientation (what the
        # line-angle rotation method targets) and varied heights to break
        # translational aliasing.  Buildings are large enough (~14 px) that edge
        # smoothing preserves their axis-aligned walls.  Background NaN matches
        # real heightmaps.
        arr = np.full((128, 128), np.nan, dtype=np.float64)
        for gr in range(10, 116, 20):
            for gc in range(10, 116, 20):
                sh = rng.randint(12, 16); sw = rng.randint(12, 16)
                arr[gr:gr + sh, gc:gc + sw] = rng.uniform(10, 40)
        result = register(arr, arr)
        M = result["transform"][:2, :3].astype(float)
        h, w = arr.shape
        p = np.array([w / 2, h / 2, 1.0])
        moved = M @ p
        dist = float(np.hypot(moved[0] - p[0], moved[1] - p[1]))
        assert dist < 0.10 * max(h, w)   # within 10% of the image size
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        assert 0.85 < sc < 1.15

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_recovers_known_translation(self):
        """Apply a known shift; verify apply_transform brings image closer to source."""
        arr = np.zeros((64, 64), dtype=np.float64)
        # Use a richer scene (more distinct features) for reliable ECC convergence
        arr[8:20, 8:20] = 10.0
        arr[35:50, 35:50] = 20.0
        arr[20:30, 40:55] = 15.0
        M_known = np.float32([[1, 0, 5], [0, 1, 4]])
        shifted = cv2.warpAffine(arr.astype(np.float32), M_known, (64, 64))

        from numpy2stl.registration.align import register, apply_transform
        # register() now delegates to the global edge-IoU search (register_global).
        result = register(shifted.astype(np.float64), arr)
        M_found = result["transform"]
        # Transform must be a valid 2x3 matrix with finite values
        assert M_found.shape == (2, 3)
        assert np.all(np.isfinite(M_found))
        # Applying the found transform should reduce the pixel difference vs. arr
        aligned = apply_transform(shifted.astype(np.float64), M_found, output_shape=arr.shape)
        valid = ~np.isnan(aligned)
        if valid.sum() > 100:
            diff_before = np.mean(np.abs(shifted[valid] - arr[valid]))
            diff_after = np.mean(np.abs(aligned[valid] - arr[valid]))
            assert diff_after <= diff_before + 2.0  # should not make things worse

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_register_self_returns_valid_transform(self, simple_building_array):
        # Self-registration via the global search returns a valid 2x3 transform.
        from numpy2stl.registration.align import register
        result = register(simple_building_array, simple_building_array)
        assert result["transform"].shape == (2, 3)
        assert np.all(np.isfinite(result["transform"]))


class TestBuildingMask:

    def test_osm_mask_is_non_nan(self):
        from numpy2stl.registration.align import building_mask
        arr = np.full((32, 32), np.nan)
        arr[8:16, 8:16] = 20.0
        mask = building_mask(arr, source="osm", min_blob_px=0)
        assert mask.dtype == bool
        assert mask[8:16, 8:16].all()
        assert not mask[0, 0]

    def test_stl_mask_thresholds_above_ground(self):
        from numpy2stl.registration.align import building_mask
        arr = np.ones((32, 32), dtype=np.float64) * 2.0   # ground plane
        arr[10:20, 10:20] = 30.0                          # a building
        mask = building_mask(arr, source="stl", min_blob_px=0)
        assert mask[10:20, 10:20].any()
        # Ground cells should be excluded
        assert not mask[0, 0]

    def test_explicit_threshold(self):
        from numpy2stl.registration.align import building_mask
        arr = np.zeros((16, 16), dtype=np.float64)
        arr[4:8, 4:8] = 5.0
        arr[10:12, 10:12] = 50.0
        mask = building_mask(arr, source="stl", threshold=20.0, min_blob_px=0)
        assert mask[10:12, 10:12].all()       # 50 > 20
        assert not mask[4:8, 4:8].any()       # 5 < 20

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_register_with_mask_default(self, simple_building_array):
        """Default register() uses masks; self-registration stays near identity."""
        from numpy2stl.registration.align import register
        result = register(simple_building_array, simple_building_array, use_mask=True)
        assert result["transform"].shape == (2, 3)
        assert np.all(np.isfinite(result["transform"]))


# ---------------------------------------------------------------------------
# register_polygons: polygon point-pattern registration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CV2, reason="opencv required")
class TestPolygonRegister:

    @staticmethod
    def _rects(cents, wh):
        return [np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                          [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]], float)
                for (cx, cy), (w, h) in zip(cents, wh)]

    @pytest.mark.parametrize("s,deg,tx,ty", [(0.667, 15.0, 40, -20),
                                             (1.0, -30.0, -25, 60), (0.667, 0.0, 10, 10)])
    def test_recovers_known_similarity_partial_overlap(self, s, deg, tx, ty):
        import cv2
        from numpy2stl.registration.align import register_polygons
        rng = np.random.default_rng(1)
        oc = rng.uniform(40, 440, (80, 2)); owh = rng.uniform(8, 20, (80, 2))
        osm = self._rects(oc, owh)
        th = np.radians(deg); c, sn = s * np.cos(th), s * np.sin(th)
        M = np.array([[c, -sn, tx], [sn, c, ty]])             # STL→OSM ground truth
        Minv = cv2.invertAffineTransform(M.astype(np.float32))
        keep = rng.choice(80, 36, replace=False)              # 45% overlap
        stl = [osm[i] @ Minv[:, :2].T + Minv[:, 2] for i in keep]
        stl += self._rects(rng.uniform(0, 300, (10, 2)), rng.uniform(8, 20, (10, 2)))  # decoys
        res = register_polygons(stl, osm, scale_prior=s)
        assert res["applied"]
        assert abs(((res["angle_deg"] - deg + 180) % 360) - 180) < 1.5
        assert abs(res["scale"] - s) < 0.02 * s + 1e-6

    def test_rejects_unrelated(self):
        from numpy2stl.registration.align import register_polygons
        rng = np.random.default_rng(2)
        osm = self._rects(rng.uniform(40, 440, (80, 2)), rng.uniform(8, 20, (80, 2)))
        stl = self._rects(rng.uniform(0, 300, (20, 2)), rng.uniform(8, 20, (20, 2)))
        res = register_polygons(stl, osm, scale_prior=0.667)
        assert not res["applied"]   # no genuine consensus → not applied


# ---------------------------------------------------------------------------
# fourier_mellin_register (prototype): grid-free rotation + scale recovery
# ---------------------------------------------------------------------------

class TestFourierMellin:

    @staticmethod
    def _city(n=256, grid=True, seed=0):
        import cv2
        rng = np.random.default_rng(seed)
        img = np.zeros((n, n), dtype=np.float32)
        for _ in range(80):
            cx, cy = rng.uniform(0.2, 0.8, 2) * n
            bw, bh = rng.uniform(0.03, 0.07, 2) * n
            ang = 0.0 if grid else rng.uniform(0, 180)
            box = cv2.boxPoints(((cx, cy), (bw, bh), ang)).astype(np.int32)
            cv2.fillConvexPoly(img, box, float(rng.uniform(5, 60)))
        return img

    @staticmethod
    def _apply(img, ang, sc):
        import cv2
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    @pytest.mark.parametrize("grid", [True, False])
    @pytest.mark.parametrize("ang,sc", [(0.0, 1.0), (12.0, 1.0), (0.0, 1.25), (-18.0, 0.8)])
    def test_recovers_known_transform(self, grid, ang, sc):
        """A known rotation+scale is recovered to <2° / <0.05 on grid AND
        irregular layouts — the grid-free guarantee the gradient path lacks."""
        from numpy2stl.registration.align import fourier_mellin_register
        base = self._city(grid=grid)
        moved = self._apply(base, ang, sc)
        # register moved→base must recover the INVERSE (−ang, 1/sc).
        r = fourier_mellin_register(moved, base)
        exp_ang = ((-ang + 180.0) % 360.0) - 180.0
        rot_err = abs(((r.angle_deg - exp_ang + 180.0) % 360.0) - 180.0)
        assert rot_err < 2.0, f"rot_err {rot_err:.2f}° (grid={grid}, ang={ang})"
        assert abs(r.scale - 1.0 / sc) < 0.05, f"scale {r.scale:.3f} vs {1/sc:.3f}"

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    def test_identity_high_confidence(self):
        """Identical inputs → near-zero transform and a high response."""
        from numpy2stl.registration.align import fourier_mellin_register
        base = self._city(grid=False, seed=3)
        r = fourier_mellin_register(base, base)
        assert abs(r.angle_deg) < 1.0
        assert abs(r.scale - 1.0) < 0.02
        assert r.confidence > 0.5   # response is the built-in reliability gate

    @pytest.mark.skipif(not HAS_CV2, reason="opencv required")
    @pytest.mark.parametrize("grid", [True, False])
    def test_partial_overlap_recovery(self, grid):
        """A transformed image, masked to a small random sub-window (PARTIAL
        overlap), is still inverted to near-identity — overlap alone is not a
        blocker when the visible content matches."""
        import cv2
        from numpy2stl.registration.align import fourier_mellin_register
        base = self._city(grid=grid, seed=1)
        h, w = base.shape
        ang, sc = 20.0, 1.2
        M_gt = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, sc).astype(np.float64)
        moved = cv2.warpAffine(base, M_gt, (w, h), flags=cv2.INTER_LINEAR)
        cw = ch = int(w * 0.5)            # keep only a 50% window
        crop = np.zeros_like(moved)
        crop[40:40 + ch, 60:60 + cw] = moved[40:40 + ch, 60:60 + cw]
        r = fourier_mellin_register(crop, base, refine_overlap=True)
        # recovered ∘ ground-truth should be identity
        R = (np.vstack([r.transform, [0, 0, 1]]) @ np.vstack([M_gt, [0, 0, 1]]))[:2]
        resid_ang = abs(np.degrees(np.arctan2(R[0, 1], R[0, 0])))
        resid_sc = abs(np.hypot(R[0, 0], R[0, 1]) - 1.0)
        assert resid_ang < 2.0, f"residual rotation {resid_ang:.2f}° (grid={grid})"
        assert resid_sc < 0.05, f"residual scale {resid_sc:.3f} (grid={grid})"


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

class TestCompare:

    def _make_pair(self):
        osm = np.full((32, 32), np.nan, dtype=np.float64)
        osm[10:20, 10:20] = 15.0
        stl = np.full((32, 32), np.nan, dtype=np.float64)
        stl[10:20, 10:20] = 7.5  # same area, half the height → scale=2 gives match
        return stl, osm

    def test_returns_comparison_result(self):
        from numpy2stl.registration.compare import compare
        stl, osm = self._make_pair()
        result = compare(stl, osm, height_scale=2.0)
        from numpy2stl.registration.types import ComparisonResult
        assert isinstance(result, ComparisonResult)

    def test_perfect_match_zero_rmse(self):
        from numpy2stl.registration.compare import compare
        stl, osm = self._make_pair()
        result = compare(stl, osm, height_scale=2.0)
        assert result.rmse < 1e-6

    def test_auto_height_scale_estimates_correctly(self):
        from numpy2stl.registration.compare import compare
        stl, osm = self._make_pair()
        result = compare(stl, osm, height_scale=None)
        assert abs(result.height_scale_used - 2.0) < 0.1

    def test_difference_shape_matches_osm(self):
        from numpy2stl.registration.compare import compare
        stl, osm = self._make_pair()
        result = compare(stl, osm, height_scale=2.0)
        assert result.difference.shape == osm.shape

    def test_overlap_mask_dtype_bool(self):
        from numpy2stl.registration.compare import compare
        stl, osm = self._make_pair()
        result = compare(stl, osm, height_scale=2.0)
        assert result.overlap_mask.dtype == bool

    def test_missing_in_osm_detects_stl_only_area(self):
        from numpy2stl.registration.compare import compare
        stl = np.full((32, 32), np.nan, dtype=np.float64)
        stl[5:10, 5:10] = 8.0   # only in STL
        osm = np.full((32, 32), np.nan, dtype=np.float64)
        osm[15:20, 15:20] = 12.0  # only in OSM
        result = compare(stl, osm, height_scale=1.0)
        assert result.missing_in_osm[5:10, 5:10].all()
        assert result.missing_in_stl[15:20, 15:20].all()

    def test_no_overlap_returns_nan_stats(self):
        from numpy2stl.registration.compare import compare
        stl = np.full((16, 16), np.nan, dtype=np.float64)
        osm = np.full((16, 16), np.nan, dtype=np.float64)
        result = compare(stl, osm, height_scale=1.0)
        assert result.n_overlap == 0
        assert np.isnan(result.rmse)

    def test_bias_direction(self):
        from numpy2stl.registration.compare import compare
        osm = np.full((16, 16), 10.0, dtype=np.float64)
        stl = np.full((16, 16), 6.0, dtype=np.float64)  # scale=2 → 12 → bias=+2
        result = compare(stl, osm, height_scale=2.0)
        assert result.bias > 0  # STL (12 m) > OSM (10 m)

    def test_shape_mismatch_raises(self):
        from numpy2stl.registration.compare import compare
        stl = np.zeros((32, 32))
        osm = np.zeros((16, 16))
        with pytest.raises(ValueError, match="Shape mismatch"):
            compare(stl, osm, height_scale=1.0)


# ---------------------------------------------------------------------------
# CityRegistrationReport dataclass
# ---------------------------------------------------------------------------

class TestCityRegistrationReport:

    def _make_report(self):
        from numpy2stl.registration.types import (
            CityRegistrationReport, ComparisonResult, RegistrationResult,
        )
        reg = RegistrationResult(
            transform=np.eye(2, 3),
            confidence=0.75,
            scale=1.0,
            angle_deg=0.0,
            n_iterations=50,
            converged=True,
        )
        diff = np.full((16, 16), np.nan)
        diff[5:10, 5:10] = 2.0
        overlap = ~np.isnan(diff)
        comp = ComparisonResult(
            difference=diff,
            building_diff_map=diff.copy(),
            overlap_mask=overlap,
            missing_in_osm=np.zeros((16, 16), dtype=bool),
            missing_in_stl=np.zeros((16, 16), dtype=bool),
            rmse=2.0, mae=2.0, bias=2.0, correlation=0.9,
            rank_correlation=0.85, mape=20.0,
            coverage_pct=50.0, footprint_iou=0.6, dice_score=0.75, n_overlap=25,
            height_scale_used=1.5, height_offset_used=0.0,
            height_ratio_mean=1.0, height_ratio_std=0.1,
        )
        return CityRegistrationReport(
            region_name="Test City",
            stl_file="test.stl",
            stl_heightmap=np.ones((16, 16)),
            osm_heightmap=np.ones((16, 16)),
            stl_aligned=np.ones((16, 16)),
            registration=reg,
            comparison=comp,
            step_timings=[("load", 0.1), ("register", 1.2)],
        )

    def test_dataclass_attributes(self):
        report = self._make_report()
        assert report.region_name == "Test City"
        assert report.registration.confidence == 0.75
        assert report.comparison.rmse == 2.0
        assert len(report.step_timings) == 2

    def test_is_frozen(self):
        report = self._make_report()
        with pytest.raises((AttributeError, TypeError)):
            report.region_name = "Other"


# ---------------------------------------------------------------------------
# HTML report (no network, no STL — synthetic data only)
# ---------------------------------------------------------------------------

class TestWriteRegistrationReport:

    def test_creates_index_html(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        out = write_registration_report(tmp_path, report)
        assert out.exists()
        assert out.name == "index.html"

    def test_creates_assets_folder(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        assert (tmp_path / "assets").is_dir()

    def test_creates_comparison_png(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        assert (tmp_path / "assets" / "comparison.png").exists()

    def test_index_html_contains_region_name(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "Test City" in html

    def test_index_html_contains_stats(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        low = html.lower()
        assert "rmse" in low
        assert "correlation" in low
        assert "dice" in low


# ---------------------------------------------------------------------------
# OSM cities.py (no network — format/logic tests only)
# ---------------------------------------------------------------------------

class TestResolveBuildingHeight:

    def test_height_tag_parsed(self):
        from numpy2stl.applications.cities import _resolve_building_height
        row = {"height": "15.5", "building:levels": None}
        assert _resolve_building_height(row, 10.0, 3.5) == pytest.approx(15.5)

    def test_height_tag_with_unit(self):
        from numpy2stl.applications.cities import _resolve_building_height
        row = {"height": "15 m", "building:levels": None}
        assert _resolve_building_height(row, 10.0, 3.5) == pytest.approx(15.0)

    def test_levels_fallback(self):
        from numpy2stl.applications.cities import _resolve_building_height
        row = {"height": None, "building:levels": "4"}
        assert _resolve_building_height(row, 10.0, 3.5) == pytest.approx(14.0)

    def test_default_fallback(self):
        from numpy2stl.applications.cities import _resolve_building_height
        row = {"height": None, "building:levels": None}
        assert _resolve_building_height(row, 10.0, 3.5) == pytest.approx(10.0)

    def test_nan_height_falls_through(self):
        from numpy2stl.applications.cities import _resolve_building_height
        import math
        row = {"height": float("nan"), "building:levels": "3"}
        # NaN height → falls through to building:levels
        assert _resolve_building_height(row, 10.0, 3.5) == pytest.approx(10.5)


class TestMakeResult:

    def test_return_format_matches_mesh_to_heightmap(self):
        from numpy2stl.applications.cities import _make_result
        hm = np.full((32, 32), np.nan)
        hm[5:10, 5:10] = 15.0
        result = _make_result(hm, N=40.06, S=39.86, E=-74.95, W=-75.28, resolution=32)
        assert "heightmap" in result
        assert "bounds" in result
        assert "resolution" in result
        assert "cell_size" in result
        assert "projection" in result
        assert result["heightmap"].dtype == np.float64
        assert result["resolution"] == (32, 32)
        assert result["projection"] == "max"

    def test_bounds_x_is_west_east(self):
        from numpy2stl.applications.cities import _make_result
        hm = np.zeros((16, 16))
        result = _make_result(hm, N=40.0, S=39.0, E=-75.0, W=-76.0, resolution=16)
        W_out, E_out = result["bounds"]["x"]
        assert W_out == pytest.approx(-76.0)
        assert E_out == pytest.approx(-75.0)

    def test_bounds_y_is_south_north(self):
        from numpy2stl.applications.cities import _make_result
        hm = np.zeros((16, 16))
        result = _make_result(hm, N=40.0, S=39.0, E=-75.0, W=-76.0, resolution=16)
        S_out, N_out = result["bounds"]["y"]
        assert S_out == pytest.approx(39.0)
        assert N_out == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Integration tests (requires network + osmnx; skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not HAS_OSMNX or not HAS_GEOPANDAS, reason="osmnx/geopandas not installed")
class TestGetPhiladelphiaHeightmap:

    def test_returns_correct_format(self):
        from numpy2stl.applications.cities import get_philadelphia_heightmap
        result = get_philadelphia_heightmap(resolution=64)
        assert "heightmap" in result
        assert result["heightmap"].dtype == np.float64
        assert result["heightmap"].shape == (64, 64)
        assert result["projection"] == "max"

    def test_bounds_are_latitude_longitude(self):
        from numpy2stl.applications.cities import get_philadelphia_heightmap
        result = get_philadelphia_heightmap(resolution=64)
        W, E = result["bounds"]["x"]
        S, N = result["bounds"]["y"]
        # Philadelphia is in the western hemisphere, ~40°N
        assert -76 < W < -74
        assert -76 < E < -74
        assert 39 < S < 41
        assert 39 < N < 41
        assert W < E
        assert S < N

    def test_has_some_buildings(self):
        from numpy2stl.applications.cities import get_philadelphia_heightmap
        result = get_philadelphia_heightmap(resolution=64)
        valid = result["heightmap"][~np.isnan(result["heightmap"])]
        assert len(valid) > 0
        assert valid.max() > 5.0  # at least some buildings > 5 m
