# Registration tests — compare (split from test_registration.py, B6).
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

