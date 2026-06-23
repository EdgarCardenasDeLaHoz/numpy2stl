# Registration tests — cities (split from test_registration.py, B6).
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

