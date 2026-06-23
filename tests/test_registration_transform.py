# Registration tests — transform (split from test_registration.py, B6).
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

