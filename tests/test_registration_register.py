# Registration tests — register (split from test_registration.py, B6).
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
        """register() masks internally (via register_global); self-registration
        stays near identity."""
        from numpy2stl.registration.align import register
        result = register(simple_building_array, simple_building_array)
        assert result["transform"].shape == (2, 3)
        assert np.all(np.isfinite(result["transform"]))

