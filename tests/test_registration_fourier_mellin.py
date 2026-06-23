# Registration tests — fourier_mellin (split from test_registration.py, B6).
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

