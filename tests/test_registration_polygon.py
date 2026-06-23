# Registration tests — polygon (split from test_registration.py, B6).
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

