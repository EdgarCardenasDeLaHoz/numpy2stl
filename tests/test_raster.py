# Tests for numpy2stl.raster: burn and fill.
import ast
from pathlib import Path

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon, box  # noqa: E402

from numpy2stl.raster import burn, burn_polygons, fill_nan  # noqa: E402

# 10×10 grid over (0, 0)–(10, 10): one unit per cell.
BOUNDS = (0.0, 0.0, 10.0, 10.0)


@pytest.fixture(params=["rasterio", "shapely"])
def backend(request, monkeypatch):
    if request.param == "rasterio":
        if not burn.HAS_RASTERIO:
            pytest.skip("rasterio not installed")
    else:
        monkeypatch.setattr(burn, "HAS_RASTERIO", False)
    return request.param


class TestBurnPolygons:

    def test_bounds_are_north_up(self, backend):
        # A strip along the south edge (y 0..2) lands in the LAST two rows.
        out = burn_polygons([box(0, 0, 10, 2)], (10, 10), bounds=BOUNDS)
        assert out[-2:].all() and not out[:-2].any()

    def test_modes(self, backend):
        a, b = box(0, 0, 6, 10), box(4, 0, 10, 10)     # overlap on columns 4, 5
        mx = burn_polygons([a, b], (10, 10), bounds=BOUNDS, values=[3.0, 1.0], mode="max")
        sm = burn_polygons([a, b], (10, 10), bounds=BOUNDS, values=[3.0, 1.0], mode="sum")
        st = burn_polygons([a, b], (10, 10), bounds=BOUNDS, values=[3.0, 1.0], mode="set")
        assert (mx[:, 4:6] == 3.0).all() and (mx[:, 6:] == 1.0).all()
        assert (sm[:, 4:6] == 4.0).all() and (sm[:, :4] == 3.0).all()
        assert (st[:, 4:6] == 1.0).all()                 # last polygon wins

    def test_holes_are_not_burnt(self, backend):
        ring = Polygon(box(0, 0, 10, 10).exterior.coords, [box(3, 3, 7, 7).exterior.coords])
        out = burn_polygons([ring], (10, 10), bounds=BOUNDS, values=5.0)
        assert (out[3:7, 3:7] == 0.0).all()
        assert out.sum() == 5.0 * (100 - 16)

    def test_fill_and_transform(self, backend):
        # Same grid through an explicit affine (a, b, c, d, e, f), fill NaN.
        t = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        out = burn_polygons([box(0, 0, 10, 2)], (10, 10), transform=t, values=2.0, fill=np.nan)
        assert np.isnan(out[:-2]).all() and (out[-2:] == 2.0).all()

    def test_geojson_and_ring_inputs(self, backend):
        ring = np.array([[0, 0], [10, 0], [10, 2], [0, 2]], dtype=float)
        gj = {"type": "Polygon", "coordinates": [ring.tolist()]}
        a = burn_polygons([ring], (10, 10), bounds=BOUNDS)
        b = burn_polygons([gj], (10, 10), bounds=BOUNDS)
        np.testing.assert_array_equal(a, b)

    def test_errors(self):
        with pytest.raises(ValueError, match="mode"):
            burn_polygons([box(0, 0, 1, 1)], (4, 4), bounds=BOUNDS, mode="min")
        with pytest.raises(ValueError, match="values"):
            burn_polygons([box(0, 0, 1, 1)], (4, 4), bounds=BOUNDS, values=[1.0, 2.0])
        with pytest.raises(ValueError, match="transform"):
            burn_polygons([box(0, 0, 1, 1)], (4, 4))


class TestFillNan:

    def _arr(self):
        a = np.arange(25, dtype=np.float64).reshape(5, 5)
        a[2, 2] = np.nan      # interior hole
        a[0, 0] = np.nan      # border-connected
        return a

    def test_no_nan_returns_input(self):
        a = np.ones((3, 3))
        assert fill_nan(a) is a

    def test_nearest(self):
        out = fill_nan(self._arr())
        assert np.isfinite(out).all()
        assert out[2, 2] in (7.0, 11.0, 13.0, 17.0)

    def test_nearest_interior_only(self):
        out = fill_nan(self._arr(), interior_only=True)
        assert np.isfinite(out[2, 2]) and np.isnan(out[0, 0])

    def test_median_and_constant(self):
        a = self._arr()
        assert (fill_nan(a, "median")[[0, 2], [0, 2]] == np.nanmedian(a)).all()
        assert (fill_nan(a, "constant", value=-1.0)[[0, 2], [0, 2]] == -1.0).all()
        assert np.isnan(a[2, 2])                          # input untouched

    def test_float32_kept(self):
        assert fill_nan(self._arr().astype(np.float32), "median").dtype == np.float32

    def test_matches_registration_inpaint(self):
        from numpy2stl.registration.stages._common import _inpaint_stl_nan
        a = self._arr()
        np.testing.assert_array_equal(_inpaint_stl_nan(a),
                                      fill_nan(a, "nearest", interior_only=True))


def test_lower_layers_do_not_import_registration_or_applications():
    """raster / processing / stl2numpy / core / io / utils never import upward."""
    root = Path(__file__).resolve().parents[1] / "src" / "numpy2stl"
    bad = []
    for pkg in ("raster", "processing", "stl2numpy", "core", "io", "utils"):
        for py in (root / pkg).rglob("*.py"):
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    mod = ("." * node.level) + (node.module or "")
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                else:
                    continue
                if "registration" in mod or "applications" in mod:
                    bad.append(f"{py.relative_to(root)}: {mod}")
    assert not bad, bad
