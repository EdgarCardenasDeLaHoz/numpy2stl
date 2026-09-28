"""core.extrude / core.heightfield / processing.boolean public helpers."""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from numpy2stl.core.extrude import prism, prisms
from numpy2stl.core.heightfield import tin_solid
from numpy2stl.processing.boolean import from_manifold, union


def test_prism_with_hole_is_a_closed_solid():
    p = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)], [[(3, 3), (6, 3), (6, 5), (3, 5)]])
    v, f = prism(p, 1.0, 4.0)
    m = trimesh.Trimesh(v, f)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume == pytest.approx((80 - 6) * 3.0)


def test_prisms_match_prism():
    polys = [Polygon([(0, 0), (10, 0), (10, 8), (0, 8)], [[(3, 3), (6, 3), (6, 5), (3, 5)]]),
             Polygon([(0, 0), (0, 5), (5, 5), (5, 0)]),                  # clockwise input
             Polygon([(20, 0), (30, 0), (30, 2), (22, 2), (22, 9), (20, 9)]),   # concave
             Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    z0, z1 = np.array([1.0, 0.0, 2.0, 3.0]), np.array([4.0, 2.5, 7.0, 3.0])
    got = prisms(polys, z0, z1)
    assert got[3] is None                                                  # zero height
    for p, a, b, g in zip(polys[:3], z0[:3], z1[:3], got[:3], strict=True):
        v, f = g
        rv, rf = prism(p, a, b)
        assert len(v) == len(rv) and len(f) == len(rf)
        m, r = trimesh.Trimesh(v, f), trimesh.Trimesh(rv, rf)
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0
        assert m.volume == pytest.approx(r.volume)
        assert np.allclose(np.sort(v, axis=0), np.sort(rv, axis=0))


def test_tin_solid_is_watertight_with_floor():
    y, x = np.mgrid[0:40, 0:50]
    z = 5 + 3 * np.exp(-(((x - 25) / 8) ** 2 + ((y - 20) / 6) ** 2))
    v, f = tin_solid(z, 0.05, mm_per_px=0.5)
    m = trimesh.Trimesh(v, f)
    assert m.is_watertight and m.volume > 0
    assert v[:, 2].min() == 0.0
    assert m.extents[0] == pytest.approx(49 * 0.5)


def test_union_reports_rejected_meshes():
    a = prism(Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), 0, 1)
    b = prism(Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), 0, 1)
    broken = (a[0], a[1][:-1])
    u, rejected = union([a, b, broken])
    assert rejected == 1
    v, f = from_manifold(u)
    assert trimesh.Trimesh(v, f).volume == pytest.approx(7.0)
