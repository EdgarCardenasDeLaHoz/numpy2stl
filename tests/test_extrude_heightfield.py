"""core.extrude / core.heightfield / processing.boolean public helpers."""

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from numpy2stl.core.extrude import prism
from numpy2stl.core.heightfield import tin_solid
from numpy2stl.processing.boolean import from_manifold, union


def test_prism_with_hole_is_a_closed_solid():
    p = Polygon([(0, 0), (10, 0), (10, 8), (0, 8)], [[(3, 3), (6, 3), (6, 5), (3, 5)]])
    v, f = prism(p, 1.0, 4.0)
    m = trimesh.Trimesh(v, f)
    assert m.is_watertight and m.is_winding_consistent
    assert m.volume == pytest.approx((80 - 6) * 3.0)


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
