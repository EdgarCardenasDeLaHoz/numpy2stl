"""Jigsaw cutters and cutting a model into interlocking pieces."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
mfd = pytest.importorskip("manifold3d")
pytest.importorskip("pymeshlab")

from numpy2stl import array_to_mesh  # noqa: E402
from numpy2stl.applications.puzzle import (  # noqa: E402
    jigsaw_outlines,
    make_base_border,
    make_jigsaw_cutters,
    make_puzzle_model,
    make_puzzle_pts,
)
from numpy2stl.processing.boolean import cut_jigsaw, mesh_volume  # noqa: E402

WX, WY, COLS, ROWS = 200.0, 144.0, 4, 3
KNOB_W, KNOB_D, CLR = 12.0, 6.0, 0.3


def _box():
    box = trimesh.creation.box((WX, WY, 20.0))
    box.apply_translation((WX / 2, WY / 2, 10.0))
    return box.vertices, box.faces


def _bumpy():
    # 1 px = 1 mm: 201 x 145 samples span exactly 200 x 144 mm.
    y, x = np.mgrid[0:145, 0:201]
    A = 8 + 3 * np.sin(x / 9.0) * np.cos(y / 7.0)
    return array_to_mesh(A, floor_val=0)


def _manifold(v, f):
    return mfd.Manifold(
        mfd.Mesh64(
            vert_properties=np.array(v, dtype=np.float64, order="C"),
            tri_verts=np.array(f, dtype=np.uint64, order="C"),
        )
    )


@pytest.fixture(scope="module")
def cutters():
    return make_jigsaw_cutters(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, clearance=CLR, z0=-1, z1=100)


def test_cutters_tile_the_rectangle(cutters):
    assert sorted(cutters) == sorted(f"r{r}c{c}" for r in range(ROWS) for c in range(COLS))
    allv = np.vstack([v for v, _ in cutters.values()])
    np.testing.assert_allclose(allv[:, :2].min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(allv[:, :2].max(axis=0), [WX + 1.0, WY + 1.0])
    for v, f in cutters.values():
        m = trimesh.Trimesh(v, f)
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0

    # Areas: the whole rectangle (+ border) minus one clearance strip per
    # interior edge and the groove clearance around each tongue.
    outlines = jigsaw_outlines(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, CLR)
    total = sum(p.area for p in outlines.values())
    n_edges = (COLS - 1) * ROWS + (ROWS - 1) * COLS
    gaps = (COLS - 1) * (WY + 2) * CLR + (ROWS - 1) * (WX + 2) * CLR
    gaps -= (COLS - 1) * (ROWS - 1) * CLR * CLR  # where the strips cross
    gaps += n_edges * 2 * CLR * KNOB_D  # groove sides
    np.testing.assert_allclose(total, (WX + 2) * (WY + 2) - gaps, rtol=1e-9)


def test_knob_too_big_rejected():
    with pytest.raises(ValueError):
        make_jigsaw_cutters(100, 100, 4, 4, knob_width=20, knob_depth=10)


@pytest.mark.parametrize("model", [_box, _bumpy], ids=["box", "heightmap"])
def test_cut_non_square_model(model, cutters):
    vertices, faces = model()
    pieces = cut_jigsaw(vertices, faces, cutters)
    assert len(pieces) == COLS * ROWS

    w, h = WX / COLS, WY / ROWS
    v_in = abs(mesh_volume(vertices, faces))
    v_out = 0.0
    solids = {}
    for key, (v, f) in pieces.items():
        m = trimesh.Trimesh(v, f)
        assert m.is_watertight and m.is_winding_consistent, key
        assert m.body_count == 1, key
        v_out += m.volume
        solids[key] = _manifold(v, f)

        r, c = int(key[1 : key.index("c")]), int(key[key.index("c") + 1 :])
        lo, hi = m.bounds[0, :2], m.bounds[1, :2]
        tol = KNOB_D + 1e-6
        assert lo[0] >= c * w - tol and hi[0] <= (c + 1) * w + tol, key
        assert lo[1] >= r * h - tol and hi[1] <= (r + 1) * h + tol, key

    # Only the clearance slivers are missing.
    lost = (v_in - v_out) / v_in
    assert 0 < lost < 0.02

    # Neighbours don't overlap, and each tongue sits in the neighbour's groove.
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in [(0, 1), (1, 0)]:
                if r + dr >= ROWS or c + dc >= COLS:
                    continue
                a, b = f"r{r}c{c}", f"r{r + dr}c{c + dc}"
                assert (solids[a] ^ solids[b]).volume() < 1e-6 * v_in
                ba, bb = trimesh.Trimesh(*pieces[a]).bounds, trimesh.Trimesh(*pieces[b]).bounds
                axis = 0 if dc else 1
                edge = (c + 1) * w if dc else (r + 1) * h
                # The tongue of one crosses the shared edge into the other's cell.
                crosses = ba[1, axis] > edge + KNOB_D - 1e-6 or bb[0, axis] < edge - KNOB_D + 1e-6
                assert crosses, (a, b)


def test_cut_detects_uncovered_model(cutters):
    vertices, faces = _box()
    partial = {k: v for k, v in cutters.items() if not k.endswith(f"c{COLS - 1}")}
    with pytest.raises(ValueError, match="beyond the clearance"):
        cut_jigsaw(vertices, faces, partial)


def test_pymeshlab_engine(cutters):
    vertices, faces = _box()
    pieces = cut_jigsaw(vertices, faces, cutters, engine="pymeshlab")
    assert len(pieces) == COLS * ROWS


def test_old_api_axes_follow_width_tuple():
    models = make_puzzle_model((200, 144), b=50, m=15, base_n=5)
    assert len(models) == 4 * 3
    allv = np.vstack([v for v, _ in models.values()])
    np.testing.assert_allclose(allv[:, :2].min(axis=0), [-5, -5])
    np.testing.assert_allclose(allv[:, :2].max(axis=0), [205, 149])

    pts = make_puzzle_pts((200, 144), b=50, m=15, base_n=5)
    assert len(pts) == 12 and np.allclose(pts[0][0], pts[0][-1])

    borders = make_base_border((200, 144), b=50, m=15, base_n=5, height=1, offset_dist=5)
    assert len(borders) == 12
    bv = np.vstack([v for v, _ in borders.values()])
    np.testing.assert_allclose(bv[:, :2].max(axis=0), [200, 144])
    assert all(trimesh.Trimesh(*vf).is_watertight for vf in borders.values())
