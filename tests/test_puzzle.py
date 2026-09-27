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


# ---------------------------------------------------------------------------
# Knob shapes, explicit edges, heightfield pieces, marks, plate layout
# ---------------------------------------------------------------------------
from numpy2stl.applications.puzzle import (  # noqa: E402
    KNOB_SHAPES,
    engrave_underside,
    heightfield_pieces,
    plate_layout,
    underside_marks,
    validate_edges,
)


@pytest.mark.parametrize("shape", KNOB_SHAPES)
def test_knob_shapes_make_watertight_cutters_that_tile(shape):
    import shapely

    outlines = jigsaw_outlines(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, 0.0, knob_shape=shape)
    union = shapely.union_all(list(outlines.values()))
    # Without clearance the pieces tile the (bordered) rectangle exactly, no overlaps.
    np.testing.assert_allclose(union.area, (WX + 2) * (WY + 2), rtol=1e-9)
    np.testing.assert_allclose(sum(p.area for p in outlines.values()), union.area, rtol=1e-9)
    cutters = make_jigsaw_cutters(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, clearance=CLR,
                                  z0=-1, z1=100, knob_shape=shape)
    for v, f in cutters.values():
        m = trimesh.Trimesh(v, f)
        assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    vertices, faces = _box()
    pieces = cut_jigsaw(vertices, faces, cutters)
    lost = 1 - sum(abs(mesh_volume(v, f)) for v, f in pieces.values()) / abs(mesh_volume(vertices, faces))
    assert len(pieces) == COLS * ROWS and 0 < lost < 0.02


def test_knob_shapes_differ():
    areas = {s: jigsaw_outlines(WX, WY, 2, 1, KNOB_W, KNOB_D, 0.0, knob_shape=s)["r0c0"].area
             for s in KNOB_SHAPES}
    assert len({round(a, 6) for a in areas.values()}) == 3
    with pytest.raises(ValueError, match="knob_shape"):
        jigsaw_outlines(WX, WY, 2, 1, KNOB_W, KNOB_D, knob_shape="star")


def test_explicit_edges_set_the_cells():
    cols_e, rows_e = [0, 30, 120, 200], [0, 100, 144]
    outlines = jigsaw_outlines(WX, WY, 3, 2, KNOB_W, KNOB_D, CLR, border=0.0,
                               col_edges=cols_e, row_edges=rows_e)
    b = outlines["r0c0"].bounds   # a tongue may reach past the cell, by at most the knob depth
    assert b[0] == 0 and b[1] == 0 and abs(b[2] - 30) <= KNOB_D + 1e-9 and abs(b[3] - 100) <= KNOB_D + 1e-9
    assert outlines["r1c2"].bounds[2] == pytest.approx(200) and outlines["r1c2"].bounds[3] == pytest.approx(144)
    # End points within the slack snap to the model edge.
    np.testing.assert_allclose(validate_edges([0.3, 50, 99.6], 100), [0, 50, 100])
    for bad, count in (([0, 60, 50, 100], None), ([0, 100], 2), ([10, 50, 100], None),
                       ([0, 50, float("nan"), 100], None)):
        with pytest.raises(ValueError):
            validate_edges(bad, 100, count=count)
    with pytest.raises(ValueError, match="does not fit"):
        jigsaw_outlines(WX, WY, 3, 2, KNOB_W, KNOB_D, CLR, col_edges=[0, 10, 120, 200])


def _heightfield():
    y, x = np.mgrid[0:145, 0:201]
    return 8 + 3 * np.sin(x / 9.0) * np.cos(y / 7.0)


@pytest.mark.parametrize("shape", ["classic", "rectangular"])
def test_heightfield_pieces_conserve_volume(shape):
    from numpy2stl.core.heightfield import tin_solid

    z = _heightfield()
    v, f = tin_solid(z, 0.05)
    whole = abs(mesh_volume(v, f))
    for clr in (0.0, CLR):
        outlines = jigsaw_outlines(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, clr, knob_shape=shape)
        pieces = heightfield_pieces(z, outlines, 0.05)
        assert sorted(pieces) == sorted(outlines)
        for pv, pf in pieces.values():
            m = trimesh.Trimesh(pv, pf)
            assert m.is_watertight and m.is_winding_consistent and m.volume > 0
        vol = sum(abs(mesh_volume(pv, pf)) for pv, pf in pieces.values())
        cutters = make_jigsaw_cutters(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, clr, z0=-1, z1=50,
                                      knob_shape=shape)
        boolean = sum(abs(mesh_volume(pv, pf)) for pv, pf in cut_jigsaw(v, f, cutters).values())
        # Same pieces as the boolean cut of the whole-model mesh, to well under 0.1 %.
        assert vol == pytest.approx(boolean, rel=1e-3)
        if clr == 0:
            assert vol == pytest.approx(whole, rel=1e-3)
        # Walls follow the outline, not the pixel grid: the knob tip is off-grid.
        key = "r0c0"
        np.testing.assert_allclose(pieces[key][0][:, 0].max(), outlines[key].bounds[2], atol=1e-6)


def test_heightfield_pieces_detect_uncovered_model():
    z = _heightfield()
    outlines = jigsaw_outlines(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, CLR)
    partial = {k: p for k, p in outlines.items() if not k.endswith(f"c{COLS - 1}")}
    with pytest.raises(ValueError, match="beyond the clearance"):
        heightfield_pieces(z, partial, 0.05)


def test_engraved_marks_remove_their_own_volume():
    z = _heightfield()
    outlines = jigsaw_outlines(WX, WY, COLS, ROWS, KNOB_W, KNOB_D, CLR)
    v, f = heightfield_pieces(z, {"r1c2": outlines["r1c2"]}, 0.05, max_loss=None)["r1c2"]
    marks = underside_marks("r1c2", outlines["r1c2"])
    assert marks is not None and outlines["r1c2"].contains(marks)
    ev, ef = engrave_underside(v, f, marks, depth=0.6)
    m = trimesh.Trimesh(ev, ef)
    assert m.is_watertight
    removed = abs(mesh_volume(v, f)) - m.volume
    assert removed == pytest.approx(marks.area * 0.6, rel=1e-6)
    # A piece too small for legible text gets no marks.
    from shapely.geometry import box as sbox
    assert underside_marks("r1c2", sbox(0, 0, 6, 6)) is None


def test_plate_layout_fits_the_bed():
    import shapely

    pieces = {f"p{k}": trimesh.creation.box((40 + k, 30, 5)) for k in range(7)}
    pieces = {k: (m.vertices, m.faces) for k, m in pieces.items()}
    plates, oversize = plate_layout(pieces, 120, 100, gap=5)
    assert not oversize and sum(len(p) for p in plates) == 7 and len(plates) >= 2
    for plate in plates:
        rects = []
        for v, _ in plate.values():
            lo, hi = v.min(axis=0), v.max(axis=0)
            assert lo[0] >= 5 - 1e-9 and lo[1] >= 5 - 1e-9 and lo[2] == pytest.approx(0)
            assert hi[0] <= 115 + 1e-9 and hi[1] <= 95 + 1e-9
            rects.append(shapely.box(lo[0], lo[1], hi[0], hi[1]))
        assert shapely.union_all(rects).area == pytest.approx(sum(r.area for r in rects))
    _, over = plate_layout({"big": (np.array([[0, 0, 0], [500, 10, 1.0]]), np.zeros((0, 3)))}, 120, 100)
    assert over == ["big"]
