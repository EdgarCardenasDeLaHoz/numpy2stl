"""Lossless flat-region simplification: same solid, far fewer faces."""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from numpy2stl import array_to_mesh  # noqa: E402
from numpy2stl.processing.simplify import simplify_mesh_surfaces  # noqa: E402


def _heightmap(n=60):
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:n, 0:n]
    A = np.where((x - n / 2) ** 2 + (y - n / 3) ** 2 < (n / 6) ** 2, 12.0, 5 + 0.1 * x)
    A[:, : n // 4] = 1.0  # flat sea ...
    A[n // 2 : n // 2 + 6, 4:8] = 3.0  # ... with an island
    A[40:48, 36:48] = rng.integers(6, 9, (8, 12))  # rough patch
    return array_to_mesh(A, floor_val=0)


def _courtyard_prism():
    from shapely.geometry import Polygon

    shell = [(0, 0), (30, 0), (30, 20), (0, 20)]
    hole = [(10, 5), (20, 5), (20, 15), (10, 15)]
    mesh = trimesh.creation.extrude_polygon(Polygon(shell, [hole]), height=9.5)
    # Subdivide so the flat roof, floor and walls carry redundant vertices.
    mesh = mesh.subdivide().subdivide()
    return mesh.vertices, mesh.faces


def _terrain_with_buildings():
    mfd = pytest.importorskip("manifold3d")

    def to_manifold(v, f):
        return mfd.Manifold(
            mfd.Mesh64(
                vert_properties=np.ascontiguousarray(v, dtype=np.float64),
                tri_verts=np.ascontiguousarray(f, dtype=np.uint64),
            )
        )

    y, x = np.mgrid[0:40, 0:50]
    A = 4 + 0.05 * x + 0.5 * np.sin(y / 5.0)
    A[10:30, 5:20] = 5.0  # flat terrace for the buildings
    solid = to_manifold(*array_to_mesh(A, floor_val=0))
    for x0, y0, w, h, z in [(6, 12, 5, 6, 15.0), (12, 20, 6, 7, 22.5), (30, 8, 8, 8, 18.0)]:
        box = mfd.Manifold.cube((w, h, z)).translate((x0, y0, 1.0))
        solid = solid + box
    out = solid.to_mesh64()
    return np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts, dtype=np.int64)


def _check_lossless(vertices, faces):
    new_faces = simplify_mesh_surfaces(vertices, faces)
    before = trimesh.Trimesh(vertices, faces, process=False)
    after = trimesh.Trimesh(vertices, new_faces, process=False)
    after.remove_unreferenced_vertices()

    assert after.is_watertight
    assert after.is_winding_consistent
    np.testing.assert_allclose(after.volume, before.volume, rtol=1e-6)
    np.testing.assert_allclose(after.area, before.area, rtol=1e-6)
    assert after.volume > 0  # still facing outward
    assert len(new_faces) < len(faces)

    # No vertex off the original surface: output positions are input positions.
    as_rows = {tuple(v) for v in np.asarray(vertices, dtype=np.float64)}
    assert all(tuple(v) in as_rows for v in after.vertices)
    return before, after, new_faces


def test_heightmap_plateaus_sea_and_slopes():
    vertices, faces = _heightmap()
    before, after, _ = _check_lossless(vertices, faces)

    # Orientation kept: the top still faces up, and every face normal agrees
    # with the original surface it lies on.
    assert (after.face_normals[:, 2] > 0.5).sum() > 0
    up_area_before = before.area_faces[before.face_normals[:, 2] > 0.999].sum()
    up_area_after = after.area_faces[after.face_normals[:, 2] > 0.999].sum()
    np.testing.assert_allclose(up_area_after, up_area_before, rtol=1e-9)
    assert len(after.faces) < len(before.faces) / 3


def test_flat_sea_with_island_keeps_hole():
    A = np.ones((30, 30))
    A[10:15, 12:18] = 4.0  # island: a hole in the flat sea region
    vertices, faces = array_to_mesh(A, floor_val=0)
    before, after, _ = _check_lossless(vertices, faces)

    def sea_area(mesh):
        sea = (mesh.face_normals[:, 2] > 0.999) & np.isclose(mesh.triangles_center[:, 2], 1.0)
        return mesh.area_faces[sea].sum()

    np.testing.assert_allclose(sea_area(after), sea_area(before), rtol=1e-9)
    assert sea_area(after) < 29 * 29 - 5 * 6  # the island is still cut out


def test_extruded_prism_with_courtyard():
    vertices, faces = _courtyard_prism()
    _, after, _ = _check_lossless(vertices, faces)
    # Each flat face of the prism becomes a minimal triangulation.
    assert len(after.faces) <= 48


def test_terrain_slab_union_prisms():
    vertices, faces = _terrain_with_buildings()
    _check_lossless(vertices, faces)


def test_non_planar_mesh_unchanged():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    new_faces = simplify_mesh_surfaces(mesh.vertices, mesh.faces)
    assert len(new_faces) == len(mesh.faces)
    after = trimesh.Trimesh(mesh.vertices, new_faces, process=False)
    assert after.is_watertight and after.is_winding_consistent


def test_empty_input():
    faces = simplify_mesh_surfaces(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert faces.shape == (0, 3)


def _manifold_mesh(solid):
    out = solid.to_mesh64()
    return np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts, dtype=np.int64)


def _flat_faces(mesh, z):
    """Faces of ``mesh`` on the upward-facing plane at height ``z``."""
    return (mesh.face_normals[:, 2] > 1 - 1e-9) & np.isclose(mesh.triangles_center[:, 2], z)


def test_sliver_strip_with_near_collinear_points():
    # A 200 mm x 1 mm strip whose long sides wobble by 1e-6: every boundary
    # point is nearly collinear with its neighbours (Triangle's
    # "segmentintersection" territory). The top must still become a minimal
    # triangulation of its boundary.
    mfd = pytest.importorskip("manifold3d")
    rng = np.random.default_rng(1)
    x = np.linspace(0, 200, 301)
    lo = 1e-6 * rng.standard_normal(len(x))
    hi = 1e-3 + 1e-6 * rng.standard_normal(len(x))
    outline = np.r_[np.c_[x, lo], np.c_[x, hi][::-1]]
    v, f = _manifold_mesh(mfd.CrossSection([outline]).extrude(2.0))
    mesh = trimesh.Trimesh(v, f, process=False).subdivide()
    _, after, _ = _check_lossless(mesh.vertices, mesh.faces)
    top = _flat_faces(after, 2.0)
    assert top.sum() == len(np.unique(after.faces[top])) - 2


def test_hole_touching_the_outer_boundary():
    # A block on a plate whose corner touches the plate's edge: the plate's
    # top is one region whose boundary passes twice through that vertex.
    mfd = pytest.importorskip("manifold3d")
    plate = mfd.Manifold.cube((10, 10, 1))
    block = mfd.CrossSection([[(0.0, 5.0), (4.0, 3.0), (4.0, 7.0)]]).extrude(1).translate((0, 0, 1))
    v, f = _manifold_mesh(plate + block)
    mesh = trimesh.Trimesh(v, f, process=False).subdivide().subdivide()
    _, after, _ = _check_lossless(mesh.vertices, mesh.faces)
    top = _flat_faces(after, 1.0)
    np.testing.assert_allclose(after.area_faces[top].sum(), 100 - 8, rtol=1e-12)
    assert top.sum() == 6  # outline: 4 corners + 3 block corners, one visited twice


def test_region_with_hole_is_minimal():
    vertices, faces = _courtyard_prism()
    _, after, _ = _check_lossless(vertices, faces)
    roof = _flat_faces(after, 9.5)
    # 8 vertices and one hole: 8 + 2 - 2 triangles.
    assert roof.sum() == 8
    np.testing.assert_allclose(after.area_faces[roof].sum(), 30 * 20 - 10 * 10, rtol=1e-12)


def test_collinear_boundary_vertices_are_kept():
    # With the walls left alone (too few faces to be candidates) every vertex
    # on the top's straight edges is shared with them and must stay: the top
    # is triangulated over all of its collinear boundary points and nothing else.
    vertices, faces = array_to_mesh(np.ones((12, 12)), floor_val=0)
    before = trimesh.Trimesh(vertices, faces, process=False)
    wall_faces = int((np.abs(before.face_normals[:, 0]) > 0.9).sum() / 2)
    new_faces = simplify_mesh_surfaces(vertices, faces, min_faces=wall_faces + 1)
    after = trimesh.Trimesh(vertices, new_faces, process=False)
    after.remove_unreferenced_vertices()
    assert after.is_watertight and after.is_winding_consistent
    np.testing.assert_allclose(after.volume, before.volume, rtol=1e-9)

    top_before = faces[before.face_normals[:, 2] > 0.999]
    edges = np.sort(top_before[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, uses = np.unique(edges, axis=0, return_counts=True)
    rim = np.unique(edges[uses == 1])  # 4 * 11 points on the top's straight edges
    top_after = new_faces[
        trimesh.Trimesh(vertices, new_faces, process=False).face_normals[:, 2] > 0.999
    ]
    # No boundary point dropped, no interior point kept, no fan left over.
    np.testing.assert_array_equal(np.unique(top_after), rim)
    assert len(top_after) == len(rim) - 2 < len(top_before)


def _city_like(seed=7, n=40):
    mfd = pytest.importorskip("manifold3d")
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:60, 0:80]
    A = 3 + 0.02 * x + 0.3 * np.sin(y / 7.0)
    A[5:55, 5:75] = 4.0  # flat district
    v, f = array_to_mesh(A, floor_val=0)
    solid = mfd.Manifold(
        mfd.Mesh64(
            vert_properties=np.ascontiguousarray(v, dtype=np.float64),
            tri_verts=np.ascontiguousarray(f, dtype=np.uint64),
        )
    )
    for _ in range(n):
        w, d = rng.uniform(2, 9, 2)
        h = rng.choice([6.0, 9.5, 12.0, rng.uniform(5, 20)])  # repeated heights: coplanar roofs
        box = mfd.Manifold.cube((w, d, h)).translate((-w / 2, -d / 2, 0.0))
        if rng.random() < 0.4:
            box = box.rotate((0.0, 0.0, float(rng.uniform(0, 90))))
        cx, cy = rng.uniform(8, 72), rng.uniform(8, 52)
        # Snap some to a 2 mm grid so neighbours share walls and edges.
        if rng.random() < 0.5:
            cx, cy = np.round(cx / 2) * 2, np.round(cy / 2) * 2
        solid = solid + box.translate((cx, cy, 3.5))
    return _manifold_mesh(solid)


@pytest.mark.parametrize("seed, n", [(7, 40), (33, 80)])  # 33: solids touching at a point
def test_city_like_union_of_prisms(seed, n):
    vertices, faces = _city_like(seed, n)
    before, after, new_faces = _check_lossless(vertices, faces)
    assert len(new_faces) < 0.5 * len(faces)


def test_simplify_does_not_use_triangle():
    import ast
    from pathlib import Path

    import numpy2stl.processing.simplify as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "triangle" not in imported
