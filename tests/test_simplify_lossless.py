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
