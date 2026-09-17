# Tests for the stl2numpy module
import numpy as np
import pytest

from numpy2stl import array_to_mesh, triangles_to_facets, writeSTL


# ---------------------------------------------------------------------------
# Fixtures — build synthetic STL files from numpy2stl itself
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pyramid_stl(tmp_path_factory):
    """Solid pyramid mesh: values 20–30, so floor sits below the surface."""
    tmp = tmp_path_factory.mktemp("stl2numpy")
    x, y = np.meshgrid(range(20), range(20))
    elevation = 20.0 + (10 - np.sqrt((x - 10) ** 2 + (y - 10) ** 2) * 0.5).clip(0)
    vertices, faces = array_to_mesh(elevation, solid=True, floor_val=0)
    triangles = vertices[faces]
    facets = triangles_to_facets(triangles)
    path = tmp / "pyramid.stl"
    writeSTL(facets, str(path))
    return str(path)


@pytest.fixture(scope="module")
def flat_stl(tmp_path_factory):
    """Flat slab: uniform elevation = 5, easy to verify round-trip."""
    tmp = tmp_path_factory.mktemp("stl2numpy_flat")
    elevation = np.ones((10, 10), dtype=np.float64) * 5.0
    vertices, faces = array_to_mesh(elevation, solid=True, floor_val=0)
    triangles = vertices[faces]
    facets = triangles_to_facets(triangles)
    path = tmp / "flat.stl"
    writeSTL(facets, str(path))
    return str(path)


# ---------------------------------------------------------------------------
# mesh_to_heightmap
# ---------------------------------------------------------------------------

class TestMeshToHeightmap:

    def test_returns_dict_with_required_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=32)
        assert "heightmap" in result
        assert "bounds" in result
        assert "resolution" in result
        assert "cell_size" in result
        assert "projection" in result

    def test_heightmap_shape_matches_resolution(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=50)
        h = result["heightmap"]
        assert h.shape[0] == 50
        assert h.shape[1] == 50

    def test_heightmap_rectangular_resolution(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=(40, 30))
        assert result["heightmap"].shape == (40, 30)

    def test_heightmap_dtype_float64(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=32)
        assert result["heightmap"].dtype == np.float64

    def test_bounds_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=32)
        for axis in ("x", "y", "z"):
            assert axis in result["bounds"]
            lo, hi = result["bounds"][axis]
            assert lo <= hi

    def test_max_projection_gives_highest_surface(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=32, projection="max")
        h = result["heightmap"]
        valid = h[~np.isnan(h)]
        assert len(valid) > 0
        # Peak of pyramid should be near the max elevation we set (≈30)
        assert valid.max() > 20

    def test_min_projection_lower_than_max(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        r_max = mesh_to_heightmap(pyramid_stl, resolution=32, projection="max")
        r_min = mesh_to_heightmap(pyramid_stl, resolution=32, projection="min")
        valid_max = r_max["heightmap"][~np.isnan(r_max["heightmap"])]
        valid_min = r_min["heightmap"][~np.isnan(r_min["heightmap"])]
        # Global max of 'max projection' must be >= global min of 'min projection'
        assert valid_max.max() >= valid_min.min()
        # Mean of top-surface projection must exceed mean of bottom projection
        assert valid_max.mean() > valid_min.mean()

    def test_auto_resolution_capped(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl)  # auto resolution
        h = result["heightmap"]
        assert max(h.shape) <= 1000

    def test_invalid_projection_raises(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        with pytest.raises(ValueError, match="projection"):
            mesh_to_heightmap(pyramid_stl, projection="bad")

    def test_allow_large_bypasses_cap(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=1200, allow_large=True)
        assert max(result["heightmap"].shape) == 1200

    def test_resolution_capped_without_allow_large(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_heightmap
        result = mesh_to_heightmap(pyramid_stl, resolution=1200, allow_large=False)
        assert max(result["heightmap"].shape) <= 1000


# ---------------------------------------------------------------------------
# get_mesh_properties
# ---------------------------------------------------------------------------

class TestGetMeshProperties:

    def test_returns_dict_with_required_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        for key in ("volume", "surface_area", "bounds", "extents",
                    "center_of_mass", "num_vertices", "num_faces",
                    "is_watertight", "is_winding_consistent", "euler_number"):
            assert key in props

    def test_surface_area_positive(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        assert props["surface_area"] > 0

    def test_face_and_vertex_counts_positive(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        assert props["num_vertices"] > 0
        assert props["num_faces"] > 0

    def test_bounds_consistent(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        for axis in ("x", "y", "z"):
            lo, hi = props["bounds"][axis]
            assert lo <= hi

    def test_extents_match_bounds(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        bounds = props["bounds"]
        extents = props["extents"]
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = bounds[axis]
            assert np.isclose(extents[i], hi - lo, atol=1e-6)

    def test_watertight_solid_has_volume(self, pyramid_stl):
        from numpy2stl.stl2numpy import get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        if props["is_watertight"]:
            assert props["volume"] is not None
            assert props["volume"] > 0


# ---------------------------------------------------------------------------
# detect_orientation
# ---------------------------------------------------------------------------

class TestDetectOrientation:

    def test_returns_dict_with_required_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import detect_orientation
        result = detect_orientation(pyramid_stl)
        for key in ("up_axis", "up_axis_name", "confidence", "method"):
            assert key in result

    def test_up_axis_valid_range(self, pyramid_stl):
        from numpy2stl.stl2numpy import detect_orientation
        result = detect_orientation(pyramid_stl)
        assert result["up_axis"] in (0, 1, 2)

    def test_up_axis_name_matches_index(self, pyramid_stl):
        from numpy2stl.stl2numpy import detect_orientation
        result = detect_orientation(pyramid_stl)
        assert result["up_axis_name"] == ["x", "y", "z"][result["up_axis"]]

    def test_confidence_between_0_and_1(self, pyramid_stl):
        from numpy2stl.stl2numpy import detect_orientation
        result = detect_orientation(pyramid_stl)
        assert 0.0 <= result["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# mesh_to_voxels
# ---------------------------------------------------------------------------

class TestMeshToVoxels:

    def test_returns_dict_with_required_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_voxels
        result = mesh_to_voxels(pyramid_stl, resolution=16)
        for key in ("voxels", "pitch", "origin", "bounds", "resolution"):
            assert key in result

    def test_voxels_is_bool_3d(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_voxels
        result = mesh_to_voxels(pyramid_stl, resolution=16)
        assert result["voxels"].ndim == 3
        assert result["voxels"].dtype == bool

    def test_voxels_shape_matches_resolution_tuple(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_voxels
        result = mesh_to_voxels(pyramid_stl, resolution=16)
        assert result["resolution"] == result["voxels"].shape

    def test_some_voxels_filled(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_voxels
        result = mesh_to_voxels(pyramid_stl, resolution=16)
        assert result["voxels"].any()

    def test_explicit_pitch(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_voxels
        result = mesh_to_voxels(pyramid_stl, pitch=2.0)
        assert np.isclose(result["pitch"], 2.0)


# ---------------------------------------------------------------------------
# mesh_to_pointcloud
# ---------------------------------------------------------------------------

class TestMeshToPointcloud:

    def test_shape_surface_method(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_pointcloud
        pts = mesh_to_pointcloud(pyramid_stl, n_points=500)
        assert pts.shape == (500, 3)

    def test_shape_with_normals(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_pointcloud
        pts = mesh_to_pointcloud(pyramid_stl, n_points=500, include_normals=True)
        assert pts.shape == (500, 6)

    def test_vertices_method_returns_all_vertices(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_pointcloud, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        pts = mesh_to_pointcloud(pyramid_stl, method="vertices")
        assert pts.shape[0] == props["num_vertices"]
        assert pts.shape[1] == 3

    def test_dtype_float64(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_pointcloud
        pts = mesh_to_pointcloud(pyramid_stl, n_points=100)
        assert pts.dtype == np.float64

    def test_points_within_mesh_bounds(self, pyramid_stl):
        from numpy2stl.stl2numpy import mesh_to_pointcloud, get_mesh_properties
        pts = mesh_to_pointcloud(pyramid_stl, n_points=1000)
        props = get_mesh_properties(pyramid_stl)
        for i, axis in enumerate(("x", "y", "z")):
            lo, hi = props["bounds"][axis]
            assert pts[:, i].min() >= lo - 1e-6
            assert pts[:, i].max() <= hi + 1e-6


# ---------------------------------------------------------------------------
# slice_mesh
# ---------------------------------------------------------------------------

class TestSliceMesh:

    def test_returns_list(self, pyramid_stl):
        from numpy2stl.stl2numpy import slice_mesh
        slices = slice_mesh(pyramid_stl, n_slices=5)
        assert isinstance(slices, list)
        assert len(slices) == 5

    def test_each_slice_has_required_keys(self, pyramid_stl):
        from numpy2stl.stl2numpy import slice_mesh
        slices = slice_mesh(pyramid_stl, n_slices=3)
        for s in slices:
            assert "z" in s
            assert "polygons" in s
            assert "section" in s

    def test_explicit_z_levels(self, pyramid_stl):
        from numpy2stl.stl2numpy import slice_mesh, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        z_lo, z_hi = props["bounds"]["z"]
        z_mid = (z_lo + z_hi) / 2
        slices = slice_mesh(pyramid_stl, z_levels=[z_mid])
        assert len(slices) == 1
        assert np.isclose(slices[0]["z"], z_mid)

    def test_polygons_is_list_of_arrays(self, pyramid_stl):
        from numpy2stl.stl2numpy import slice_mesh, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        z_lo, z_hi = props["bounds"]["z"]
        z_mid = (z_lo + z_hi) / 2
        slices = slice_mesh(pyramid_stl, z_levels=[z_mid])
        polys = slices[0]["polygons"]
        assert isinstance(polys, list)
        if polys:
            assert isinstance(polys[0], np.ndarray)
            assert polys[0].ndim == 2
            assert polys[0].shape[1] == 2


# ---------------------------------------------------------------------------
# rasterize_slice
# ---------------------------------------------------------------------------

class TestRasterizeSlice:

    def test_returns_bool_2d(self, pyramid_stl):
        from numpy2stl.stl2numpy import rasterize_slice, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        z_lo, z_hi = props["bounds"]["z"]
        z_mid = (z_lo + z_hi) / 2
        mask = rasterize_slice(pyramid_stl, z=z_mid, resolution=64)
        assert mask.dtype == bool
        assert mask.ndim == 2

    def test_mask_shape_matches_resolution(self, pyramid_stl):
        from numpy2stl.stl2numpy import rasterize_slice, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        z_lo, z_hi = props["bounds"]["z"]
        z_mid = (z_lo + z_hi) / 2
        mask = rasterize_slice(pyramid_stl, z=z_mid, resolution=128)
        assert mask.shape == (128, 128)


# ---------------------------------------------------------------------------
# decimate_mesh
# ---------------------------------------------------------------------------

class TestDecimateMesh:

    def test_returns_vertices_and_faces(self, pyramid_stl):
        from numpy2stl.stl2numpy import decimate_mesh
        vertices, faces = decimate_mesh(pyramid_stl, target_ratio=0.5)
        assert vertices.ndim == 2
        assert vertices.shape[1] == 3
        assert faces.ndim == 2
        assert faces.shape[1] == 3

    def test_face_count_reduced(self, pyramid_stl):
        from numpy2stl.stl2numpy import decimate_mesh, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        original = props["num_faces"]
        _, faces = decimate_mesh(pyramid_stl, target_ratio=0.5)
        assert len(faces) < original

    def test_explicit_target_faces(self, pyramid_stl):
        from numpy2stl.stl2numpy import decimate_mesh, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        original = props["num_faces"]
        target = max(4, original // 4)
        _, faces = decimate_mesh(pyramid_stl, target_faces=target)
        # trimesh may not hit exact count but should be substantially fewer
        assert len(faces) < original

    def test_unchanged_when_target_exceeds_original(self, pyramid_stl):
        from numpy2stl.stl2numpy import decimate_mesh, get_mesh_properties
        props = get_mesh_properties(pyramid_stl)
        original = props["num_faces"]
        _, faces = decimate_mesh(pyramid_stl, target_faces=original * 10)
        assert len(faces) == original
