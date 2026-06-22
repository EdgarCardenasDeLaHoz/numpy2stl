# Tests for solid.py - Solid class and mesh operations
import numpy as np
import pytest
from numpy2stl import Solid, vertices_to_index, get_open_edges, triangles_to_facets, validate_object


class TestSolid:
    """Test Solid class."""

    def test_solid_creation_from_tuple(self, cube_mesh):
        """Test Solid creation from (vertices, faces) tuple."""
        vertices, faces = cube_mesh
        solid = Solid((vertices, faces))

        assert solid.vertices.shape == vertices.shape
        assert solid.faces.shape == faces.shape

    def test_solid_creation_from_triangles(self):
        """Test Solid creation from raw triangles."""
        # Create simple triangle
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0.5, 1, 0]]], dtype=np.float64)

        solid = Solid(triangles)

        assert len(solid.vertices) == 3
        assert len(solid.faces) == 1

    def test_solid_save_stl(self, cube_mesh, tmp_stl_file):
        """Test STL file writing."""
        solid = Solid(cube_mesh)
        solid.save_stl(str(tmp_stl_file))

        assert tmp_stl_file.exists()
        assert tmp_stl_file.stat().st_size > 0

    def test_solid_save_stl_ascii(self, cube_mesh, tmp_stl_file):
        """Test ASCII STL file writing."""
        solid = Solid(cube_mesh)
        solid.save_stl(str(tmp_stl_file), ascii=True)

        assert tmp_stl_file.exists()
        # ASCII files should be larger
        content = tmp_stl_file.read_text()
        assert "solid ffd_geom" in content
        assert "endsolid" in content

    def test_solid_none_handling(self):
        """Test Solid handles None input."""
        solid = Solid(None)

        assert len(solid.vertices) == 0
        assert len(solid.faces) == 0


class TestVerticesToIndex:
    """Test vertices_to_index function."""

    def test_simple_deduplication(self):
        """Test vertex deduplication."""
        # Create triangles with duplicate vertices
        triangles = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],  # Triangle 1
                [[1, 0, 0], [1, 1, 0], [0, 1, 0]],  # Triangle 2 (shares 2 vertices)
            ],
            dtype=np.float64,
        )

        vertices, faces = vertices_to_index(triangles)

        # Should have 4 unique vertices (not 6)
        assert len(vertices) == 4
        assert len(faces) == 2
        assert faces.shape == (2, 3)

    def test_all_unique_vertices(self):
        """Test when all vertices are unique."""
        triangles = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[2, 0, 0], [3, 0, 0], [2, 1, 0]],
            ],
            dtype=np.float64,
        )

        vertices, faces = vertices_to_index(triangles)

        # All 6 vertices should remain
        assert len(vertices) == 6

    def test_precision_handling(self):
        """Test that similar vertices (within precision) are deduplicated."""
        # Create vertices that differ by small amounts
        triangles = np.array(
            [
                [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                [[1.0000001, 0, 0], [1, 1, 0], [0.0000001, 1, 0]],
            ],
            dtype=np.float64,
        )

        vertices, faces = vertices_to_index(triangles)

        # Should deduplicate similar vertices (rounds to 7 decimals)
        assert len(vertices) == 4


class TestGetOpenEdges:
    """Test get_open_edges function."""

    def test_closed_mesh_no_open_edges(self, cube_mesh):
        """Test that a closed mesh has no open edges."""
        _, faces = cube_mesh
        open_edges = get_open_edges(faces)

        assert len(open_edges) == 0

    def test_open_mesh_has_edges(self):
        """Test that an open mesh (single triangle) has open edges."""
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        open_edges = get_open_edges(faces)

        # Single triangle has 3 open edges
        assert len(open_edges) == 3

    def test_mesh_with_hole(self):
        """Test mesh with a hole has open edges."""
        # Create two separate triangles (not connected)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        open_edges = get_open_edges(faces)

        # Each triangle has 3 open edges
        assert len(open_edges) == 6


class TestTrianglesToFacets:
    """Test triangles_to_facets function."""

    def test_facet_format(self, simple_triangle_mesh):
        """Test that facets have correct format (normal + 3 vertices)."""
        vertices, faces = simple_triangle_mesh
        triangles = vertices[faces]

        facets = triangles_to_facets(triangles)

        # Each facet should have 12 values (normal + 3 vertices)
        assert facets.shape == (1, 12)

        # Normal should be normalized
        normal = facets[0, :3]
        normal_length = np.linalg.norm(normal)
        assert np.isclose(normal_length, 1.0)

    def test_normal_direction(self):
        """Test that normals are calculated correctly."""
        # Create triangle in XY plane
        triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)

        facets = triangles_to_facets(triangles)

        # Normal should point in +Z direction
        normal = facets[0, :3]
        assert np.isclose(normal[2], 1.0)  # Z component is 1
        assert np.isclose(normal[0], 0.0)  # X component is 0
        assert np.isclose(normal[1], 0.0)  # Y component is 0


class TestValidateObject:
    """Test validate_object function."""

    def test_valid_closed_mesh(self, cube_mesh):
        """Test validation of a valid closed mesh."""
        solid = Solid(cube_mesh)

        # Should not raise or print errors for valid mesh
        try:
            validate_object(solid)
        except Exception as e:
            pytest.fail(f"validate_object raised exception: {e}")

    def test_open_mesh_detection(self, simple_triangle_mesh):
        """Test detection of open edges."""
        solid = Solid(simple_triangle_mesh)

        # Single triangle is not watertight, should detect open edges
        # Function prints but doesn't raise - just verify it runs
        validate_object(solid)
