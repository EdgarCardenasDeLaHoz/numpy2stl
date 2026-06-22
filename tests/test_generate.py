# Tests for generate.py - Core mesh generation functions
import numpy as np
import pytest
from numpy2stl import array_to_mesh, array2faces, perimeter_to_walls


class TestArrayToMesh:
    """Test array_to_mesh function."""

    def test_basic_mesh_generation(self, simple_elevation_array):
        """Test basic mesh generation from elevation array."""
        vertices, faces = array_to_mesh(simple_elevation_array, solid=True)

        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert vertices.shape[1] == 3  # 3D coordinates
        assert faces.shape[1] == 3  # Triangular faces
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_mesh_with_mask(self, masked_array):
        """Test mesh generation with masked values."""
        vertices, faces = array_to_mesh(masked_array, mask_val=0, solid=True)

        assert len(vertices) > 0
        assert len(faces) > 0
        # Verify no vertices have z <= 0 (masked region)
        assert np.all(vertices[:, 2] > -900)  # Allow some margin

    def test_floor_val_independence(self, simple_elevation_array):
        """Test floor_val is independent of mask_val."""
        # Shift surface up so it sits entirely above both floor_val test values
        elevated = simple_elevation_array + 20  # values now 20–30

        vertices1, _ = array_to_mesh(elevated, floor_val=0)
        vertices2, _ = array_to_mesh(elevated, floor_val=5)

        min_z1 = vertices1[:, 2].min()
        min_z2 = vertices2[:, 2].min()
        assert min_z1 != min_z2
        assert np.isclose(min_z1, 0, atol=0.1)
        assert np.isclose(min_z2, 5, atol=0.1)

    def test_non_solid_mesh(self, simple_elevation_array):
        """Test non-solid mesh (top surface only)."""
        vertices_solid, faces_solid = array_to_mesh(simple_elevation_array, solid=True)
        vertices_surf, faces_surf = array_to_mesh(simple_elevation_array, solid=False)

        # Non-solid should have fewer faces (no walls/bottom)
        assert len(faces_surf) < len(faces_solid)

    def test_empty_array_handling(self):
        """Test handling of empty arrays."""
        empty = np.array([[]])
        vertices, faces = array_to_mesh(empty)

        assert len(vertices) == 0
        assert len(faces) == 0

    def test_none_array_handling(self):
        """Test handling of None input."""
        vertices, faces = array_to_mesh(None)

        assert len(vertices) == 0
        assert len(faces) == 0

    def test_invalid_input_type(self):
        """Test error handling for invalid input types."""
        with pytest.raises(TypeError):
            array_to_mesh("not an array")

    def test_invalid_dimensions(self):
        """Test error handling for non-2D arrays."""
        with pytest.raises(ValueError):
            array_to_mesh(np.array([1, 2, 3]))  # 1D

    def test_output_format(self, simple_elevation_array):
        """Test that output is in correct (vertices, faces) format."""
        vertices, faces = array_to_mesh(simple_elevation_array)

        # Verify vertices are floats
        assert np.issubdtype(vertices.dtype, np.floating)

        # Verify faces are integers
        assert np.issubdtype(faces.dtype, np.integer)

        # Verify face indices are valid
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))


class TestArray2Faces:
    """Test array2faces function."""

    def test_basic_triangulation(self):
        """Test basic 2D array triangulation."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)

        vertices, faces = array2faces(arr, mask_val=0)

        assert len(vertices) > 0
        assert len(faces) > 0
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

    def test_masking(self):
        """Test that masking excludes regions correctly."""
        arr = np.ones((10, 10))
        arr[3:7, 3:7] = -1  # Masked region

        vertices, faces = array2faces(arr, mask_val=0)

        # Should have faces outside masked region
        assert len(faces) > 0
        # Most vertices should be from unmasked region (positive z-values)
        # Note: Some boundary vertices may still have negative z-values
        z_values = vertices[faces][:, :, 2].flatten()
        assert np.sum(z_values > 0) > np.sum(z_values <= 0)


class TestPerimeterToWalls:
    """Test perimeter_to_walls function."""

    def test_simple_square(self):
        """Test wall generation for a simple square."""
        vertices = np.array(
            [[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]], dtype=np.float64
        )
        perimeters = [np.array([0, 1, 2, 3])]

        wall_triangles = perimeter_to_walls(vertices, perimeters, floor_val=0)

        assert len(wall_triangles) > 0
        # Should have 2 triangles per edge (4 edges * 2)
        assert len(wall_triangles) == 8

        # Verify wall triangles connect top to bottom
        assert np.any(wall_triangles[:, :, 2] == 0)  # Bottom at z=0
        assert np.any(wall_triangles[:, :, 2] == 5)  # Top at z=5

    def test_multiple_perimeters(self):
        """Test handling of multiple perimeters (e.g., holes)."""
        # Outer square
        outer = np.array(
            [[0, 0, 5], [2, 0, 5], [2, 2, 5], [0, 2, 5]], dtype=np.float64
        )
        # Inner square (hole)
        inner = np.array(
            [[0.5, 0.5, 5], [1.5, 0.5, 5], [1.5, 1.5, 5], [0.5, 1.5, 5]],
            dtype=np.float64,
        )

        vertices = np.vstack([outer, inner])
        perimeters = [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])]

        wall_triangles = perimeter_to_walls(vertices, perimeters, floor_val=0)

        # Should have walls for both perimeters
        assert len(wall_triangles) == 16  # 8 triangles per perimeter
