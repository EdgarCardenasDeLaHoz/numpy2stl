# Tests for polygon.py - Polygon utilities
import numpy as np
from numpy2stl import get_ordered_perimeter, triangulate_polygon, rotate_3D


class TestGetOrderedPerimeter:
    """Test get_ordered_perimeter function."""

    def test_simple_square(self):
        """Test ordering of square perimeter."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64
        )

        # Open edges form a square
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        perimeters = get_ordered_perimeter(vertices, edges)

        assert len(perimeters) == 1  # Single closed loop
        assert len(perimeters[0]) == 4  # 4 vertices

        # Should form a closed loop
        peri = perimeters[0]
        assert len(set(peri)) == 4  # All vertices unique

    def test_multiple_perimeters(self):
        """Test multiple disconnected perimeters."""
        vertices = np.array(
            [
                # Square 1
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                # Square 2
                [2, 0, 0],
                [3, 0, 0],
                [3, 1, 0],
                [2, 1, 0],
            ],
            dtype=np.float64,
        )

        edges = np.array(
            [
                # Square 1
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                # Square 2
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ]
        )

        perimeters = get_ordered_perimeter(vertices, edges)

        assert len(perimeters) == 2  # Two separate loops
        assert len(perimeters[0]) == 4
        assert len(perimeters[1]) == 4


class TestRotate3D:
    """Test rotate_3D function."""

    def test_identity_rotation(self):
        """Test rotation from vector to itself (identity)."""
        pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        # Rotate Z to Z (should be identity)
        rotated = rotate_3D(pts, [0, 0, 1], [0, 0, 1])

        np.testing.assert_array_almost_equal(rotated, pts)

    def test_90_degree_rotation(self):
        """Test 90-degree rotation."""
        pts = np.array([[1, 0, 0]], dtype=np.float64)

        # Rotate X axis to Y axis
        rotated = rotate_3D(pts, [1, 0, 0], [0, 1, 0])

        # Should end up pointing in Y direction
        expected = np.array([[0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)

    def test_180_degree_rotation(self):
        """Test 180-degree rotation."""
        pts = np.array([[1, 0, 0]], dtype=np.float64)

        # Rotate to opposite direction
        rotated = rotate_3D(pts, [1, 0, 0], [-1, 0, 0])

        expected = np.array([[-1, 0, 0]], dtype=np.float64)
        np.testing.assert_array_almost_equal(rotated, expected, decimal=5)


class TestTriangulatePolygon:
    """Test triangulate_polygon function."""

    def test_simple_quad(self):
        """Test triangulation of a simple quad."""
        vertices_2d = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
        )
        perimeters = [np.array([0, 1, 2, 3])]

        vertices, faces = triangulate_polygon(vertices_2d, perimeters)

        # Quad should be split into 2 triangles
        assert len(faces) == 2
        assert faces.shape[1] == 3

        # All face indices should be valid
        assert np.all(faces >= 0)
        assert np.all(faces < len(vertices))

    def test_polygon_with_hole(self):
        """Test triangulation with a hole."""
        # Outer square
        outer_verts = np.array(
            [[0, 0], [4, 0], [4, 4], [0, 4]], dtype=np.float64
        )

        # Inner square (hole)
        inner_verts = np.array(
            [[1, 1], [3, 1], [3, 3], [1, 3]], dtype=np.float64
        )

        vertices_2d = np.vstack([outer_verts, inner_verts])
        perimeters = [
            np.array([0, 1, 2, 3]),  # Outer
            np.array([4, 5, 6, 7]),  # Inner (hole)
        ]

        vertices, faces = triangulate_polygon(vertices_2d, perimeters)

        # Should have faces between outer and inner squares
        assert len(faces) > 0

        # Verify no faces penetrate the hole (this is hard to test directly,
        # but at least verify we got a reasonable number of triangles)
        # Area of (4x4 - 2x2) = 12, so roughly 12-16 triangles expected
        assert 8 <= len(faces) <= 24
