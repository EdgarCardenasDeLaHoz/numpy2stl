"""Benchmark tests for numpy2stl performance."""
import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

from numpy2stl import array_to_mesh


class TestArrayToMeshPerformance:
    """Benchmark array_to_mesh function."""

    def test_array_to_mesh_solid(self, benchmark, elevation_array):
        """Benchmark solid mesh generation."""
        result = benchmark(array_to_mesh, elevation_array, solid=True)
        vertices, faces = result
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_array_to_mesh_surface_only(self, benchmark, elevation_array):
        """Benchmark surface-only mesh generation."""
        result = benchmark(array_to_mesh, elevation_array, solid=False)
        vertices, faces = result
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_array_to_mesh_with_mask(self, benchmark, masked_array):
        """Benchmark mesh generation with masking."""
        result = benchmark(array_to_mesh, masked_array, mask_val=0, solid=True)
        vertices, faces = result
        assert len(vertices) > 0
        assert len(faces) > 0
