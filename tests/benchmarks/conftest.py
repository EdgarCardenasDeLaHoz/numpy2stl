"""Benchmark fixtures for numpy2stl performance testing."""
import numpy as np
import pytest


@pytest.fixture(params=[100, 500, 1000])
def array_size(request):
    """Parametrized array size for benchmarks."""
    return request.param


@pytest.fixture
def elevation_array(array_size):
    """Generate random elevation array of given size."""
    np.random.seed(42)  # Reproducible
    return np.random.rand(array_size, array_size) * 100


@pytest.fixture
def pyramid_array(array_size):
    """Generate pyramid elevation array."""
    x, y = np.meshgrid(range(array_size), range(array_size))
    center = array_size // 2
    return center - np.abs(x - center) - np.abs(y - center)


@pytest.fixture
def masked_array(array_size):
    """Generate array with masked region."""
    np.random.seed(42)
    arr = np.random.rand(array_size, array_size) * 100
    # Mask center region
    mask_size = array_size // 4
    center = array_size // 2
    arr[
        center - mask_size : center + mask_size, center - mask_size : center + mask_size
    ] = -999
    return arr
