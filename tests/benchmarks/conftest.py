"""Benchmark fixtures for numpy2stl performance testing.

All names are prefixed ``bench_`` so they cannot shadow the regular fixtures in
tests/conftest.py (when collected from map2stl, a parametrized ``masked_array``
here leaked into test_generate.py and ran it at 100/500/1000).
"""
import numpy as np
import pytest


@pytest.fixture(params=[100, 500, 1000])
def bench_size(request):
    """Parametrized array size for benchmarks."""
    return request.param


@pytest.fixture
def bench_elevation_array(bench_size):
    """Generate random elevation array of given size."""
    np.random.seed(42)  # Reproducible
    return np.random.rand(bench_size, bench_size) * 100


@pytest.fixture
def bench_pyramid_array(bench_size):
    """Generate pyramid elevation array."""
    x, y = np.meshgrid(range(bench_size), range(bench_size))
    center = bench_size // 2
    return center - np.abs(x - center) - np.abs(y - center)


@pytest.fixture
def bench_masked_array(bench_size):
    """Generate array with masked region."""
    np.random.seed(42)
    arr = np.random.rand(bench_size, bench_size) * 100
    # Mask center region
    mask_size = bench_size // 4
    center = bench_size // 2
    arr[
        center - mask_size : center + mask_size, center - mask_size : center + mask_size
    ] = -999
    return arr
