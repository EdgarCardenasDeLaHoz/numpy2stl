# Test configuration for numpy2stl
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _isolated_stl_cache(request, tmp_path_factory):
    """Point mesh_to_heightmap's on-disk cache at a temp dir.

    The default cache (src/numpy2stl/registration/runs/stl_cache) is keyed on the
    mesh's absolute path, so entries written for pytest tmp files are never reused
    and used to pile up (~1.6 GB). One dir per session keeps cache hits testable.

    When the suite is collected from ../map2stl (rootdir outside this repo), pytest
    registers this conftest's fixtures for *every* test, map2stl's included. That is
    fine here (map2stl tests reach mesh_to_heightmap too), but it is why this patches
    by hand instead of requesting ``monkeypatch``: an extra autouse ``monkeypatch``
    set up before map2stl's ``_isolate_export_tasks`` would undo a test's patches
    only after that fixture's teardown had run with them (a patched time.monotonic).
    Integration tests keep the real cache, as in map2stl's ``_isolate_disk_cache``.
    """
    if request.node.get_closest_marker("integration"):
        yield
        return
    try:
        from numpy2stl.stl2numpy import heightmap
    except ImportError:  # optional deps missing: nothing to redirect
        yield
        return
    saved = heightmap._STL_CACHE_DIR
    heightmap._STL_CACHE_DIR = tmp_path_factory.getbasetemp() / "stl_cache"
    yield
    heightmap._STL_CACHE_DIR = saved


@pytest.fixture
def simple_elevation_array():
    """10x10 pyramid for testing."""
    x, y = np.meshgrid(range(10), range(10))
    pyramid = (5 - np.abs(x - 5)) + (5 - np.abs(y - 5))
    return pyramid.astype(np.float64)


@pytest.fixture
def masked_array():
    """Array with masked region."""
    arr = np.random.RandomState(42).rand(20, 20) * 10
    arr[5:10, 5:10] = -999  # Mask value
    return arr


@pytest.fixture
def large_array():
    """500x500 array for performance testing."""
    return np.random.RandomState(42).rand(500, 500) * 100


@pytest.fixture
def tmp_stl_file(tmp_path):
    """Temporary file for STL output."""
    return tmp_path / "test_output.stl"


@pytest.fixture
def tmp_3mf_file(tmp_path):
    """Temporary file for 3MF output."""
    return tmp_path / "test_output.3mf"


@pytest.fixture
def tmp_obj_file(tmp_path):
    """Temporary file for OBJ output."""
    return tmp_path / "test_output.obj"


@pytest.fixture
def simple_triangle_mesh():
    """Simple triangle mesh for testing."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def cube_mesh():
    """Simple cube mesh for testing."""
    vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            # Bottom
            [0, 1, 2],
            [0, 2, 3],
            # Top
            [4, 6, 5],
            [4, 7, 6],
            # Front
            [0, 5, 1],
            [0, 4, 5],
            # Back
            [2, 7, 3],
            [2, 6, 7],
            # Left
            [0, 7, 4],
            [0, 3, 7],
            # Right
            [1, 6, 2],
            [1, 5, 6],
        ],
        dtype=np.int32,
    )
    return vertices, faces


# --- Registration fixtures (moved from test_registration.py, B6 split) ---

@pytest.fixture
def simple_building_array():
    """32×32 array with a few rectangular 'buildings' — controlled test data."""
    arr = np.zeros((32, 32), dtype=np.float64)
    arr[6:14, 6:14] = 20.0    # building A
    arr[18:26, 18:26] = 35.0  # building B
    arr[10:16, 20:28] = 15.0  # building C
    return arr


@pytest.fixture
def osm_mock():
    """Synthetic OSM-like heightmap dict (the reference-source building_heightmap format)."""
    arr = np.full((32, 32), np.nan, dtype=np.float64)
    arr[6:14, 6:14] = 18.0
    arr[18:26, 18:26] = 32.0
    arr[10:16, 20:28] = 14.0
    return {
        "heightmap": arr,
        "bounds": {"x": (-75.28, -74.95), "y": (39.86, 40.06), "z": (0.0, 35.0)},
        "resolution": (32, 32),
        "cell_size": (0.01, 0.006),
        "projection": "max",
    }

