"""numpy2stl is geo-free: no module may fetch map data (F-ARCH decision 2026-09-26).

OSM (osmnx / Overpass), HTTP clients and point-cloud readers live in map2stl's
city2stl; this scans every numpy2stl source file, including function-local imports.
"""
import ast
from pathlib import Path

import pytest

import numpy2stl

FORBIDDEN = {"osmnx", "requests", "pdal", "py3dep", "urllib", "httpx", "overpy", "map2stl",
             "city2stl", "geo2stl", "app"}
SRC = Path(numpy2stl.__file__).parent


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno


def test_no_module_imports_network_or_geo_fetchers():
    bad = [f"{p.relative_to(SRC)}:{line} imports {mod}"
           for p in sorted(SRC.rglob("*.py")) for mod, line in _imports(p) if mod in FORBIDDEN]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("module", ["numpy2stl.applications.cities",
                                    "numpy2stl.applications.lidar"])
def test_removed_geo_modules_point_at_map2stl(module):
    import importlib
    with pytest.raises(ImportError, match="city2stl"):
        importlib.import_module(module)


def test_moved_names_point_at_map2stl():
    with pytest.raises(ImportError, match="city2stl.osm_raster"):
        from numpy2stl.applications import get_osm_building_heightmap  # noqa: F401
