"""Offline end-to-end smoke test for the registration orchestrator.

`register_city_stl` and its stage helpers (`_simplify_stage`, `_run_registration`,
`_run_comparison`) had no automated coverage — the unit tests exercise
`register`/`compare`/`apply_transform` in isolation but never the wiring that
glues them together. This test mocks the two IO boundaries (STL rasterization and
the OSM fetch) and drives the whole pipeline on synthetic heightmaps, so a broken
import or data hand-off between stages fails fast here rather than only in a live
run. It was added alongside the B1 split (pipeline.py → stages/).
"""

from unittest.mock import patch

import numpy as np
import pytest

from numpy2stl.registration import register_city_stl
from numpy2stl.registration.types import CityRegistrationReport


def _heightmap(resolution: int) -> np.ndarray:
    """A centred square 'building' blob on a NaN background."""
    hm = np.full((resolution, resolution), np.nan, dtype=np.float32)
    q = resolution // 4
    hm[q:3 * q, q:3 * q] = 50.0
    return hm


def _fake_mesh_to_heightmap(stl_file, resolution=512, **kw):
    return {
        "heightmap": _heightmap(resolution),
        "bounds": {"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 60.0)},
    }


def _fake_osm_heightmap(target, resolution=512, **kw):
    return {
        "heightmap": _heightmap(resolution),
        "bounds": {"x": (-75.2, -75.1), "y": (39.9, 40.0)},
    }


def _fake_semantic_masks(target, resolution=512, **kw):
    return {"vegetation": None, "water": None}


@patch("numpy2stl.applications.cities.get_osm_semantic_masks", _fake_semantic_masks)
@patch("numpy2stl.applications.cities.get_osm_building_heightmap", _fake_osm_heightmap)
@patch("numpy2stl.stl2numpy.heightmap.mesh_to_heightmap", _fake_mesh_to_heightmap)
def test_register_city_stl_wiring_offline():
    """Full pipeline runs offline on synthetic data and returns a report.

    Uses a bbox tuple for city_name (skips geocoding/scale estimation) and
    detect_resolution_factor=1 (skips the hi-res footprint re-render). Asserts the
    stages produced wired-up outputs, not registration accuracy.
    """
    report = register_city_stl(
        stl_file="synthetic.stl",
        city_name=(40.0, 39.9, -75.1, -75.2),   # bbox → no geocode, known_scale=None
        resolution=512,                          # == REGISTER_RES, no re-render branch
        out_dir=False,                           # suppress HTML/PNG output
        detect_resolution_factor=2,              # exercise the hi-res footprint path
                                                 # (which calls _inpaint_stl_nan in compare)
        refine=True,                             # exercise projection discovery + ECC
    )

    assert isinstance(report, CityRegistrationReport)
    assert report.registration is not None
    assert report.comparison is not None
    assert report.stl_aligned.shape == report.osm_heightmap.shape
    # The recovered transform is a 2x3 affine.
    assert report.registration.transform.shape == (2, 3)
    assert report.step_timings, "stage timings should be recorded"
