"""Offline end-to-end smoke test for the registration orchestrator.

`register_city_stl` and its stage helpers (`_simplify_stage`, `_run_registration`,
`_run_comparison`) had no automated coverage — the unit tests exercise
`register`/`compare`/`apply_transform` in isolation but never the wiring that
glues them together. This test mocks the two IO boundaries (STL rasterization and
the reference source) and drives the whole pipeline on synthetic heightmaps, so a broken
import or data hand-off between stages fails fast here rather than only in a live
run. It was added alongside the B1 split (pipeline.py → stages/).
"""

from unittest.mock import patch

import numpy as np
import pytest

from numpy2stl.registration import StaticReference, register_city_stl
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


class _FakeReference:
    """Offline ReferenceSource: a fixed frame, no scale anchor, no masks."""

    name = "synthetic"

    def resolve_target(self, stl_z_max, stl_xy_extent, margin):
        return "frame", False

    def building_heightmap(self, target, resolution):
        return {"heightmap": _heightmap(resolution),
                "bounds": {"x": (0.0, 1.0), "y": (0.0, 1.0)},
                "cell_size_m": 2.0 * 512 / resolution}

    def semantic_masks(self, target, resolution):
        return {"vegetation": None, "water": None}

    def ndsm(self, target, resolution):
        return None

    def m_per_unit(self, stl_z_max):
        return None

    def candidate_targets(self, stl_z_max, stl_xy_extent, margin):
        return []


@patch("numpy2stl.stl2numpy.heightmap.mesh_to_heightmap", _fake_mesh_to_heightmap)
def test_register_city_stl_wiring_offline():
    """Full pipeline runs offline on synthetic data and returns a report.

    An unanchored reference (known_scale=None, no centre search) and
    detect_resolution_factor=2 (exercises the hi-res footprint re-render). Asserts
    the stages produced wired-up outputs, not registration accuracy.
    """
    report = register_city_stl(
        stl_file="synthetic.stl",
        reference=_FakeReference(),
        resolution=512,                          # == REGISTER_RES, no re-render branch
        out_dir=False,                           # suppress HTML/PNG output
        detect_resolution_factor=2,              # exercise the hi-res footprint path
                                                 # (which calls _inpaint_stl_nan in compare)
        refine=True,                             # exercise projection discovery + ECC
    )

    assert isinstance(report, CityRegistrationReport)
    assert report.region_name == "synthetic"
    assert report.cell_size_m == 2.0
    assert report.registration is not None
    assert report.comparison is not None
    assert report.stl_aligned.shape == report.osm_heightmap.shape
    # The recovered transform is a 2x3 affine.
    assert report.registration.transform.shape == (2, 3)
    assert report.step_timings, "stage timings should be recorded"


def test_static_reference_resamples():
    hm = _heightmap(64)
    ref = StaticReference(hm, cell_size_m=4.0)
    out = ref.building_heightmap(None, 128)
    assert out["heightmap"].shape == (128, 128)
    assert out["cell_size_m"] == 2.0
    assert ref.resolve_target(1.0, 1.0, 1.5) == (None, False)


def test_city_name_is_rejected_with_pointer():
    with pytest.raises(TypeError, match="city2stl.registration"):
        register_city_stl("synthetic.stl", "Philadelphia, PA, USA", out_dir=False)
