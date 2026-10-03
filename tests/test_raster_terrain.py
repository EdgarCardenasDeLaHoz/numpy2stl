"""numpy2stl.raster.terrain: the ground under a surface heightmap (F-STL2NUMPY, 2026-10-03)."""
import numpy as np
import pytest

from numpy2stl.raster import estimate_dtm, ground_mask_pmf, ground_mask_steps, terrain_residual

CELL = 2.0   # metres per cell


def _hillside(slope=0.25, n=200):
    """A plane rising *slope* m per m to the east, with 10 m buildings 20-60 m wide on it."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    ground = slope * x * CELL
    dsm = ground.copy()
    built = np.zeros_like(dsm, bool)
    for r0, c0, h, w in ((20, 20, 10, 10), (80, 60, 25, 15), (140, 120, 30, 30), (40, 150, 12, 20)):
        built[r0:r0 + h, c0:c0 + w] = True
    dsm[built] += 10.0
    return dsm, ground, built


@pytest.mark.parametrize("slope", [0.0, 0.25, 0.6])
def test_steps_find_every_roof_and_keep_the_slope(slope):
    dsm, truth, built = _hillside(slope=slope)
    ground = ground_mask_steps(dsm, CELL)
    assert (ground == ~built).all()                        # exact split, any roof width
    err = np.abs(estimate_dtm(dsm, CELL, ground=ground) - truth)
    assert err[~built].max() < 1e-9 and err.mean() < 0.15


def test_steps_beat_the_wide_opening_on_a_hillside():
    """The 2026-05 finding: a single wide opening flattens hills and cuts hillside
    buildings into the ground."""
    dsm, truth, built = _hillside(slope=0.35)
    residual, valid = terrain_residual(dsm, cell_size_m=CELL)
    opening_err = np.abs((dsm - residual) - truth)[valid].mean()
    steps_err = np.abs(estimate_dtm(dsm, CELL, ground=ground_mask_steps(dsm, CELL)) - truth).mean()
    assert steps_err < opening_err / 5


def test_progressive_filter_misses_wide_roofs_on_a_slope():
    """Why the default is the step method: a 60 m roof is only caught by a window over
    60 m, where the slope allowance already exceeds its 10 m height."""
    dsm, truth, built = _hillside(slope=0.25)
    ground = ground_mask_pmf(dsm, CELL, slope=0.3)
    assert (ground & built).sum() > 0.3 * built.sum()     # at slope 0.3; 0.15 is the default
    assert not (ground & built)[20:30, 20:30].any()       # the 20 m building is caught


@pytest.mark.parametrize("slope", [0.0, 0.3])
def test_default_ground_is_where_both_masks_agree(slope):
    dsm, truth, built = _hillside(slope=slope)
    err = np.abs(estimate_dtm(dsm, CELL) - truth)
    # Slopes steeper than the progressive filter's 0.15 lose a few bare cells to
    # interpolation; the error stays small.
    assert err[~built].mean() < 0.05 and err[~built].max() < 2.0
    assert np.median(err[built]) < 0.5


def test_off_model_cells_stay_nan_and_flat_ground_is_kept():
    dsm = np.full((60, 60), 5.0)
    dsm[:, :5] = np.nan
    dsm[20:30, 20:30] = 15.0
    dtm = estimate_dtm(dsm, CELL, ground=ground_mask_steps(dsm, CELL))
    assert np.isnan(dtm[:, :5]).all()
    assert np.allclose(dtm[:, 5:], 5.0)
