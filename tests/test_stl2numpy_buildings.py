"""numpy2stl.stl2numpy.buildings: a building table from a surface and its ground (F-STL2NUMPY)."""
import numpy as np
import pytest

from numpy2stl.stl2numpy.buildings import building_table, roof_shape

CELL = 2.0


def _scene(slope=0.2, n=120):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    ground = slope * x * CELL
    dsm = ground.copy()
    dsm[10:20, 10:30] += 12.0                          # 40 x 20 m, 12 m flat
    dsm[10:20, 30:40] += 25.0                          # touching it, 25 m: a separate building
    dsm[60:80, 60:80] = dsm[60:80, 60:80] + 8.0 + 0.5 * (np.arange(20) * CELL)[None, :]  # sloped roof
    dsm[100:102, 100:102] += 6.0                       # 16 m2: under the 20 m2 minimum
    return dsm, ground


def test_table_splits_at_roof_steps_and_measures_each_building():
    dsm, ground = _scene()
    t = building_table(dsm, ground, CELL)
    rows = sorted(t["buildings"], key=lambda r: r["area_m2"])
    assert len(rows) == 3
    low, high = sorted(rows[:2], key=lambda r: r["height_m"])
    assert low["area_m2"] == 800 and high["area_m2"] == 400
    assert low["height_m"] == pytest.approx(12.0) and high["height_m"] == pytest.approx(25.0)
    assert low["roof"]["shape"] == "sloped" or low["roof"]["shape"] == "flat"
    sloped = rows[2]
    assert sloped["area_m2"] == 1600 and sloped["roof"]["shape"] == "sloped"
    assert t["labels"].max() == 3 and t["labels"][100, 100] == 0


def test_footprint_polygon_is_in_cells():
    dsm = np.zeros((40, 40))
    dsm[5:15, 20:30] = 10.0
    (row,) = building_table(dsm, np.zeros_like(dsm), CELL)["buildings"]
    xs, ys = zip(*row["polygon"], strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == (20, 29, 5, 14)
    assert row["roof"]["shape"] == "flat" and row["base_m"] == 0


@pytest.mark.parametrize("pitch, shape", [(0.0, "flat"), (0.5, "sloped")])
def test_roof_shape(pitch, shape):
    y, x = np.mgrid[0:10, 0:10] * 1.0
    assert roof_shape(x.ravel(), y.ravel(), (pitch * x).ravel())["shape"] == shape
    gable = 5 - np.abs(x - 4.5)
    assert roof_shape(x.ravel(), y.ravel(), gable.ravel())["shape"] == "complex"


def test_region_features_tell_a_roof_from_a_canopy():
    """F-TREES: a flat roof is smooth with a sharp edge; a crown is rough and tapers."""
    from numpy2stl.stl2numpy.buildings import FEATURE_NAMES, region_features

    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:80, 0:80]
    dsm = np.zeros((80, 80))
    dsm[10:30, 10:30] = 12.0
    d2 = (x - 55) ** 2 + (y - 55) ** 2
    dsm = np.maximum(dsm, np.clip(10 - 0.12 * d2, 0, None) + rng.normal(0, 0.8, (80, 80)) * (d2 < 80))
    t = building_table(dsm, np.zeros_like(dsm), CELL)
    f = dict(zip(FEATURE_NAMES, region_features(dsm, np.zeros_like(dsm), t["labels"], CELL).T, strict=True))
    roof, crown = (0, 1) if t["buildings"][0]["area_m2"] == 1600 else (1, 0)
    assert f["roughness_m"][roof] < 0.1 < f["roughness_m"][crown]
    assert f["height_std_m"][roof] < 0.1 < f["height_std_m"][crown]
    assert f["edge_rise"][roof] == pytest.approx(1.0) and f["edge_rise"][crown] < 0.6
    assert f["rectangularity"][roof] == pytest.approx(1.0)
