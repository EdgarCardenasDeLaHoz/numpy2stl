# Registration tests — the mask-producer seam (align/mask_source.py).
#
# Two things need proving, and they are different claims:
#   1. With no producer installed, every routed call is bit-identical to the
#      classic building_mask / building_edges it replaced.  This is what makes
#      the seam safe to land before any learned segmenter exists.
#   2. With a producer installed, the pipeline actually reaches it — the seam is
#      wired into the real call sites, not just importable.
# The second claim is the one a unit test of mask_source.py alone cannot make,
# so TestSeamIsWiredIntoThePipeline drives register_global() itself.
import numpy as np
import pytest

try:
    import cv2  # noqa: F401
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.fixture
def stl_like():
    """96x96 'STL' heightmap: gentle terrain ramp plus blocky buildings."""
    yy, xx = np.mgrid[0:96, 0:96]
    arr = 4.0 + 0.05 * yy + 0.03 * xx          # terrain the top-hat must remove
    arr[12:28, 12:30] += 24.0
    arr[40:56, 20:34] += 31.0
    arr[60:80, 55:78] += 18.0
    arr[20:32, 62:74] += 27.0
    return arr


@pytest.fixture
def osm_like():
    """The same footprints as an OSM heightmap: NaN everywhere but buildings."""
    arr = np.full((96, 96), np.nan, dtype=np.float64)
    arr[12:28, 12:30] = 24.0
    arr[40:56, 20:34] = 31.0
    arr[60:80, 55:78] = 18.0
    arr[20:32, 62:74] = 27.0
    return arr


class TestDefaultPathIsClassic:
    """No producer installed => byte-for-byte the classic implementation."""

    def test_mask_identical_to_building_mask(self, stl_like):
        from numpy2stl.registration.align import building_mask, produce_mask
        assert np.array_equal(produce_mask(stl_like, source="stl"),
                              building_mask(stl_like, source="stl"))

    def test_mask_identical_with_tuning_kwargs(self, stl_like):
        from numpy2stl.registration.align import building_mask, produce_mask
        kw = dict(cell_size_m=3.0, segment_features=True, fill_holes_px=None)
        assert np.array_equal(produce_mask(stl_like, source="stl", **kw),
                              building_mask(stl_like, source="stl", **kw))

    @pytest.mark.skipif(not HAS_CV2, reason="requires opencv")
    def test_edges_identical_to_building_edges(self, stl_like):
        from numpy2stl.registration.align import building_edges, produce_edges
        assert np.array_equal(
            produce_edges(stl_like, source="stl", allow_forced_split=False),
            building_edges(stl_like, source="stl", allow_forced_split=False))

    def test_osm_mask_identical(self, osm_like):
        from numpy2stl.registration.align import building_mask, produce_mask
        assert np.array_equal(produce_mask(osm_like, source="osm"),
                              building_mask(osm_like, source="osm"))

    def test_no_producer_active_by_default(self):
        from numpy2stl.registration.align import active_mask_producer
        assert active_mask_producer() is None


class TestProducerIsUsed:

    def test_producer_replaces_stl_mask(self, stl_like):
        from numpy2stl.registration.align import produce_mask, use_mask_producer
        sentinel = np.zeros(stl_like.shape, dtype=bool)
        sentinel[0:5, 0:5] = True

        def producer(hm, source="stl", **kwargs):
            return sentinel

        with use_mask_producer(producer):
            assert np.array_equal(produce_mask(stl_like, source="stl"), sentinel)

    @pytest.mark.skipif(not HAS_CV2, reason="requires opencv")
    def test_producer_drives_edges_too(self, stl_like):
        """produce_edges outlines the PRODUCER's mask, not the classic one."""
        from numpy2stl.registration.align import produce_edges, use_mask_producer
        blob = np.zeros(stl_like.shape, dtype=bool)
        blob[30:50, 30:50] = True

        with use_mask_producer(lambda hm, source="stl", **kw: blob):
            edges = produce_edges(stl_like, source="stl")
        # A solid 20x20 square erodes to 18x18: the outline is the 1px ring.
        assert edges.sum() == 20 * 20 - 18 * 18
        assert edges[30, 30] and not edges[35, 35]

    def test_osm_bypasses_producer(self, osm_like):
        """OSM masks are ground truth (~isnan of rasterized footprints)."""
        from numpy2stl.registration.align import (building_mask, produce_mask,
                                                  use_mask_producer)
        called = []

        def producer(hm, source="stl", **kwargs):
            called.append(source)
            return np.zeros(hm.shape, dtype=bool)

        with use_mask_producer(producer):
            out = produce_mask(osm_like, source="osm")
        assert called == []
        assert np.array_equal(out, building_mask(osm_like, source="osm"))

    def test_scope_is_restored(self, stl_like):
        from numpy2stl.registration.align import (active_mask_producer,
                                                  use_mask_producer)
        producer = lambda hm, source="stl", **kw: np.zeros(hm.shape, bool)  # noqa: E731
        with use_mask_producer(producer):
            assert active_mask_producer() is producer
            with use_mask_producer(None):        # inner A/B against classic
                assert active_mask_producer() is None
            assert active_mask_producer() is producer
        assert active_mask_producer() is None

    def test_scope_restored_on_exception(self):
        from numpy2stl.registration.align import (active_mask_producer,
                                                  use_mask_producer)
        with pytest.raises(RuntimeError):
            with use_mask_producer(lambda hm, source="stl", **kw: None):
                raise RuntimeError("boom")
        assert active_mask_producer() is None

    def test_use_config_installs_field(self, stl_like):
        from numpy2stl.registration.align import active_mask_producer, use_config
        from numpy2stl.registration.config import RegistrationConfig
        producer = lambda hm, source="stl", **kw: np.zeros(hm.shape, bool)  # noqa: E731
        with use_config(RegistrationConfig(mask_producer=producer)):
            assert active_mask_producer() is producer
        with use_config(RegistrationConfig()):
            assert active_mask_producer() is None

    def test_wrong_shape_rejected(self, stl_like):
        """Fail where the producer is still named, not deep in the search."""
        from numpy2stl.registration.align import produce_mask, use_mask_producer
        with use_mask_producer(lambda hm, source="stl", **kw: np.zeros((7, 7), bool)):
            with pytest.raises(ValueError, match="expected"):
                produce_mask(stl_like, source="stl")

    def test_non_bool_return_is_cast(self, stl_like):
        from numpy2stl.registration.align import produce_mask, use_mask_producer
        with use_mask_producer(lambda hm, source="stl", **kw: np.ones(hm.shape, np.uint8)):
            out = produce_mask(stl_like, source="stl")
        assert out.dtype == bool and out.all()


@pytest.mark.skipif(not HAS_CV2, reason="requires opencv")
class TestSeamIsWiredIntoThePipeline:
    """The claim a unit test of mask_source.py alone cannot make."""

    def test_register_global_reaches_the_producer(self, stl_like, osm_like):
        from numpy2stl.registration.align import (building_mask, register_global,
                                                  use_mask_producer)
        seen = {"n": 0}

        def passthrough(hm, source="stl", **kwargs):
            seen["n"] += 1
            return building_mask(hm, source=source, **kwargs)

        with use_mask_producer(passthrough):
            register_global(stl_like, osm_like, cell_size_m=3.0)
        assert seen["n"] > 0, "register_global never routed through the seam"

    def test_passthrough_producer_gives_identical_transform(self, stl_like, osm_like):
        """A producer that IS building_mask must change nothing — this is the
        regression guard on the routing edits themselves."""
        from numpy2stl.registration.align import (building_mask, register_global,
                                                  use_mask_producer)
        classic = register_global(stl_like, osm_like, cell_size_m=3.0)
        with use_mask_producer(
                lambda hm, source="stl", **kw: building_mask(hm, source=source, **kw)):
            seamed = register_global(stl_like, osm_like, cell_size_m=3.0)
        assert np.array_equal(classic["transform"], seamed["transform"])
        assert classic["edge_iou"] == seamed["edge_iou"]
        assert classic["angle_deg"] == seamed["angle_deg"]
        assert classic["scale"] == seamed["scale"]
