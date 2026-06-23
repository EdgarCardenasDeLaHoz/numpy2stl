# Registration tests — report (split from test_registration.py, B6).
# Tests for the registration pipeline
# No-network unit tests run always; integration tests require osmnx + a real STL.
import numpy as np
import pytest

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import osmnx  # noqa: F401
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False

try:
    import geopandas  # noqa: F401
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ---------------------------------------------------------------------------
# Synthetic fixtures (no files, no network)
# ---------------------------------------------------------------------------


class TestCityRegistrationReport:

    def _make_report(self):
        from numpy2stl.registration.types import (
            CityRegistrationReport, ComparisonResult, RegistrationResult,
        )
        reg = RegistrationResult(
            transform=np.eye(2, 3),
            confidence=0.75,
            scale=1.0,
            angle_deg=0.0,
            n_iterations=50,
            converged=True,
        )
        diff = np.full((16, 16), np.nan)
        diff[5:10, 5:10] = 2.0
        overlap = ~np.isnan(diff)
        comp = ComparisonResult(
            difference=diff,
            building_diff_map=diff.copy(),
            overlap_mask=overlap,
            missing_in_osm=np.zeros((16, 16), dtype=bool),
            missing_in_stl=np.zeros((16, 16), dtype=bool),
            rmse=2.0, mae=2.0, bias=2.0, correlation=0.9,
            rank_correlation=0.85, mape=20.0,
            coverage_pct=50.0, footprint_iou=0.6, dice_score=0.75, n_overlap=25,
            height_scale_used=1.5, height_offset_used=0.0,
            height_ratio_mean=1.0, height_ratio_std=0.1,
        )
        return CityRegistrationReport(
            region_name="Test City",
            stl_file="test.stl",
            stl_heightmap=np.ones((16, 16)),
            osm_heightmap=np.ones((16, 16)),
            stl_aligned=np.ones((16, 16)),
            registration=reg,
            comparison=comp,
            step_timings=[("load", 0.1), ("register", 1.2)],
        )

    def test_dataclass_attributes(self):
        report = self._make_report()
        assert report.region_name == "Test City"
        assert report.registration.confidence == 0.75
        assert report.comparison.rmse == 2.0
        assert len(report.step_timings) == 2

    def test_is_frozen(self):
        report = self._make_report()
        with pytest.raises((AttributeError, TypeError)):
            report.region_name = "Other"


class TestWriteRegistrationReport:

    def test_creates_index_html(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        out = write_registration_report(tmp_path, report)
        assert out.exists()
        assert out.name == "index.html"

    def test_creates_assets_folder(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        assert (tmp_path / "assets").is_dir()

    def test_creates_comparison_png(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        assert (tmp_path / "assets" / "comparison.png").exists()

    def test_index_html_contains_region_name(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "Test City" in html

    def test_index_html_contains_stats(self, tmp_path):
        from numpy2stl.registration.html_report import write_registration_report
        report = TestCityRegistrationReport()._make_report()
        write_registration_report(tmp_path, report)
        html = (tmp_path / "index.html").read_text(encoding="utf-8")
        low = html.lower()
        assert "rmse" in low
        assert "correlation" in low
        assert "dice" in low

