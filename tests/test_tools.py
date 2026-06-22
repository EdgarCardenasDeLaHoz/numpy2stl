# Tests for tools.py - Utility functions
import numpy as np
import pytest


class TestRescale:
    """Test rescale function."""

    def test_rescale_import_error(self):
        """Test that rescale raises ImportError if opencv not installed."""
        from numpy2stl.utils.image import HAS_CV2

        if not HAS_CV2:
            from numpy2stl import rescale

            im = np.random.rand(100, 100)
            with pytest.raises(ImportError, match="opencv-python"):
                rescale(im)

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="opencv-python not installed"), reason="opencv-python required"
    )
    def test_basic_rescaling(self):
        """Test basic image rescaling."""
        from numpy2stl import rescale

        im = np.random.rand(100, 100) * 100
        scaled = rescale(im, max_size=50, height=20, base=10)

        # Should be rescaled to max_size
        assert scaled.shape[0] <= 50
        assert scaled.shape[1] <= 50

        # Values should be scaled to height + base range
        assert scaled.min() >= 10  # base
        assert scaled.max() <= 30  # base + height

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="opencv-python not installed"), reason="opencv-python required"
    )
    def test_clipping(self):
        """Test percentile clipping."""
        from numpy2stl import rescale

        im = np.random.RandomState(42).rand(200, 200) * 100
        scaled = rescale(im, max_size=200, height=20, base=10, clip=[10, 90])

        # Clipping should reduce outliers
        original_range = im.ptp()
        scaled_range = (scaled - 10).ptp()

        # Scaled range should be close to height (20)
        assert 15 <= scaled_range <= 25


class TestResizeMax:
    """Test resize_max function."""

    def test_resize_max_import_error(self):
        """Test that resize_max raises ImportError if opencv not installed."""
        from numpy2stl.utils.image import HAS_CV2

        if not HAS_CV2:
            from numpy2stl import resize_max

            im = np.random.rand(100, 100)
            with pytest.raises(ImportError, match="opencv-python"):
                resize_max(im)

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="opencv-python not installed"), reason="opencv-python required"
    )
    def test_basic_resizing(self):
        """Test basic image resizing."""
        from numpy2stl import resize_max

        im = np.random.rand(1000, 500)
        resized = resize_max(im, max_size=200)

        # Longest dimension should be 200
        assert max(resized.shape) == 200
        # Aspect ratio should be preserved
        assert resized.shape[0] / resized.shape[1] == pytest.approx(
            im.shape[0] / im.shape[1], rel=0.01
        )

    @pytest.mark.skipif(
        not pytest.importorskip("cv2", reason="opencv-python not installed"), reason="opencv-python required"
    )
    def test_no_upscaling(self):
        """Test that small images are not upscaled."""
        from numpy2stl import resize_max

        im = np.random.rand(50, 50)
        resized = resize_max(im, max_size=200)

        # Should not upscale
        assert resized.shape[0] == 50
        assert resized.shape[1] == 50
