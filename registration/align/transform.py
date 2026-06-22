"""Heightmap preprocessing and affine warp application.

Part of the align/ subpackage (split from the former align.py).
"""
from __future__ import annotations

import logging
import time
from math import atan2, degrees, sqrt

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from scipy.ndimage import sobel, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    sobel = gaussian_filter = None
    HAS_SCIPY = False


def _preprocess_for_registration(
    heightmap: np.ndarray,
    nan_fill: str = "zero",
    edge_method: str = "sobel",
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """
    Convert a float64 heightmap to a uint8 gradient-magnitude image.

    Steps:
    1. Fill NaN with 0 (buildings are positive height on zero ground)
    2. Normalize to [0, 1]
    3. Pre-blur (Gaussian) to reduce noise
    4. Sobel gradient magnitude (smooth edges, good for ECC cost landscape)
    5. Normalize by 99th percentile → clip → uint8

    Parameters
    ----------
    heightmap   : (rows, cols) float64, may contain NaN
    nan_fill    : 'zero' — replace NaN with 0
    edge_method : 'sobel' (default) or 'canny'
    blur_sigma  : pre-Sobel Gaussian blur sigma

    Returns
    -------
    uint8 ndarray, same shape
    """
    arr = heightmap.copy().astype(np.float64)

    # 1. NaN fill
    arr[np.isnan(arr)] = 0.0

    # 2. Normalize
    vmax = arr.max()
    if vmax > 0:
        arr /= vmax

    # 3. Pre-blur
    if HAS_SCIPY and blur_sigma > 0:
        arr = gaussian_filter(arr, sigma=blur_sigma)
    elif blur_sigma > 0 and HAS_CV2:
        k = max(3, int(blur_sigma * 3) | 1)
        arr = cv2.GaussianBlur(arr.astype(np.float32), (k, k), blur_sigma).astype(np.float64)

    # 4. Gradient magnitude
    if edge_method == "canny" and HAS_CV2:
        u8 = (arr * 255).clip(0, 255).astype(np.uint8)
        grad = cv2.Canny(u8, 50, 150).astype(np.float64)
    else:
        if HAS_SCIPY:
            gx = sobel(arr, axis=1)
            gy = sobel(arr, axis=0)
        else:
            # pure-numpy Sobel fallback
            k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
            from scipy.signal import convolve2d
            gx = np.pad(arr, 1, mode='edge')
            gx = sum(k[i, j] * np.roll(np.roll(arr, -i + 1, 0), -j + 1, 1)
                     for i in range(3) for j in range(3))
            gy = gx.T
        grad = np.sqrt(gx ** 2 + gy ** 2)

    # 5. Normalize by 99th percentile → uint8
    p99 = np.percentile(grad, 99)
    if p99 > 0:
        grad = grad / p99
    grad = grad.clip(0, 1)
    return (grad * 255).astype(np.uint8)


def apply_transform(
    image: np.ndarray,
    transform: np.ndarray,
    output_shape: tuple[int, int] | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Apply a 2×3 affine or 3×3 homography warp matrix to a heightmap.

    Parameters
    ----------
    image        : (rows, cols) float64 source heightmap
    transform    : (2, 3) affine or (3, 3) homography warp matrix
    output_shape : (rows, cols) of the output; defaults to image.shape
    fill_value   : fill for pixels outside the source domain (NaN by default)

    Returns
    -------
    ndarray (rows, cols) float64, warped image
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required. Install with: pip install opencv-python")

    if output_shape is None:
        output_shape = image.shape

    out_h, out_w = output_shape
    M = transform.astype(np.float32)
    is_homography = M.shape == (3, 3)

    def _warp(img, border):
        if is_homography:
            return cv2.warpPerspective(
                img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=border)
        return cv2.warpAffine(
            img, M, (out_w, out_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border)

    warped = _warp(image.astype(np.float32), 0.0)
    mask = _warp(np.ones(image.shape, dtype=np.float32), 0.0)
    warped_nan = _warp(np.isnan(image).astype(np.float32), 1.0)

    result = warped.astype(np.float64)
    # Pixels outside source domain OR where source was NaN get fill_value
    result[mask < 0.5] = fill_value
    result[warped_nan > 0.5] = fill_value

    return result


# ---------------------------------------------------------------------------
# Internal matrix helpers
# ---------------------------------------------------------------------------


def _decompose_matrix(M: np.ndarray) -> tuple[float, float]:
    """Extract scale and rotation angle (degrees) from a 2×3 affine matrix."""
    scale = float(sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
    angle_deg = float(degrees(atan2(M[1, 0], M[0, 0])))
    return scale, angle_deg


def _compose_resize_scale(M: np.ndarray, scale_x: float, scale_y: float) -> None:
    """In-place: adjust M so it maps original (pre-resize) source coords → target."""
    M[0, 0] *= scale_x
    M[0, 1] *= scale_y
    M[1, 0] *= scale_x
    M[1, 1] *= scale_y
