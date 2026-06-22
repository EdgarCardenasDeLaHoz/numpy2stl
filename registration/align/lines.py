"""Rotation estimation from gradient-orientation histograms.

Part of the align/ subpackage.  Rotation is read off the image GRADIENT (no
building mask / Hough lines — the legacy Hough line detector was removed as it
depended on segmentation quality and is no longer used).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


def gradient_angle_histogram(img: np.ndarray, n_bins: int = 180,
                             mag_pct: float = 90.0) -> np.ndarray:
    """
    Magnitude-weighted histogram of edge orientations from the IMAGE GRADIENT.

    Needs no building mask or Hough lines — it reads wall orientations straight off
    the height field via Sobel, so the rotation estimate is **independent of
    segmentation quality** (which fixed the 45°/90° aliases on irregular cities
    like Boston).

    For each strong-gradient cell (top ``mag_pct`` percentile of |∇|) we take the
    edge-tangent angle (gradient direction + 90°, folded to [0,180)) weighted by
    |∇|.  Returns the unnormalised angle histogram.
    """
    a = np.nan_to_num(np.asarray(img, dtype=np.float64))
    if HAS_CV2:
        gx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
    else:
        gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    if mag.max() <= 0:
        return np.zeros(n_bins)
    thr = np.percentile(mag, mag_pct)
    sel = mag > thr
    if not sel.any():
        sel = mag > 0
    ang = (np.degrees(np.arctan2(gy[sel], gx[sel])) + 90.0) % 180.0
    hist, _ = np.histogram(ang, bins=n_bins, range=(0.0, 180.0), weights=mag[sel])
    return hist.astype(np.float64)


def rotation_from_angle_histograms(
    hist_src: np.ndarray,
    hist_tgt: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Find the rotation (degrees) that best aligns hist_src to hist_tgt.

    A rotation of θ° shifts all line angles by θ°, which shifts the histogram
    by θ * n_bins / 180 bins.  The circular cross-correlation of the two
    histograms peaks at that shift.  The result is translation-invariant —
    the same rotation is found regardless of how the images are translated.

    Parameters
    ----------
    hist_src, hist_tgt : 1D float arrays of equal length (n_bins bins, 0-180°)

    Returns
    -------
    (rotation_deg, xcorr_curve)
        rotation_deg : best rotation in degrees, in [-90, 90)
        xcorr_curve  : full cross-correlation (length n_bins) for visualisation
    """
    n = len(hist_src)
    s = hist_src / (hist_src.sum() + 1e-9)
    t = hist_tgt / (hist_tgt.sum() + 1e-9)

    # 1-D circular cross-correlation via FFT
    xcorr = np.fft.irfft(np.conj(np.fft.rfft(s)) * np.fft.rfft(t), n=n)

    peak_bin = int(np.argmax(xcorr))
    # Sub-bin (sub-degree) refinement: parabolic interpolation through the peak
    # and its two circular neighbours.  The histogram is only 1°/bin, but the
    # true grid angle is continuous — the parabola vertex recovers the fraction.
    y0 = xcorr[(peak_bin - 1) % n]
    y1 = xcorr[peak_bin]
    y2 = xcorr[(peak_bin + 1) % n]
    denom = (y0 - 2.0 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    delta = float(np.clip(delta, -0.5, 0.5))
    rot_deg = (peak_bin + delta) * 180.0 / n   # convert bin → degrees
    if rot_deg > 90.0:
        rot_deg -= 180.0                       # fold to [-90, 90)

    return float(rot_deg), xcorr
