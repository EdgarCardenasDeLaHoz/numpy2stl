"""Spatial-scale estimation (area ratio + Fourier profile) — report diagnostics.

Part of the align/ subpackage.  The scale actually used by the pipeline is the
geometric anchor (1/osm_margin); `estimate_scale` here is kept for the report's
scale diagnostics.  (The legacy log-polar / grid-period estimators were removed
with the ECC register() path.)
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

from .segmentation import building_mask


def estimate_scale(source: np.ndarray, target: np.ndarray) -> dict:
    """
    Estimate the source→target scale from spatial-texture statistics, robust to
    height noise and single-building outliers.

    Combines two whole-image measures (no reliance on extremes like the tallest
    building):
      - area ratio: sqrt(target_building_area / source_building_area) — direct,
        assumes comparable building-coverage fraction.
      - Fourier profile match: resample factor that best aligns the radially-
        averaged power spectra of the two building masks (matches the dominant
        spatial-frequency/block-grid content).

    Returns
    -------
    dict: {'scale', 'area_scale', 'fourier_scale', 'fourier_corr'}
        'scale' is the recommended value (geometric mean of the two when the
        Fourier match is confident, else the area ratio).
    """
    s_mask = building_mask(source, source="stl").astype(np.float64)
    t_mask = building_mask(target, source="osm").astype(np.float64)

    s_area, t_area = s_mask.sum(), t_mask.sum()
    area_scale = float(np.sqrt(t_area / s_area)) if s_area > 0 else 1.0

    fourier_scale, fourier_corr = _fourier_profile_scale(s_mask, t_mask)

    # The area ratio is the more reliable estimate in practice; the Fourier
    # profile match tends to run high when neither image has a sharp grid peak.
    # Weight the area ratio heavily and only nudge toward the Fourier value.
    if np.isfinite(fourier_scale) and fourier_corr > 0.6:
        scale = float(area_scale ** 0.8 * fourier_scale ** 0.2)
    else:
        scale = area_scale

    scale = float(np.clip(scale, 0.25, 4.0))
    return {
        "scale": scale,
        "area_scale": area_scale,
        "fourier_scale": float(fourier_scale),
        "fourier_corr": float(fourier_corr),
    }


def _fourier_profile_scale(s_mask: np.ndarray, t_mask: np.ndarray) -> tuple:
    """Scale that best aligns the two radial power-spectrum profiles."""
    def _radial(img):
        a = img - img.mean()
        wy = np.hanning(a.shape[0])[:, None]
        wx = np.hanning(a.shape[1])[None, :]
        F = np.fft.fftshift(np.fft.fft2(a * wy * wx))
        P = np.abs(F) ** 2
        h, w = P.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
        rmax = min(cy, cx)
        return np.array([P[r == i].mean() if np.any(r == i) else 0.0
                         for i in range(rmax)])

    rp_s, rp_t = _radial(s_mask), _radial(t_mask)
    L = min(len(rp_s), len(rp_t))
    if L < 16:
        return float("nan"), 0.0

    lo = 2  # skip DC / lowest-frequency bins (no scale information there)
    s_lo, s_hi, n_s = 0.5, 2.0, 151
    best_s, best_c = float("nan"), -1.0
    for s in np.linspace(s_lo, s_hi, n_s):
        # Resample the target profile onto the source frequency axis by factor s.
        idx = (np.arange(L) * s).astype(int)
        # Only correlate over NON-saturated bins; clipping to the last index
        # creates a degenerate flat tail that fakes a perfect correlation.
        valid = (idx >= lo) & (idx < len(rp_t))
        if int(valid.sum()) < max(16, int(0.4 * L)):
            continue
        a = rp_s[:L][valid]
        b = rp_t[idx[valid]]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        a = a / (a.max() + 1e-9)
        b = b / (b.max() + 1e-9)
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best_c:
            best_c, best_s = c, float(s)

    # A best at the search boundary means the true factor is outside the window
    # (or the profiles carry no scale signal) — report as unreliable so callers
    # fall back to the area ratio instead of trusting a boundary artefact.
    if not np.isfinite(best_s) or best_s <= s_lo + 1e-6 or best_s >= s_hi - 1e-6:
        return float("nan"), 0.0
    return best_s, best_c
