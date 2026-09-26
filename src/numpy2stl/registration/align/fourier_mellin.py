"""Fourier–Mellin registration (rotation + scale, grid-free) — PROTOTYPE.

The textbook similarity-registration method (Reddy & Chatterji, 1996): the
magnitude spectrum of an image is *translation-invariant*, a rotation of the
image rotates its spectrum by the same angle, and a uniform scaling scales the
spectrum by the inverse factor.  Re-sampling the magnitude spectrum onto a
log-polar grid turns that rotation into a shift along the angular axis and that
scaling into a shift along the log-radius axis — both then recovered by a single
phase correlation.  No street grid, no segmentation, no parameter sweep.

Why this over the current gradient-histogram + scale-sweep:
  - grid-free: works on irregular / radial cities (Boston, Paris) that have no
    dominant orthogonal orientation for the gradient histogram to lock onto;
  - joint: recovers rotation AND scale in one transform (the sweep needs a
    separate pass and a prior window);
  - principled: a standard, well-characterised algorithm rather than a stack of
    bespoke heuristics.

Improvements over the legacy `scale._coarse_estimate` prototype:
  - Hanning window before the FFT (kills the frame-edge cross artefact that
    otherwise dominates the spectrum and biases the angle);
  - Reddy–Chatterji high-pass emphasis on the magnitude (suppresses the DC blob
    that swamps the informative mid-band, instead of a hard DC punch-out);
  - explicit resolution of the 180° spectrum ambiguity (the magnitude spectrum
    is centrosymmetric, so rotation is only recoverable mod 180°) by warping the
    source for each candidate and scoring the normalised cross-correlation;
  - `refine_overlap`: an iterative overlap-masking loop that makes FM usable on
    PARTIAL overlap (the failure mode that sank it on real STL↔OSM data, where
    the STL covers only part of the OSM frame).  Each pass warps the source by
    the running estimate, masks BOTH images to their intersection, and re-runs
    FM for the residual rotation+scale+translation, composing into the total.

This is a prototype: it exposes `fourier_mellin_register(source, target)` for
evaluation against the production gradient path; it is not yet wired into
`register_global`.

Evaluated against real STL/OSM data (Miami, tight downtown bbox) while
investigating whether a "match the coastline first" coarse pre-pass would
help registration quality:
  - Water masks ALONE (STL's companion _Water.stl mesh vs OSM's water
    polygons) gave a garbage result (confidence 0.03, angle way off) — OSM
    water coverage in a tight downtown bbox was only 2.1% of the frame, too
    sparse for the log-polar spectral method to lock onto reliably.
  - Building masks alone: correct (angle 0.00° matching the known-good
    answer, plausible scale, confidence 0.17) — FM works fine on real data
    when there's enough signal.
  - Building + water combined (water weighted 0.3-0.5x buildings): matched
    the building-only result, confidence ticked up slightly (0.17→0.18) —
    doesn't hurt, isn't a clear win either. Weighting water at 1.0x (equal to
    buildings) broke the result (water's low-frequency content dominated the
    spectrum and drowned the useful building signal).
  - Self-registration on a synthetic PERFECTLY PERIODIC building grid (the
    exact pathology that made register_global's edge-IoU sweep lock onto a
    spurious 0.7x scale in tests/test_registration_register.py, see that
    test's docstring) came back scale=1.000 angle=0.000 via FM — the
    spectral method doesn't share that failure mode, since it isn't a
    discrete per-candidate overlap sweep.
Conclusion: FM is real, works, and is structurally immune to a known
failure mode of the current search — but the water-based coarse-match idea
specifically wasn't supported by evidence on real data, and wiring FM in as
a general coarse pre-pass wasn't clearly justified by the (modest) measured
gain against the real cost of adding it to the pipeline. Left unwired
pending either a case where the current method visibly fails on real data,
or evidence from a city with substantially more water coverage in frame.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    cv2 = None
    HAS_CV2 = False


class FourierMellinResult(NamedTuple):
    angle_deg: float        # source→target rotation (degrees, CCW positive)
    scale: float            # source→target uniform scale
    dx: float               # source→target translation of the image centre (px)
    dy: float
    confidence: float       # log-polar phase-correlation peak response (0–1)
    overlap_corr: float     # normalised xcorr over the recovered overlap
    transform: np.ndarray   # 2×3 affine mapping source → target


def _to_float(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """NaN-filled float32 resized to `shape`."""
    a = np.nan_to_num(np.asarray(img, dtype=np.float32))
    if a.shape != shape:
        a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return a


def _highpass_filter(h: int, w: int) -> np.ndarray:
    """Reddy–Chatterji high-pass emphasis filter H = (1−X)(2−X), X = cos·cos."""
    eta = np.cos(np.pi * (np.linspace(-0.5, 0.5, h)))[:, None]
    xi = np.cos(np.pi * (np.linspace(-0.5, 0.5, w)))[None, :]
    X = eta * xi
    return (1.0 - X) * (2.0 - X)


def _log_magnitude_spectrum(img: np.ndarray, hp: np.ndarray) -> np.ndarray:
    """Windowed, high-pass-emphasised log-magnitude spectrum (fftshifted)."""
    h, w = img.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    f = np.fft.fftshift(np.fft.fft2(img * win))
    mag = np.abs(f) * hp
    return np.log1p(mag).astype(np.float32)


def _rot_scale_response(src: np.ndarray, tgt: np.ndarray,
                        max_scale_ratio: float) -> tuple[float, float, float]:
    """Core Fourier–Mellin: recover (angle_deg mod 180, scale, response).

    Both arrays must already share a shape.  Angle is returned in [-90, 90);
    the 180° ambiguity is left to the caller (resolved by overlap correlation).
    """
    h, w = tgt.shape
    hp = _highpass_filter(h, w)
    m_src = _log_magnitude_spectrum(src, hp)
    m_tgt = _log_magnitude_spectrum(tgt, hp)

    # Log-polar resample.  cv2.warpPolar lays the ANGLE along rows (y) and the
    # RADIUS along columns (x); dsize (w, h) → h angle bins over 360°, w radius
    # bins over max_radius.
    center = (w / 2.0, h / 2.0)
    max_radius = min(h, w) / 2.0
    flags = cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR
    lp_src = cv2.warpPolar(m_src, (w, h), center, max_radius, flags)
    lp_tgt = cv2.warpPolar(m_tgt, (w, h), center, max_radius, flags)

    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (shift_x, shift_y), response = cv2.phaseCorrelate(lp_tgt, lp_src, win)

    # Decode.  Angle = row shift (h rows ↔ 360°); scale = column shift along the
    # log-radius axis.  Sign convention: the value to feed getRotationMatrix2D on
    # the SOURCE so it aligns to the target (matches register_global; inverse
    # scale follows).
    angle_deg = shift_y * 360.0 / h
    angle_deg = ((angle_deg + 90.0) % 180.0) - 90.0   # fold to [-90, 90)
    log_base = np.exp(np.log(max_radius) / w)
    scale = float(log_base ** shift_x)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    scale = float(np.clip(scale, 1.0 / max_scale_ratio, max_scale_ratio))
    return angle_deg, scale, float(response)


def _phase_translation(src: np.ndarray, tgt: np.ndarray) -> tuple[float, float, float]:
    """Translation (dx, dy) to apply to `src` so it aligns to `tgt`, + peak."""
    h, w = tgt.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    (dx, dy), peak = cv2.phaseCorrelate(src.astype(np.float32),
                                        tgt.astype(np.float32), win)
    return float(dx), float(dy), float(peak)


def _affine_rst(center, angle, scale, dx, dy) -> np.ndarray:
    M = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float64)
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def _compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """2×3 affine for x → outer(inner(x))."""
    O = np.vstack([outer, [0, 0, 1]])
    I = np.vstack([inner, [0, 0, 1]])
    return (O @ I)[:2].astype(np.float64)


def _overlap_corr(src_w: np.ndarray, tgt: np.ndarray) -> float:
    both = (src_w != 0) & (tgt != 0)
    if int(both.sum()) < 50:
        return -1.0
    a, b = src_w[both].astype(np.float64), tgt[both].astype(np.float64)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def fourier_mellin_register(
    source: np.ndarray,
    target: np.ndarray,
    max_scale_ratio: float = 5.0,
    refine_overlap: bool = False,
    n_iter: int = 4,
) -> FourierMellinResult:
    """Estimate the source→target similarity transform via Fourier–Mellin.

    `source` is resized to `target`'s shape first; the recovered scale is the
    source→target factor.  With ``refine_overlap=True`` the estimate is refined
    by iteratively masking both images to their mutual overlap — required when
    the two images only partially overlap (e.g. an STL that covers part of the
    OSM frame, or a cropped tile).
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required for Fourier–Mellin.")

    h, w = target.shape
    src = _to_float(source, (h, w))
    tgt = _to_float(target, (h, w))
    center = (w / 2.0, h / 2.0)

    def _full_estimate(s_img: np.ndarray, t_img: np.ndarray) -> np.ndarray:
        """One FM pass on the given pair → 2×3 transform (with 180° + trans)."""
        ang, sc, resp = _rot_scale_response(s_img, t_img, max_scale_ratio)
        # Resolve the 180° ambiguity, then solve translation, by overlap corr.
        best = None
        for a in (ang, ang + 180.0):
            a_n = ((a + 180.0) % 360.0) - 180.0
            M_rs = _affine_rst(center, a_n, sc, 0.0, 0.0)
            w_rs = cv2.warpAffine(s_img, M_rs, (w, h), flags=cv2.INTER_LINEAR)
            dxc, dyc, _ = _phase_translation(w_rs, t_img)
            M = _affine_rst(center, a_n, sc, dxc, dyc)
            w_full = cv2.warpAffine(s_img, M, (w, h), flags=cv2.INTER_LINEAR)
            c = _overlap_corr(w_full, t_img)
            if best is None or c > best[0]:
                best = (c, M, resp)
        return best[1], best[2]

    M_cum, response = _full_estimate(src, tgt)

    if refine_overlap:
        for _ in range(n_iter):
            src_w = cv2.warpAffine(src, M_cum, (w, h), flags=cv2.INTER_LINEAR)
            mask = (src_w != 0) & (tgt != 0)
            if int(mask.sum()) < max(200, int(0.02 * h * w)):
                break   # overlap too small to refine on
            s_m = np.where(mask, src_w, 0.0).astype(np.float32)
            t_m = np.where(mask, tgt, 0.0).astype(np.float32)
            M_res, _ = _full_estimate(s_m, t_m)
            M_cum = _compose(M_res, M_cum)
            # Converged once the residual is sub-degree / sub-percent.
            ar, sr = _decompose(M_res)
            if abs(((ar + 180) % 360) - 180) < 0.3 and abs(sr - 1.0) < 0.01:
                break

    angle_deg, scale = _decompose(M_cum)
    # Net translation of the image centre under the full transform.
    mapped = M_cum @ np.array([center[0], center[1], 1.0])
    dx, dy = float(mapped[0] - center[0]), float(mapped[1] - center[1])
    src_w = cv2.warpAffine(src, M_cum, (w, h), flags=cv2.INTER_LINEAR)
    ov_corr = _overlap_corr(src_w, tgt)

    logger.info("Fourier–Mellin%s: angle=%.2f° scale=%.3f trans=(%.1f,%.1f) "
                "response=%.3f overlap_corr=%.3f",
                " [overlap-refined]" if refine_overlap else "",
                angle_deg, scale, dx, dy, response, ov_corr)

    return FourierMellinResult(
        angle_deg=float(angle_deg), scale=float(scale), dx=dx, dy=dy,
        confidence=float(response), overlap_corr=float(ov_corr),
        transform=M_cum,
    )


def _decompose(M: np.ndarray) -> tuple[float, float]:
    """(angle_deg, scale) from the linear part of a similarity 2×3 matrix.

    Angle is the rotation that getRotationMatrix2D used to build it (CCW for the
    forward source→target map, consistent with `_affine_rst`).
    """
    a, b = M[0, 0], M[0, 1]
    scale = float(np.hypot(a, b))
    angle = float(np.degrees(np.arctan2(b, a)))   # getRotationMatrix2D: [[c,s],[-s,c]]
    return angle, scale
