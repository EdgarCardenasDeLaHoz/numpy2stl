"""Global FFT cross-correlation registration (register_global).

Part of the align/ subpackage (split from the former align.py).
"""
from __future__ import annotations

import logging
import time

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

# Edge-IoU rotation refinement (the only tuning constants this module needs):
_ROT_IOU_WINDOW = 20.0   # ± window (deg) around the gradient estimate; stays in the 90° quadrant
_ROT_IOU_MARGIN = 0.02   # min edge-IoU gain to override the histogram rotation

from .lines import gradient_angle_histogram, rotation_from_angle_histograms
from .metrics import _dice, _tolerant_iou
from .mask_source import produce_edges
from .segmentation import building_edges

def register_global(
    source: np.ndarray,
    target: np.ndarray,
    scale_prior: float | None = None,
    scale_search: float = 0.35,
    rot_search_deg: float = 45.0,
    trans_frac: float = 0.30,
    cell_size_m: float | None = None,
    forced_rotation: float | None = None,
    source_mask: np.ndarray | None = None,
    free_scale: bool = False,
    source_exclude_mask: np.ndarray | None = None,
) -> dict:
    """
    Global registration via cross-correlation + IoU scoring.

    `source_mask` (optional): a CLEAN, separated building-footprint mask in the
    source grid (e.g. rasterized prism-decomposition polygons).  When given, the
    rotation and scale are derived from these polygon LINES instead of the binary
    heightmap segmentation (which fuses neighbours into blobs): the gradient
    orientation histogram and the footprint edges come from the mask, and scale is
    SWEPT (never locked) and chosen by the footprint edge-IoU peak.  Translation
    still uses the raw heightmap cross-correlation (its base-plate anchor).

    `source_exclude_mask` (optional): cells in `source` (the STL heightmap) to
    treat as non-building regardless of height — vegetation/hillside/water from
    OSM semantic tags, on the SAME grid as `source`. Passed to every
    building_mask/building_edges("stl", ...) call the search makes. Matters a
    lot on hilly cities: a hillside wider than the building-scale top-hat
    kernel can't be removed by the kernel itself, floods the STL mask (measured
    on Salzburg: 66.6% of frame vs 29.1% OSM ground truth), and corrupts BOTH
    the edge-IoU search signal and the L0 height-correlation rotation
    disambiguator (measured: it scored the CORRECT rotation candidate negative
    and a wrong 90°-family candidate positive, because the "overlap" it
    correlated over was mostly hillside, not real buildings).

    For each (scale, rotation) candidate, FFT phase correlation finds the
    optimal translation in O(n log n) — replacing the O(N²) brute-force
    translation grid.  The outer loops sweep scale and rotation only.

    Structure:
      L0 — scale × rotation at 1° step; phase-corr for translation
      L1 — fine rotation ±3° (0.5° step) around L0 best; phase-corr for trans
      L2 — finest rotation ±1° (0.1° step) around L1 best; phase-corr for trans

    When scale_search=0 the scale is locked to scale_prior (the physical
    anchor 1/osm_margin) and only rotation+translation are searched.

    Returns
    -------
    dict: transform (2,3), edge_iou, scale, angle_deg, confidence,
          substep_timings, scale_sweep [(scale, iou), ...]
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required.")

    t0 = time.perf_counter()
    substep_timings: list[tuple[str, float]] = []
    h, w = target.shape
    cx, cy = w / 2.0, h / 2.0

    # Use RAW HEIGHTMAPS for translation cross-correlation.
    # Heightmaps with p10 threshold give optimal registration (correlation 0.4609).
    # Absolute height values are essential for distinguishing buildings from terrain.
    se_f = np.nan_to_num(source, nan=0.0).astype(np.float32)
    te_f = np.nan_to_num(target, nan=0.0).astype(np.float32)
    if se_f.shape != target.shape:
        se_f = cv2.resize(se_f, (w, h), interpolation=cv2.INTER_LINEAR)

    # Zero the excluded (hillside/vegetation/water) cells out of the heightmap that
    # drives the FFT translation cross-correlation.  On a hilly city the hill mass
    # is far larger than any building and, left in, dominates the xcorr — it pulls
    # the translation onto the hill instead of the buildings (measured on Salzburg:
    # at the correct 0° rotation the raw-heightmap xcorr lands the model with only
    # ~8 building px of OSM overlap, so registration collapses even once the rotation
    # is right).  These cells are already excluded from the L0 height-correlation
    # overlap; removing them here makes TRANSLATION building-driven too.  On a flat
    # city the mask is empty/irrelevant, so this is a no-op.  se_f is used only for
    # the xcorr score and the L0 overlap (both of which already ignore these cells) —
    # NOT for any reported heightmap — so the blast radius is limited to translation.
    if source_exclude_mask is not None:
        _excl = np.asarray(source_exclude_mask)
        if _excl.shape != se_f.shape:
            _excl = cv2.resize(_excl.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST)
        se_f[_excl.astype(bool)] = 0.0

    t_masks = time.perf_counter()
    substep_timings.append(("Build heightmaps", t_masks - t0))
    logger.info("  register_global substep: build heightmaps            %.2f s  (src valid=%d  tgt valid=%d)",
                t_masks - t0, int((se_f > 0).sum()), int((te_f > 0).sum()))

    # --- Rotation from the IMAGE GRADIENT (segmentation-independent) ---
    # Wall orientations are read straight off the height field via Sobel — no
    # building mask, so a blobby/poor segmentation can't corrupt rotation (this
    # fixed the 45°/90° aliases on irregular cities).  The STL gradient is taken
    # on the terrain RESIDUAL so hill slopes don't add spurious orientations.
    # cell_size_m sizes the terrain top-hat kernel in real metres (see
    # terrain_residual()) — without it, building_mask/building_edges fall back to
    # a resolution-naive pixel-based kernel that isn't tied to the model's actual
    # scale, which is exactly the physical anchor this function otherwise takes
    # care to establish before running the search.
    te_edges = building_edges(target, source="osm", cell_size_m=cell_size_m).astype(np.float32)   # report only
    if te_edges.shape != target.shape:
        te_edges = cv2.resize(te_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    # Footprint EDGES for the edge-IoU + report.  When a clean separated polygon
    # mask is supplied, take its 1-px outline (no blob-prone segmentation); else
    # use building_edges of the heightmap.
    _use_polys = source_mask is not None
    if _use_polys:
        _sm = (np.asarray(source_mask) > 0).astype(np.uint8)
        if _sm.shape != target.shape:
            _sm = cv2.resize(_sm, (w, h), interpolation=cv2.INTER_NEAREST)
        se_edges = (_sm - cv2.erode(_sm, np.ones((3, 3), np.uint8))).astype(np.float32)
    else:
        # allow_forced_split=False: the search must keep the exact mask
        # structure it was tuned against — forcing a watershed split here
        # (even though it improves report-quality segmentation elsewhere)
        # can shift edge geometry enough to flip the L0 rotation
        # disambiguator's candidate scores. See building_mask()'s
        # allow_forced_split docstring (measured regression on Bilbao).
        # Routed through the mask-producer seam (mask_source.produce_edges) so
        # the segmentation can be swapped without touching the search; with no
        # producer installed this IS building_edges.
        se_edges = produce_edges(source, source="stl", cell_size_m=cell_size_m,
                                   exclude_mask=source_exclude_mask,
                                   allow_forced_split=False).astype(np.float32)
        if se_edges.shape != target.shape:
            se_edges = cv2.resize(se_edges, (w, h), interpolation=cv2.INTER_NEAREST)

    # COARSE rotation: always from the heightmap GRADIENT (Gaussian high-pass).  The
    # polygon-mask gradient is NOT used here — regularized footprints over-represent
    # one axis and give a spurious dominant orientation (e.g. +26.6° on Philadelphia
    # vs the true ~−9°).  The heightmap gradient gives a stable base; the edge-IoU
    # refinement below (on se_edges = clean polygon edges when available) then finds
    # the true angle.
    _hp_sigma = max(15.0, min(h, w) / 20.0)
    _src_hp = np.nan_to_num(source.astype(np.float32))
    _src_hp = _src_hp - cv2.GaussianBlur(_src_hp, (0, 0), _hp_sigma)
    hist_src = gradient_angle_histogram(_src_hp)
    hist_tgt = gradient_angle_histogram(np.nan_to_num(target))
    hist_rot_deg, hist_xcorr = rotation_from_angle_histograms(hist_src, hist_tgt)
    # Second pass at higher ANGULAR resolution to fine-tune the rotation: the
    # coarse histogram is 1°/bin, so rebuild at 0.1°/bin and re-correlate.  The
    # finer peak (with sub-bin parabolic interpolation → ~0.01°) replaces the
    # coarse value only when it AGREES with it (within 2°), guarding against a
    # noisy fine-bin alias on weak/non-grid scenes.
    try:
        _nb_fine = 1800   # 0.1° per bin
        _hs_f = gradient_angle_histogram(_src_hp, n_bins=_nb_fine)
        _ht_f = gradient_angle_histogram(np.nan_to_num(target), n_bins=_nb_fine)
        _rot_fine, _ = rotation_from_angle_histograms(_hs_f, _ht_f)
        _d = ((_rot_fine - hist_rot_deg + 90.0) % 180.0) - 90.0
        if abs(_d) <= 2.0:
            logger.info("  register_global: rotation fine-tuned %.2f° → %.2f° "
                        "(0.1° histogram, Δ%.2f°)", hist_rot_deg, _rot_fine, _d)
            hist_rot_deg = float(_rot_fine)
        else:
            logger.info("  register_global: fine rotation %.2f° rejected (%.2f° from "
                        "coarse %.2f°); keeping coarse", _rot_fine, _d, hist_rot_deg)
    except Exception as _exc:
        logger.debug("rotation fine pass failed (%s); keeping coarse", _exc)
    # Confidence: the line-angle histogram only gives a reliable rotation when the
    # scene has a DOMINANT orientation (a street grid).  Measure the cross-
    # correlation peak prominence; if low (flat histogram — random/non-grid scene)
    # we fall back to the xcorr-refined rotation later instead of trusting it.
    _hx = np.asarray(hist_xcorr, dtype=np.float64)
    hist_prominence = float((_hx.max() - _hx.mean()) / (_hx.std() + 1e-9))
    hist_confident = hist_prominence >= 3.0
    logger.info("  register_global: histogram rotation estimate = %.1f° "
                "(prominence %.1f, %s)", hist_rot_deg, hist_prominence,
                "confident" if hist_confident else "LOW — will use xcorr")

    # Because histograms are 90°-periodic for rectangular grids (orthogonal walls
    # produce two peaks 90° apart) there can be a 90° ambiguity.  Evaluate both
    # candidates ±0° and ±90° and pick the one with the better IoU later in L0.
    hist_rot_candidates = [hist_rot_deg, hist_rot_deg + 90.0, hist_rot_deg - 90.0]

    t_hist = time.perf_counter()
    substep_timings.append(("Histogram rotation estimate", t_hist - t_masks))

    if se_f.sum() == 0 or te_f.sum() == 0:
        logger.warning("register_global: empty edge mask (src=%d tgt=%d px); returning identity.",
                       int(se_f.sum()), int(te_f.sum()))
        return {
            "transform": np.array([[1.0, 0, 0], [0, 1.0, 0]]),
            "edge_iou": 0.0, "scale": 1.0, "angle_deg": 0.0,
            "confidence": 0.0, "n_iterations": 0, "converged": False,
            "substep_timings": substep_timings, "scale_sweep": [],
        }

    s0 = float(scale_prior) if scale_prior else 1.0

    # Precompute FFT of the target edge mask once (reused for every candidate).
    # Non-normalized cross-correlation: xcorr = IFFT(conj(F_src) × F_tgt).
    # This gives peak at (dx, dy) = shift to apply to src to maximise raw
    # overlap count with tgt.  Unlike phaseCorrelate, no division by magnitude
    # — much more reliable for sparse binary masks where magnitude-normalisation
    # amplifies noise.
    # FFT of the filled OSM mask (used for translation cross-correlation).
    _F_te = np.fft.rfft2(te_f.astype(np.float64))

    def _warp_f(sc: float, rot: float) -> np.ndarray:
        """Warp STL heightmap by scale+rotation."""
        M = cv2.getRotationMatrix2D((cx, cy), rot, sc).astype(np.float32)
        return cv2.warpAffine(se_f, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _warp_edges(sc: float, rot: float) -> np.ndarray:
        """Warp STL edges by scale+rotation (for final IoU reporting)."""
        M = cv2.getRotationMatrix2D((cx, cy), rot, sc).astype(np.float32)
        return cv2.warpAffine(se_edges, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    _last_xcorr: list = [None]   # mutable container so inner funcs can write it

    # Energy of the target mask (denominator for normalisation, constant).
    _energy_te = float((te_f.astype(np.float64) ** 2).sum()) + 1e-9

    # Physical prior: models are centered ~on the same point (City Hall).
    # Prefer small translations with soft Gaussian penalty.
    _prior_sigma = min(h, w) / 4.0   # σ ≈ 128px; 2σ ≈ 256px (half frame)
    _ty_p, _tx_p = np.ogrid[:h, :w]
    _dy_dist = np.minimum(_ty_p, h - _ty_p).astype(np.float64)
    _dx_dist = np.minimum(_tx_p, w - _tx_p).astype(np.float64)
    # Gaussian prior on translation: exp(-(dx² + dy²) / 2σ²)
    _prior = np.exp(-(_dx_dist ** 2 + _dy_dist ** 2) / (2.0 * _prior_sigma ** 2))

    def _xcorr_best(warped: np.ndarray) -> tuple[float, float, float]:
        """
        Find globally optimal translation via normalized FFT cross-correlation
        with a soft Gaussian prior favoring translations near (0, 0).

        The Gaussian prior breaks the periodic ambiguity: when multiple peaks
        have similar xcorr heights (from city block spacing), the prior selects
        the one closest to the center — correct because STL and OSM are both
        centered on City Hall.

        Returns (dx, dy, score) where score is xcorr × prior.
        """
        w_f = warped.astype(np.float64)
        F_w = np.fft.rfft2(w_f)
        xcorr = np.fft.irfft2(np.conj(F_w) * _F_te, s=(h, w))
        energy_w = float((w_f ** 2).sum()) + 1e-9
        xcorr_norm = xcorr / np.sqrt(energy_w * _energy_te)

        # Find raw peak (no prior)
        peak_raw = np.unravel_index(np.argmax(xcorr_norm), xcorr_norm.shape)
        ry_raw, rx_raw = int(peak_raw[0]), int(peak_raw[1])
        dy_raw = ry_raw if ry_raw <= h // 2 else ry_raw - h
        dx_raw = rx_raw if rx_raw <= w // 2 else rx_raw - w
        raw_max = float(xcorr_norm[ry_raw, rx_raw])

        # Pure xcorr signal — no prior weighting.
        # Heightmaps have unique signatures per building, no periodic ambiguity.
        score_map = xcorr_norm
        _last_xcorr[0] = score_map
        peak = np.unravel_index(np.argmax(score_map), score_map.shape)
        ry, rx = int(peak[0]), int(peak[1])
        dy = ry if ry <= h // 2 else ry - h
        dx = rx if rx <= w // 2 else rx - w

        if abs(dy_raw) < 10 and abs(dx_raw) < 10:  # Log if raw peak near center
            logger.debug("  xcorr: raw_max=%.4f at (%d,%d)  prior_adjusted=%.4f at (%d,%d)",
                        raw_max, int(dx_raw), int(dy_raw), score_map[ry, rx], int(dx), int(dy))

        return float(dx), float(dy), float(score_map[ry, rx])

    # Scale candidates: locked to s0 when scale_search=0, swept otherwise.
    if scale_search <= 0.0:
        scales = np.array([s0])
    else:
        s_lo = max(0.4, s0 - scale_search)
        scales = np.arange(s_lo, s0 + scale_search + 1e-9, 0.05)

    # --- L0: resolve orientation by per-pixel HEIGHT correlation, biased to 0° ---
    # The gradient base θ fixes the grid mod 90°; the true rotation is one of the
    # 4 grid orientations {θ, θ+90, θ+180, θ+270}.  Footprint/xcorr overlap is
    # 90°-ambiguous (it flipped Boston), so each candidate is scored by the PEARSON
    # correlation of the warped STL heights vs OSM heights over the building
    # overlap — the correct orientation aligns tall-with-tall.  A 0° BIAS breaks
    # near-ties toward the smallest |rotation| (cities are usually ~north-aligned),
    # which recovers Boston; a genuinely-rotated grid (Denver) wins on height
    # correlation despite the bias.  A manual `forced_rotation` overrides all.
    def _norm180(a):
        return ((a + 180.0) % 360.0) - 180.0

    _se_excl_f = None
    if source_exclude_mask is not None and source_exclude_mask.shape == source.shape:
        _se_excl_f = source_exclude_mask.astype(np.float32)
        if _se_excl_f.shape != target.shape:
            _se_excl_f = cv2.resize(_se_excl_f, (w, h), interpolation=cv2.INTER_NEAREST)

    def _height_corr_at(sc, rot):
        dxv, dyv, _ = _xcorr_best(_warp_f(float(sc), float(rot)))
        Mr = cv2.getRotationMatrix2D((cx, cy), float(rot), float(sc)).astype(np.float32)
        Mr[0, 2] += dxv
        Mr[1, 2] += dyv
        wse = cv2.warpAffine(se_f, Mr, (w, h), flags=cv2.INTER_LINEAR)
        both = (wse > 0) & (te_f > 0)
        # Exclude hillside/vegetation/water cells from the correlation overlap —
        # `wse > 0` alone can't tell a hill from a building (both are "positive
        # elevation"), so without this a wide hillside dominates the "overlap"
        # and the correlation measures hill-vs-OSM-building noise, not real
        # structure agreement. Warped with nearest-neighbour + >0.5 threshold
        # to stay a clean boolean mask through the same rotation as se_f.
        if _se_excl_f is not None:
            w_excl = cv2.warpAffine(_se_excl_f, Mr, (w, h), flags=cv2.INTER_NEAREST) > 0.5
            both = both & ~w_excl
        if int(both.sum()) < 50:
            return -1.0, dxv, dyv
        a = wse[both].astype(np.float64)
        b = te_f[both].astype(np.float64)
        if a.std() < 1e-6 or b.std() < 1e-6:
            return -1.0, dxv, dyv
        return float(np.corrcoef(a, b)[0, 1]), dxv, dyv

    base_rot = _norm180(float(hist_rot_candidates[0]))
    sc0 = float(scales[len(scales) // 2]) if len(scales) > 1 else s0
    if forced_rotation is not None:
        cand_rots = [_norm180(float(forced_rotation))]
    else:
        cand_rots = [_norm180(base_rot + k * 90.0) for k in (0, 1, 2, 3)]

    scale_best_sc: dict[float, float] = {round(sc0, 4): 0.0}
    rot_best_sc:   dict[float, float] = {}
    scored = []
    for rc in cand_rots:
        hc, dxv, dyv = _height_corr_at(sc0, rc)
        scored.append((rc, hc, dxv, dyv))
        rot_best_sc[round(rc, 1)] = hc

    # 0° BIAS (dominant): default to the candidate closest to 0°.  The per-pixel
    # height correlation is only a weak disambiguator (raw terrain/base-plate
    # correlates spuriously and can favour a 90° flip), so it overrides the bias
    # ONLY when a more-rotated candidate beats the near-0 one by a large margin
    # (clear evidence of a genuinely rotated grid, e.g. Denver) — otherwise use a
    # manual `forced_rotation`.
    _STRONG_MARGIN = 0.15
    # Absolute floor: the *relative* margin above alone is fragile when near0's
    # own score is weak or negative (no real correlation signal either way) —
    # a barely-positive candidate can then "beat near0 by 0.15" trivially and
    # win purely on being less-bad, not on genuine structural agreement.
    # Measured on Bilbao: candidates scored {0.2°: -0.095, 90.2°: -0.03,
    # -179.8°: 0.13-0.16 (varies run to run), -89.8°: 0.055-0.093} — none is a
    # real correlation, but -179.8° cleared a 0.15 floor often enough to still
    # flip a correct ~0° histogram answer to a wrong 180° (0.15 sat INSIDE
    # Bilbao's noise band, not above it). Calibrated against real L0 scores
    # across 8 cities: every city's noise-band candidates (no genuine grid
    # match) topped out around 0.06-0.16 (Bilbao 0.16, Salzburg 0.067, Lisbon
    # 0.066, Valencia 0.036, Prague 0.011), while a genuine match (Miami, a
    # real ~0°-aligned grid) scored 0.517 -- an order of magnitude clear of
    # the noise band. 0.20 sits above every measured noise-band score with
    # margin and well below the one measured genuine signal.
    _MIN_ABS_CORR = 0.20
    near0 = min(scored, key=lambda s: abs(s[0]))          # closest to 0°

    # Diagnostic-only footprint edge-IoU per candidate (surfaced in the L0 log).
    # NOTE: deliberately NOT used to pick the quadrant.  On a near-square street
    # grid the footprints self-overlap after a 90°/180° flip about as well as (or
    # BETTER than) at 0° — measured Barcelona: 90.4° edge-IoU 0.276 > 0° 0.193;
    # Valencia: −90° 0.157 > 0° 0.067 — so edge-IoU is fooled by the same grid
    # self-alignment that fools height correlation.  It is logged only to show how
    # ambiguous the overlap signal is on these grids.
    def _edge_iou_at(rot, dxv, dyv):
        Mr = cv2.getRotationMatrix2D((cx, cy), float(rot), float(sc0)).astype(np.float32)
        Mr[0, 2] += dxv
        Mr[1, 2] += dyv
        we = cv2.warpAffine((se_edges > 0).astype(np.uint8), Mr, (w, h),
                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0).astype(bool)
        return float(_tolerant_iou(we, te_edges.astype(bool), tol_px=2))
    _eiou = {round(s[0], 1): _edge_iou_at(s[0], s[2], s[3]) for s in scored}

    # Sentinel guard: _height_corr_at returns exactly -1.0 when it could NOT measure
    # a correlation at a candidate (its xcorr translation landed <50 px of building
    # overlap, or a degenerate constant overlap).  That is "unmeasurable", not "a real
    # low correlation".  On hill-dominated cities the raw-heightmap xcorr translation
    # fails at the UN-rotated pose specifically — the hillside outweighs the buildings
    # and pulls the translation off, so near0 comes back as this sentinel (measured on
    # Salzburg: 0° → 8 px overlap → -1.0, while a wrong 90° flip found a large
    # hillside-vs-OSM overlap scoring a spurious +0.254 and won the override).  When
    # near0 is a sentinel there is no valid baseline for a rotated candidate to "beat",
    # so the override has no evidence behind it — fall back to the 0° bias.  The
    # histogram itself already fixes the grid mod 90°; keeping near0 respects it.
    _SENTINEL = -1.0
    near0_unmeasured = near0[1] <= _SENTINEL + 1e-9
    strong = [s for s in scored
              if s[1] > near0[1] + _STRONG_MARGIN and s[1] >= _MIN_ABS_CORR]
    if near0_unmeasured:
        strong = []   # no valid near-0 baseline → do not let a spurious flip override
    # DOMINANT 0° prior on a confident histogram (STRENGTHENED): when the line-angle
    # histogram is confident, it has already pinned the grid orientation to ~0° mod
    # 90° (both STL and OSM are rendered north-up, so the true STL→OSM rotation is
    # intrinsically ≈0° for every city in this set).  The remaining {θ,θ+90,θ+180,
    # θ+270} choice is then purely which grid quadrant — and on a near-square grid
    # EVERY overlap-based tiebreak (height correlation AND footprint edge-IoU) is
    # fooled, because the grid self-aligns building-on-building at the flip and
    # scores a spuriously high "overlap" there (Valencia −90° height-corr 0.376 vs
    # 0° −0.037; that spurious 0.054 ov_iou match is a FALSE match, not a real one).
    # So when the histogram is confident, trust its 0°-resolved quadrant and keep
    # near0 — do NOT let the unreliable height-corr flip it away from ~0°.  (The
    # override still runs for LOW-confidence / non-grid scenes, where near0 is not
    # already histogram-anchored and the height-corr tiebreak is the only signal.)
    if hist_confident:
        strong = []
    chosen_s = max(strong, key=lambda s: s[1]) if strong else near0
    rc, hc, dx_b, dy_b = chosen_s
    best = (sc0, float(rc), dx_b, dy_b, hc)

    t_l0 = time.perf_counter()
    substep_timings.append(("L0: orientation (height-corr + 0° bias)", t_l0 - t_hist))
    logger.info(
        "  register_global substep: L0 orientation  base=%.1f°  candidates=%s  "
        "edge_iou=%s  chosen=%.1f° (height-corr=%.3f, 0°-biased)  %.2f s",
        base_rot, {round(s[0], 1): round(s[1], 3) for s in scored},
        {k: round(v, 3) for k, v in _eiou.items()},
        best[1], best[4], t_l0 - t_hist,
    )
    # Rotation comes from the translation-invariant LINE-ANGLE HISTOGRAM; L0 just
    # resolved its 90° ambiguity.  Capture it now — the L1/L2 xcorr sweeps below
    # are computed for the report plot only and must NOT move the rotation
    # (xcorr/Dice rotation refinement depends on correct translation+scale, which
    # is exactly the contamination we are avoiding).
    hist_rot_resolved = float(best[1])

    # --- L1: fine rotation ±1°, 0.25° step — xcorr score only ---
    bs, br0 = best[0], best[1]
    for rot in np.arange(br0 - 1.0, br0 + 1.0 + 1e-9, 0.25):
        warped = _warp_f(bs, float(rot))
        dx, dy, v = _xcorr_best(warped)
        if v > best[4] + 1e-5:
            best = (bs, float(rot), dx, dy, v)
        rot_best_sc[round(float(rot), 2)] = v
    t_l1 = time.perf_counter()
    substep_timings.append(("L1 rot ±1° (xcorr)", t_l1 - t_l0))
    logger.info("  register_global substep: L1 rot refine  %.2f s  (xcorr=%.4f)",
                t_l1 - t_l0, best[4])
    rot_l1_sc = dict(rot_best_sc)   # snapshot L0+L1 xcorr for the report

    # --- L2: finest rotation ±0.25°, 0.05° step ---
    bs = best[0]
    rot_l2_sc: dict[float, float] = {}
    for rot in np.arange(best[1] - 0.25, best[1] + 0.25 + 1e-9, 0.05):
        warped = _warp_f(bs, float(rot))
        dx, dy, v = _xcorr_best(warped)
        rot_l2_sc[round(float(rot), 2)] = v   # capture for report
        if v > best[4] + 1e-5:
            best = (bs, float(rot), dx, dy, v)
    t_l2 = time.perf_counter()
    substep_timings.append(("L2 rot ±0.25° (xcorr)", t_l2 - t_l1))
    logger.info("  register_global substep: L2 rot refine  %.2f s  (xcorr=%.4f)",
                t_l2 - t_l1, best[4])

    # Use the line-angle histogram rotation when it is confident (grid scene);
    # otherwise keep the L1/L2 xcorr-refined rotation (the histogram is unreliable
    # on scenes without a dominant orientation).
    if hist_confident:
        best = (best[0], hist_rot_resolved, best[2], best[3], best[4])
    else:
        hist_rot_resolved = float(best[1])   # fall back to xcorr-refined rotation

    # Post-hoc sweeps — score three metrics per pose so the report can compare
    # how each behaves vs scale / rotation:
    #   • Dice    — building-EDGE footprint Dice (the selection metric; filled
    #               masks overlap at any pose, so edges are used for a real peak)
    #   • IoU     — edge-mask intersection-over-union (tolerant)
    #   • xcorr   — normalized FFT cross-correlation score of the raw heightmaps
    # For each scale/rotation we re-solve translation by xcorr, then evaluate all
    # three at that pose.  _metrics_at returns (dice, iou, xcorr, dx, dy).
    bs, br, bdx, bdy, _ = best

    se_edges_u8 = (se_edges > 0).astype(np.uint8)
    te_edges_bool = te_edges.astype(bool)
    # STL coverage mask (where the model has data) — used to crop the OSM to the
    # STL's footprint so Dice/IoU don't penalize the STL for OSM buildings that
    # lie outside the model's extent (the STL covers only part of the OSM frame).
    se_cov_u8 = (~np.isnan(source)).astype(np.uint8)
    if se_cov_u8.shape != target.shape:
        se_cov_u8 = cv2.resize(se_cov_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    def _bbox_of(mask: np.ndarray):
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return (slice(int(ys.min()), int(ys.max()) + 1),
                slice(int(xs.min()), int(xs.max()) + 1))

    _prof = {"xcorr": 0.0, "metric": 0.0}   # accumulated cost across all calls

    def _metrics_at(sc_v: float, rot_v: float, crop: bool = True
                    ) -> tuple[float, float, float, float, float]:
        """Returns (dice, iou, xcorr, dx, dy) at the given pose.

        crop=True (reporting/plots): Dice/IoU are computed only within the STL's
        warped footprint, so the STL is not penalized for OSM buildings outside
        its extent.  crop=False (rotation selection): full-frame, because the
        cropped Dice saturates on the dense grid and loses rotation discrimination.
        """
        _t0 = time.perf_counter()
        dx_v, dy_v, xc = _xcorr_best(_warp_f(sc_v, rot_v))   # FFT translation solve
        _t1 = time.perf_counter()
        _prof["xcorr"] += _t1 - _t0
        Mr = cv2.getRotationMatrix2D((cx, cy), rot_v, sc_v).astype(np.float32)
        Mr[0, 2] += dx_v
        Mr[1, 2] += dy_v
        we = cv2.warpAffine(se_edges_u8, Mr, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if crop:
            wcov = cv2.warpAffine(se_cov_u8, Mr, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            bb = _bbox_of(wcov)
            if bb is None:
                _prof["metric"] += time.perf_counter() - _t1
                return 0.0, 0.0, float(xc), dx_v, dy_v
            we_e, te_e = we[bb], te_edges_bool[bb]
        else:
            we_e, te_e = we, te_edges_bool
        out = (_dice(we_e, te_e, tol_px=2),
               _tolerant_iou(we_e, te_e, tol_px=2),
               float(xc), dx_v, dy_v)
        _prof["metric"] += time.perf_counter() - _t1
        return out

    # sweeps map x -> (dice, iou, xcorr).  Timed separately so the cost is visible:
    #  - scale sweep feeds the scale determinism plot AND the xcorr-peak scale pick
    #  - rotation sweep is PLOT-ONLY (rotation now comes from the histogram), so it
    #    is pure diagnostic cost and can be cheapened by a coarser step if needed.
    t_sw0 = time.perf_counter()
    scale_metrics: dict[float, tuple[float, float, float]] = {}
    viz_lo = max(0.25, s0 - 0.45)
    viz_hi = s0 + 0.45
    for sc_v in np.arange(viz_lo, viz_hi + 1e-9, 0.025):
        d, i, x, _, _ = _metrics_at(float(sc_v), br)
        scale_metrics[round(float(sc_v), 3)] = (d, i, x)
    t_sw1 = time.perf_counter()
    substep_timings.append((f"↳ scale sweep ({len(scale_metrics)} pts)", t_sw1 - t_sw0))

    # NOTE: the full ±45° rotation metric sweep was REMOVED.  Rotation now comes
    # from the translation-invariant line-angle histogram (the angle-histogram
    # figure is its diagnostic); a Dice/IoU-vs-rotation sweep showed metrics that
    # don't drive the decision and cost ~1.7 s of plot-only FFT solves.
    rot_metrics: dict[float, tuple[float, float, float]] = {}

    t_sweep = time.perf_counter()
    substep_timings.append(("Metric sweeps (scale Dice/IoU/xcorr)", t_sweep - t_l2))

    sc, rot, dx, dy, xcorr_score = best

    # Scale: when it is LOCKED (scale_search<=0, i.e. a physical/geometric anchor
    # was supplied) keep that exact value — the anchor is geometrically determined
    # (OSM fetched at osm_margin× the STL footprint), so re-picking by xcorr would
    # only drift off it.  Otherwise (no anchor) pick the PEAK-XCORR scale: the
    # scale maximizing the raw-heightmap cross-correlation at the best rotation,
    # refined ±0.05 — xcorr is segmentation-independent, so it is the most
    # self-consistent signal when nothing else fixes scale.
    if _use_polys and free_scale and scale_metrics:
        # OPT-IN un-lock (free_scale): refine scale within a TIGHT ±10% window of the
        # geometric anchor by polygon edge-IoU, falling back to the anchor when the
        # peak is at the window boundary.  NOTE: the scale is geometrically determined
        # (OSM fetched at osm_margin× the STL footprint, both at one resolution →
        # scale = 1/osm_margin for every object), so this can only drift off the
        # correct value — kept only for objects where that geometric assumption is
        # violated.  Default OFF (the anchor wins on every tested model).
        _lo, _hi = 0.90 * sc, 1.10 * sc
        cand = {k: v for k, (d, v, x) in scale_metrics.items() if _lo - 1e-9 <= k <= _hi + 1e-9}
        if len(cand) >= 3:
            sc_peak = max(cand, key=cand.get)
            ks = sorted(cand)
            if sc_peak in (ks[0], ks[-1]):
                logger.info("  scale: free-scale edge-IoU monotonic in ±10%% → keeping anchor %.4f", sc)
            else:
                logger.info("  scale (free) refined by polygon edge-IoU: %.4f → %.4f", sc, sc_peak)
                sc = float(sc_peak)
    elif _use_polys:
        logger.info("  scale = geometric anchor %.4f (1/osm_margin; polygon lines drive "
                    "ROTATION only — scale is geometric, free_scale=off)", sc)
    elif scale_search <= 0.0 and free_scale and scale_metrics:
        # OPT-IN small nudge (free_scale, non-polygon path): the geometric anchor
        # assumes the OSM fetch bbox landed at EXACTLY osm_margin x the STL
        # footprint, but that's only as good as the upstream footprint-size
        # estimate (e.g. a commercial STL pack's rounded "~2km" size tier) — a
        # small, real mismatch there shows up as a small, consistent scale bias.
        # Deliberately narrower than the polygon path's +-10% window: a prior
        # attempt to trust the raw area-ratio/Fourier estimate directly caused
        # wild drift on some cities (a dense grid pushed it to 1.25x/1.67x — see
        # pipeline.py's estimate_scale() comment) — the failure mode this must
        # not reintroduce is "confidently wrong by a lot", not "slightly right".
        # So: only nudge within +-5% of the anchor (a mismeasured footprint size
        # is a few-percent error, not tens of percent), and only when the Dice
        # peak clears the same sharpness bar as the unlocked-scale path uses.
        _lo, _hi = 0.95 * sc, 1.05 * sc
        dice_by_scale = {k: v[0] for k, v in scale_metrics.items()}
        cand = {k: v for k, v in dice_by_scale.items() if _lo - 1e-9 <= k <= _hi + 1e-9}
        if len(cand) >= 3:
            dice_vals = sorted(dice_by_scale.values())
            dice_median = dice_vals[len(dice_vals) // 2]
            sc_peak = max(cand, key=cand.get)
            peak_val = cand[sc_peak]
            ks = sorted(cand)
            at_window_edge = sc_peak in (ks[0], ks[-1])
            _FREE_ANCHOR_MARGIN = 0.10
            if not at_window_edge and (peak_val - dice_median) >= _FREE_ANCHOR_MARGIN:
                logger.info("  scale: anchor %.4f nudged to sharp Dice peak %.4f "
                            "(within +-5%%, free_scale)", sc, sc_peak)
                sc = float(sc_peak)
            else:
                logger.info("  scale LOCKED to anchor %.4f (free_scale on, but no sharp "
                            "peak within +-5%%)", sc)
        else:
            logger.info("  scale LOCKED to anchor %.4f (free_scale on, too few sweep "
                        "points in +-5%% window)", sc)
    elif scale_search <= 0.0:
        logger.info("  scale LOCKED to anchor %.4f (no xcorr re-pick)", sc)
    elif scale_metrics:
        # Prefer the Dice/edge-IoU peak over the raw-heightmap xcorr peak when
        # Dice shows a real, sharp peak: xcorr correlates absolute height
        # magnitude, which is dominated by a handful of tall buildings and can
        # be nearly flat/noisy across scale (no structural signal), while Dice
        # measures actual footprint-shape agreement across the whole frame and
        # empirically has a much sharper, more reliable peak (Miami: Dice peak
        # at 0.75-0.85x vs. xcorr peak pinned to the search-range boundary at
        # 1.5x — a strong sign xcorr's "peak" is just the search-window edge,
        # not a real optimum). "Sharp" = peak clears the sweep's median by a
        # solid margin, so a flat/ambiguous Dice curve still falls back to xcorr.
        dice_by_scale = {k: v[0] for k, v in scale_metrics.items()}
        dice_vals = sorted(dice_by_scale.values())
        dice_median = dice_vals[len(dice_vals) // 2]
        sc_dice_peak = max(dice_by_scale, key=dice_by_scale.get)
        dice_peak_val = dice_by_scale[sc_dice_peak]
        scales_sorted = sorted(scale_metrics)
        at_boundary = sc_dice_peak in (scales_sorted[0], scales_sorted[-1])
        _DICE_PEAK_MARGIN = 0.10  # min (peak - median) to trust the Dice peak over xcorr

        if not at_boundary and (dice_peak_val - dice_median) >= _DICE_PEAK_MARGIN:
            sc_coarse = sc_dice_peak
            pick_metric = "dice"
        else:
            sc_coarse = max(scale_metrics, key=lambda k: scale_metrics[k][2])  # [2] = xcorr
            pick_metric = "xcorr"

        sfine = {}
        for sv in np.arange(sc_coarse - 0.05, sc_coarse + 0.05 + 1e-9, 0.0125):
            if sv > 0.1:
                d, i, x, _, _ = _metrics_at(float(sv), br)
                sfine[round(float(sv), 4)] = d if pick_metric == "dice" else x
        sc_peak = max(sfine, key=sfine.get) if sfine else sc_coarse
        logger.info("  scale by peak %s: %.4f (dice-peak=%.4f @ %.4f, xcorr-peak-scale=%.4f) [global %.4f]",
                    pick_metric, sc_peak, dice_peak_val, sc_dice_peak,
                    max(scale_metrics, key=lambda k: scale_metrics[k][2]), sc)
        sc = float(sc_peak)

    # ROTATION: start from the gradient-histogram estimate, then refine it by
    # maximizing footprint EDGE-IoU.  The histogram gradient can lock onto roof /
    # clutter orientation and miss the true building-grid rotation by ~10° (e.g.
    # Philadelphia, whose grid is tilted ~12° from north — the histogram reported
    # ~0°).  Now that scale is fixed and translation is re-solved per angle, the
    # earlier contamination concern is gone, so we sweep rotation within
    # ±_ROT_IOU_WINDOW of the histogram estimate (staying inside its 90° quadrant
    # to avoid alias flips), and adopt the edge-IoU peak ONLY if it beats the
    # histogram pose by a clear margin (guards irregular cities like Boston where
    # the 0°-biased histogram is already right).
    #
    # The fixed _ROT_IOU_MARGIN alone is not enough: on dense grid cities (e.g.
    # Miami — near-uniform diagonal blocks over the whole frame), full-frame
    # edge-IoU vs. rotation is multi-modal/noisy (several comparably-tall local
    # peaks a few degrees apart, not one clean maximum) — see rot_sweep.png.
    # A peak that clears the *margin* by chance is common in that noise; a peak
    # that also clears the *sweep's own median* by a solid amount is a much
    # stronger signal that it reflects real structure, not noise. Require both.
    rot = float(hist_rot_resolved)
    rot_iou_sweep: dict[float, float] = {}
    _iou_base = _metrics_at(sc, rot, crop=False)[1]
    for _rr in np.arange(rot - _ROT_IOU_WINDOW, rot + _ROT_IOU_WINDOW + 1e-9, 1.0):
        rot_iou_sweep[round(float(_rr), 2)] = _metrics_at(sc, float(_rr), crop=False)[1]
    _rr_best = max(rot_iou_sweep, key=rot_iou_sweep.get)
    _sweep_vals = sorted(rot_iou_sweep.values())
    _sweep_median = _sweep_vals[len(_sweep_vals) // 2]
    _ROT_PEAK_MARGIN = 0.10  # min (peak - sweep median) to trust the IoU peak over the histogram
    _is_sharp_peak = (rot_iou_sweep[_rr_best] - _sweep_median) >= _ROT_PEAK_MARGIN
    if (rot_iou_sweep[_rr_best] > _iou_base + _ROT_IOU_MARGIN and abs(_rr_best - rot) > 1e-6
            and _is_sharp_peak):
        # fine refine ±1° at 0.25° around the IoU peak
        _fine = {}
        for _rr in np.arange(_rr_best - 1.0, _rr_best + 1.0 + 1e-9, 0.25):
            _fine[round(float(_rr), 2)] = _metrics_at(sc, float(_rr), crop=False)[1]
            rot_iou_sweep[round(float(_rr), 2)] = _fine[round(float(_rr), 2)]
        _rr_fine = max(_fine, key=_fine.get)
        logger.info("  rotation refined by edge-IoU: %.2f° → %.2f° "
                    "(IoU %.3f → %.3f, +%.3f over histogram, peak-vs-median margin %.3f)", rot, _rr_fine,
                    _iou_base, _fine[_rr_fine], _fine[_rr_fine] - _iou_base,
                    rot_iou_sweep[_rr_best] - _sweep_median)
        rot = float(_rr_fine)
    elif rot_iou_sweep[_rr_best] > _iou_base + _ROT_IOU_MARGIN and abs(_rr_best - rot) > 1e-6:
        logger.info("  rotation kept at histogram estimate %.2f° "
                    "(edge-IoU peak %.2f° gain %.3f > margin, but sweep is noisy/multi-modal: "
                    "peak-vs-median %.3f < %.3f)",
                    rot, _rr_best, rot_iou_sweep[_rr_best] - _iou_base,
                    rot_iou_sweep[_rr_best] - _sweep_median, _ROT_PEAK_MARGIN)
    else:
        logger.info("  rotation kept at histogram estimate %.2f° "
                    "(edge-IoU peak %.2f° gain %.3f ≤ margin %.3f)",
                    rot, _rr_best, rot_iou_sweep[_rr_best] - _iou_base, _ROT_IOU_MARGIN)
    dice_fine_sc: dict[float, float] = {}
    dx, dy, _ = _xcorr_best(_warp_f(sc, rot))
    # Expose the edge-IoU rotation sweep for the report (decomposed convention).
    rot_metrics = {-a: (0.0, v, 0.0) for a, v in rot_iou_sweep.items()}

    t_sel = time.perf_counter()
    substep_timings.append(("Scale pick + translation solve", t_sel - t_sweep))
    # Where the sweep time actually goes (accumulated across every _metrics_at):
    substep_timings.append(("↳↳ FFT translation solves (in sweeps)", _prof["xcorr"]))
    substep_timings.append(("↳↳ Dice/IoU eval (in sweeps)", _prof["metric"]))

    # Final xcorr map at best rotation (for visualization).
    _xcorr_best(_warp_f(sc, rot))
    xcorr_map = _last_xcorr[0]

    # Edge IoU for reporting (computed on edges, not heightmaps).
    te_edges_bool = te_edges.astype(bool)
    T_final = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    warped_edges_final = cv2.warpAffine(
        (_warp_edges(sc, rot) > 0).astype(np.uint8), T_final, (w, h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    # Crop to the STL footprint so the reported IoU isn't penalized for OSM
    # buildings outside the model's extent.
    _wcov_final = cv2.warpAffine(se_cov_u8, cv2.getRotationMatrix2D((cx, cy), rot, sc).astype(np.float32),
                                 (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    _T = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    _wcov_final = cv2.warpAffine(_wcov_final, _T, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    _bbf = _bbox_of(_wcov_final)
    if _bbf is not None:
        edge_iou = _tolerant_iou(warped_edges_final[_bbf], te_edges_bool[_bbf], tol_px=2)
    else:
        edge_iou = _tolerant_iou(warped_edges_final, te_edges_bool, tol_px=2)

    M = cv2.getRotationMatrix2D((cx, cy), rot, sc).astype(np.float64)
    M[0, 2] += dx
    M[1, 2] += dy
    logger.info(
        "Global xcorr: scale=%.3f rot=%.1f shift=(%.1f,%.1f)  "
        "xcorr=%.4f  edge_iou=%.3f (report only)  total=%.2f s",
        sc, rot, dx, dy, xcorr_score, edge_iou, time.perf_counter() - t0,
    )

    # Sweeps carry (x, dice, iou, xcorr) so the report can draw all three metrics.
    scale_sweep = sorted((s, d, i, x) for s, (d, i, x) in scale_metrics.items())
    # Return the rotation sweep in the REPORTED (decomposed) convention so the
    # plot x-axis matches registration.angle_deg.  The internal sweep angles are
    # cv2.getRotationMatrix2D inputs, whose decomposed angle is the negation.
    rot_sweep   = sorted((-a, d, i, x) for a, (d, i, x) in rot_metrics.items())
    return {
        "transform":    M,
        "edge_iou":     float(edge_iou),
        "scale":        float(sc),
        "angle_deg":    float(rot),
        "confidence":   float(xcorr_score),
        "n_iterations": 0,
        "converged":    True,
        "substep_timings": substep_timings,
        "scale_sweep":  scale_sweep,
        "rot_sweep":    rot_sweep,
        "hist_src":     hist_src,
        "hist_tgt":     hist_tgt,
        "hist_xcorr":   hist_xcorr,
        "hist_rot_deg": hist_rot_deg,
        "xcorr_map":    xcorr_map,
        "best_dx":      float(dx),
        "best_dy":      float(dy),
        "rot_l1_xcorr": rot_l1_sc,
        "rot_l2_xcorr": rot_l2_sc,
        "dice_fine":    dice_fine_sc,
    }
