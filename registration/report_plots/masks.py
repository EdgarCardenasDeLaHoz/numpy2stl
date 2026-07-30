"""Segmentation figures: binarization, mask overlay, matched buildings, lines.

Part of the report_plots/ subpackage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Public render functions (one per asset PNG)
# ---------------------------------------------------------------------------

from ._common import _imshow_heightmap

def render_binarization_png(out_path: str | Path, report) -> Path:
    """
    Show how each heightmap is binarized into a building-presence mask, plus the
    edge (outline) signal actually used for registration.

    Heights are noisy, so registration works on the *binary footprint* (and its
    outline), not the height values. This panel makes that explicit:

      Row 1: STL heightmap | STL binary mask | STL footprint edges
      Row 2: OSM heightmap | OSM binary mask | OSM footprint edges
      Row 3: STL Sobel gradient magnitude | OSM Sobel gradient magnitude | (blank)

    The STL mask is shown BEFORE alignment (raw model) so the binarization is
    transparent; the alignment is shown separately in the agreement overlay.

    Row 3 shows the raw Sobel gradient magnitude that the rotation estimator
    (gradient_angle_histogram in align/lines.py) actually reads wall
    orientations from -- independent of building-mask segmentation quality.
    Comparing it against the binary mask in row 1/2 makes it visible when the
    two disagree (e.g. a dense mask blob vs. sharp gradient edges still
    tracing individual buildings), which is a real, useful segmentation-
    quality signal on its own.
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    from ..align import building_mask, building_edges

    stl = report.stl_heightmap
    osm = report.osm_heightmap

    def _sobel_mag(img: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(np.asarray(img, dtype=np.float64))
        if HAS_CV2:
            gx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=3)
        else:
            gy, gx = np.gradient(a)
        return np.hypot(gx, gy)

    stl_sobel = _sobel_mag(stl)
    osm_sobel = _sobel_mag(osm)

    # cell_size_m sizes the STL top-hat kernel in real metres (see
    # terrain_residual()) — matches what register_global() actually uses during
    # the search (report.cell_size_m is the same report-resolution grid these
    # heightmaps are on). Without it the kernel falls back to resolution-naive
    # pixel sizing and this panel can show a visibly different (and misleading)
    # segmentation than what the algorithm actually registered against.
    _csm = getattr(report, "cell_size_m", None)
    stl_mask = building_mask(stl, source="stl", cell_size_m=_csm)
    osm_mask  = building_mask(osm, source="osm")
    stl_edge  = building_edges(stl, source="stl", cell_size_m=_csm)
    osm_edge  = building_edges(osm, source="osm")

    # Component stats — OSM cell size from known_scale if available
    known_scale = getattr(report, "known_scale", None)
    h, w = osm.shape
    # OSM covers (stl_footprint_m * osm_margin) per side; if known_scale=1/osm_margin,
    # osm_margin=1/known_scale and osm cell size = stl_footprint_m/(known_scale*resolution).
    # We don't have stl_footprint_m here, so leave cell_m=None (px only in this panel).
    stl_stats = building_component_stats(stl_mask)
    osm_stats = building_component_stats(osm_mask)

    def _stats_line(s):
        return (f"n={s['count']}  mean={s['mean_px']:.0f} px²  "
                f"median={s['median_px']:.0f} px²  "
                f"IQR [{s['p25_px']:.0f}–{s['p75_px']:.0f}]")

    fig, ax = plt.subplots(3, 3, figsize=(16, 16))

    # Row 1 — STL
    _imshow_heightmap(ax[0, 0], stl, "STL heightmap (model units)", "viridis", "#cccccc")
    ax[0, 1].imshow(stl_mask, origin="lower", cmap="gray")
    ax[0, 1].set_title(
        f"STL binary mask  (adaptive top-hat threshold, morphological opening)\n"
        f"{100*stl_mask.mean():.1f}% of frame\n"
        f"{_stats_line(stl_stats)}", fontsize=9)
    ax[0, 2].imshow(stl_edge, origin="lower", cmap="gray")
    ax[0, 2].set_title(f"STL footprint edges  ({100*stl_edge.mean():.2f}%)\n"
                       "registration signal", fontsize=10)

    # Row 2 — OSM
    _imshow_heightmap(ax[1, 0], osm, "OSM heightmap (metres)", "viridis", "#ffffff")
    ax[1, 1].imshow(osm_mask, origin="lower", cmap="gray")
    ax[1, 1].set_title(
        f"OSM binary mask  (any rasterized footprint)\n"
        f"{100*osm_mask.mean():.1f}% of frame\n"
        f"{_stats_line(osm_stats)}", fontsize=9)
    ax[1, 2].imshow(osm_edge, origin="lower", cmap="gray")
    ax[1, 2].set_title(f"OSM footprint edges  ({100*osm_edge.mean():.1f}%)\n"
                       "registration signal", fontsize=10)

    # Row 3 — Sobel gradient magnitude (what the rotation histogram reads wall
    # orientations from). log1p-scaled: gradient magnitude is heavy-tailed
    # (a few strong building edges, lots of near-zero flat terrain/roof).
    stl_sobel_disp = np.log1p(stl_sobel)
    osm_sobel_disp = np.log1p(osm_sobel)
    ax[2, 0].imshow(stl_sobel_disp, origin="lower", cmap="magma")
    ax[2, 0].set_title("STL Sobel gradient magnitude (log scale)\n"
                       "rotation histogram signal — segmentation-independent", fontsize=10)
    ax[2, 1].imshow(osm_sobel_disp, origin="lower", cmap="magma")
    ax[2, 1].set_title("OSM Sobel gradient magnitude (log scale)\n"
                       "rotation histogram signal — segmentation-independent", fontsize=10)
    ax[2, 2].axis("off")

    for a in ax.ravel():
        a.set_xlabel("col"); a.set_ylabel("row")
    ax[2, 2].set_xlabel(""); ax[2, 2].set_ylabel("")

    fig.suptitle("Binarization: heightmaps -> building masks -> edges -> gradient "
                 "(heights are noisy; registration uses the binary footprint; "
                 "rotation uses the raw gradient)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_mask_overlay_png(out_path: str | Path, report) -> Path:
    """
    Building-presence agreement map (the coarse true/false registration signal).

    Cropped to the STL extent. Each pixel is colored by which dataset reports
    a building there:
        green  = both agree (building in STL and OSM)  -> good registration
        orange = STL only   (in model, not in OSM)
        blue   = OSM only   (in OSM, not in model)
    The intersection-over-union (IoU) of the two footprint masks is the
    headline registration-quality number.
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    from ..align import building_mask, apply_transform, _tolerant_iou

    osm_mask = building_mask(report.osm_heightmap, source="osm")
    # Prefer the prism-decomposition footprint polygons (separated, regularized,
    # already in OSM space) rasterized directly — these are the building LINES with
    # no working-resolution re-segmentation that would fuse neighbours into blobs.
    polys = getattr(report, "_stl_polygons", None)
    stl_building_mask = getattr(report, "stl_building_mask", None)
    if polys:
        stl_mask = np.zeros(report.osm_heightmap.shape, dtype=np.uint8)
        import cv2 as _cv2
        for p in polys:
            _cv2.fillPoly(stl_mask, [np.asarray(p, dtype=np.int32)], 1)
        stl_mask = stl_mask.astype(bool)
    elif stl_building_mask is not None and stl_building_mask.shape == report.osm_heightmap.shape:
        # The mask _run_comparison actually scored comp_result against — already
        # in OSM space and already has vegetation/water/elevated-roadway excluded.
        # Re-deriving from stl_heightmap here would silently skip that exclusion
        # and show a mask that doesn't match what comp_result.footprint_iou means.
        stl_mask = stl_building_mask.astype(bool)
    else:
        # Fallback: segment the heightmap and warp into OSM space.
        stl_mask_orig = building_mask(report.stl_heightmap, source="stl",
                                      split_watershed=True,
                                      cell_size_m=getattr(report, "cell_size_m", None)).astype(np.float64)
        stl_mask = apply_transform(
            stl_mask_orig, report.registration.transform,
            output_shape=report.osm_heightmap.shape, fill_value=0.0) > 0.5

    both = stl_mask & osm_mask
    stl_only = stl_mask & ~osm_mask
    osm_only = osm_mask & ~stl_mask

    inter = both.sum()
    union = (stl_mask | osm_mask).sum()
    iou = float(inter / union) if union > 0 else 0.0
    iou_tol = _tolerant_iou(stl_mask, osm_mask, tol_px=1)

    # Crop to STL extent + margin
    rows = np.where(stl_mask.any(axis=1))[0]
    cols = np.where(stl_mask.any(axis=0))[0]
    if len(rows) and len(cols):
        mr = max(5, int((rows[-1] - rows[0]) * 0.1))
        mc = max(5, int((cols[-1] - cols[0]) * 0.1))
        r0, r1 = max(0, rows[0] - mr), min(stl_mask.shape[0], rows[-1] + mr + 1)
        c0, c1 = max(0, cols[0] - mc), min(stl_mask.shape[1], cols[-1] + mc + 1)
    else:
        r0, r1, c0, c1 = 0, stl_mask.shape[0], 0, stl_mask.shape[1]
    crop = np.s_[r0:r1, c0:c1]

    rgb = np.ones((*both[crop].shape, 3))
    rgb[both[crop]]     = [0.20, 0.70, 0.25]   # green  — agreement
    rgb[stl_only[crop]] = [1.00, 0.55, 0.10]   # orange — STL only
    rgb[osm_only[crop]] = [0.20, 0.45, 0.85]   # blue   — OSM only

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, origin="lower", interpolation="none")
    ax.set_title(
        f"Building footprint agreement  |  IoU = {iou:.3f}  "
        f"(tolerant {iou_tol:.3f})  |  projection: {report.registration.projection}\n"
        f"green = both ({100*inter/max(union,1):.0f}% of union)   "
        f"orange = STL only   blue = OSM only",
        fontsize=11,
    )
    ax.set_xlabel("col (OSM pixel)")
    ax.set_ylabel("row (OSM pixel)")

    fig.tight_layout()
    # Save at a dpi that keeps the cropped footprint array near 1:1 (or finer),
    # so the higher-resolution agreement detail isn't thrown away on downscale.
    _dpi = int(np.clip(max(both[crop].shape) / 8 * 1.1, 150, 400))
    fig.savefig(str(out_path), dpi=_dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_matched_buildings_png(out_path: str | Path, report) -> Path:
    """
    Per-building height comparison on the *matched* footprints.

    Matches OSM building components to overlapping STL footprint, then plots:
      - left : scatter of STL height vs OSM height for each matched building,
               with the y=x line and a fitted scale.
      - right: spatial map of per-building height error (matched buildings only).

    This is the signal that matters for downstream work: where the datasets
    agree on footprint, how well do their heights agree?
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    from ..align import building_mask
    import cv2

    comp = report.comparison
    stl_m = report.stl_aligned * comp.height_scale_used + comp.height_offset_used
    osm = report.osm_heightmap

    stl_mask = building_mask(stl_m, source="stl", cell_size_m=getattr(report, "cell_size_m", None))
    osm_mask = building_mask(osm, source="osm")

    # Label OSM building components; match each to STL where they overlap.
    n, labels = cv2.connectedComponents(osm_mask.astype(np.uint8), connectivity=8)
    recs = []  # (label_id, comp_mask, oh, sh)
    for i in range(1, n):
        comp_mask = labels == i
        overlap = comp_mask & stl_mask
        if overlap.sum() < 3:        # require a real overlap
            continue
        oh = float(np.nanmedian(osm[comp_mask]))
        sh = float(np.nanpercentile(stl_m[overlap], 95))   # tip height (matches OSM)
        if np.isnan(oh) or np.isnan(sh):
            continue
        recs.append((comp_mask, oh, sh))

    # Exclude OSM fill-default buildings (the over-represented modal height, e.g.
    # untagged buildings all stamped at default_height=10 m) — same rule as the
    # height metrics, so the scatter matches the reported numbers.
    all_oh = np.array([r[1] for r in recs]) if recs else np.array([])
    fill_val = np.nan
    if all_oh.size:
        rvals, rcounts = np.unique(np.round(all_oh, 1), return_counts=True)
        if int(rcounts.max()) >= max(5, int(0.10 * all_oh.size)):
            fill_val = float(rvals[np.argmax(rcounts)])

    osm_h, stl_h, err_map = [], [], np.full(osm.shape, np.nan)
    n_fill = 0
    for comp_mask, oh, sh in recs:
        if np.isfinite(fill_val) and abs(round(oh, 1) - fill_val) <= 0.05:
            n_fill += 1
            continue
        osm_h.append(oh); stl_h.append(sh)
        err_map[comp_mask] = sh - oh

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    if osm_h:
        osm_h = np.array(osm_h); stl_h = np.array(stl_h)
        lim = max(osm_h.max(), stl_h.max()) * 1.05
        axes[0].scatter(osm_h, stl_h, s=14, alpha=0.5, color="#2a7a2a")
        axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="y = x (perfect)")
        # Affine least-squares fit (slope + intercept) — matches the height fit in
        # compare(); STL is already converted to metres so a good fit lies on y=x.
        if len(osm_h) > 2:
            A = np.vstack([osm_h, np.ones_like(osm_h)]).T
            (slope, intc), *_ = np.linalg.lstsq(A, stl_h, rcond=None)
            axes[0].plot([0, lim], [intc, slope * lim + intc], "r-", lw=1.5,
                         label=f"fit: STL = {slope:.2f}·OSM {intc:+.1f}")
        r = np.corrcoef(osm_h, stl_h)[0, 1] if len(osm_h) > 2 else float("nan")
        axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
        axes[0].set_xlabel("OSM building height (m)")
        axes[0].set_ylabel("STL building height (m)")
        _fill_note = (f"  (excl. {n_fill} OSM fill @ {fill_val:.0f} m)"
                      if n_fill else "")
        axes[0].set_title(f"Matched buildings: {len(osm_h)}   r = {r:.3f}{_fill_note}",
                          fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].set_aspect("equal")
    else:
        axes[0].text(0.5, 0.5, "No matched buildings", ha="center", va="center",
                     transform=axes[0].transAxes)

    # Spatial error map (crop to STL extent)
    rows = np.where(stl_mask.any(axis=1))[0]
    cols = np.where(stl_mask.any(axis=0))[0]
    if len(rows) and len(cols):
        crop = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    else:
        crop = np.s_[:, :]
    em = err_map[crop]
    valid = em[~np.isnan(em)]
    vmax = float(np.percentile(np.abs(valid), 95)) if len(valid) else 1.0
    vmax = max(vmax, 0.1)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#f4f4f4")
    im = axes[1].imshow(em, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
    fig.colorbar(im, ax=axes[1], label="STL - OSM (m)", fraction=0.046, pad=0.04)
    axes[1].set_title("Per-building height error (matched only)", fontsize=11)
    axes[1].set_xlabel("col"); axes[1].set_ylabel("row")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_footprint_rgchannel_png(out_path: str | Path, report) -> Path:
    """
    False-color footprint overlay after alignment.

    STL building footprint → red channel.
    OSM building footprint → green channel.
    Overlap (both) appears yellow; STL-only = red; OSM-only = green; neither = black.
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    from ..align import building_mask, apply_transform
    from matplotlib.patches import Patch

    # Prefer report.stl_building_mask -- the mask _run_comparison actually scored
    # against, already warped into OSM space with split_watershed=True applied
    # AFTER the warp (compare.py's _warp_mask_compare). Re-deriving here by
    # masking the RAW heightmap first and warping the binary result afterward
    # (the old unconditional path below) re-fuses watershed-split buildings back
    # together: downscaling a binary mask (e.g. Paris/Barcelona's ~0.667x locked
    # scale) blurs/merges the 1px ridge lines between adjacent buildings well
    # before the >0.5 threshold can preserve them, so dense cities rendered as
    # one solid red blob here even though score_alignment() (which warps the
    # CONTINUOUS heightmap first, then masks) never saw that corruption --
    # measured on Paris: 48 fused components at max 12.4% of frame this way vs.
    # 139 correctly-separated components via stl_building_mask, same IoU.
    stl_building_mask = getattr(report, "stl_building_mask", None)
    if stl_building_mask is not None and stl_building_mask.shape == report.osm_heightmap.shape:
        stl_mask = stl_building_mask.astype(np.float32)
    else:
        # Fallback: warp the heightmap FIRST (continuous data warps cleanly),
        # then mask -- matches score_alignment()'s safe ordering.
        aligned_hm = apply_transform(
            report.stl_heightmap, report.registration.transform,
            output_shape=report.osm_heightmap.shape,
        )
        stl_mask = building_mask(aligned_hm, source="stl",
                                  cell_size_m=getattr(report, "cell_size_m", None)).astype(np.float32)
    osm_mask = building_mask(report.osm_heightmap, source="osm").astype(np.float32)

    # Crop to STL extent + small margin
    rows = np.where(stl_mask.astype(bool).any(axis=1))[0]
    cols = np.where(stl_mask.astype(bool).any(axis=0))[0]
    if len(rows) and len(cols):
        mr = max(10, int((rows[-1] - rows[0]) * 0.05))
        mc = max(10, int((cols[-1] - cols[0]) * 0.05))
        r0 = max(0, rows[0] - mr)
        r1 = min(stl_mask.shape[0], rows[-1] + mr + 1)
        c0 = max(0, cols[0] - mc)
        c1 = min(stl_mask.shape[1], cols[-1] + mc + 1)
    else:
        r0, r1 = 0, stl_mask.shape[0]
        c0, c1 = 0, stl_mask.shape[1]

    sr = stl_mask[r0:r1, c0:c1]
    gr = osm_mask[r0:r1, c0:c1]
    rgb = np.stack([sr, gr, np.zeros_like(sr)], axis=-1)

    iou = float(np.logical_and(sr > 0, gr > 0).sum() / max(np.logical_or(sr > 0, gr > 0).sum(), 1))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(rgb, origin="lower", interpolation="none")
    ax.set_title(
        f"Footprint overlay (post-alignment)  |  IoU = {iou:.3f}\n"
        "red = STL only   green = OSM only   yellow = both",
        fontsize=11,
    )
    ax.set_xlabel("col (OSM pixel)")
    ax.set_ylabel("row (OSM pixel)")
    ax.legend(
        handles=[
            Patch(facecolor="#ff0000", label="STL only"),
            Patch(facecolor="#00cc00", label="OSM only"),
            Patch(facecolor="#ffff00", label="Both"),
        ],
        loc="lower right", fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Building component statistics
# ---------------------------------------------------------------------------


def render_vectorized_png(out_path: str | Path, report) -> Path | None:
    """
    Vectorized building footprints: the raster masks traced into simplified
    polygons (cv2.findContours + Douglas-Peucker).  STL (left) and OSM (right),
    each polygon drawn as a closed outline.  This is the polygon representation
    OSM uses natively, so STL footprints become directly comparable / exportable.
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        return None
    from ..align import building_mask, vectorize_buildings
    from matplotlib.collections import LineCollection

    # Prefer the hi-res adaptive STL polygons (separated footprints) when present.
    stl_polys = getattr(report, "_stl_polygons", None)
    if not stl_polys:
        stl_polys = vectorize_buildings(building_mask(
            report.stl_aligned, source="stl", cell_size_m=getattr(report, "cell_size_m", None)))
    osm_polys = vectorize_buildings(building_mask(report.osm_heightmap, source="osm"))

    def _draw(ax, polys, title, color):
        segs, nverts = [], 0
        for p in polys:
            pts = np.vstack([p, p[:1]])           # close the ring
            segs.extend([[pts[i], pts[i + 1]] for i in range(len(pts) - 1)])
            nverts += len(p)
        if segs:
            ax.add_collection(LineCollection(segs, colors=color, linewidths=0.7))
        ax.set_xlim(0, report.osm_heightmap.shape[1])
        ax.set_ylim(0, report.osm_heightmap.shape[0])
        ax.set_aspect("equal")
        avg = nverts / max(1, len(polys))
        ax.set_title(f"{title}: {len(polys)} polygons, {nverts} verts ({avg:.1f}/poly)",
                     fontsize=10)
        ax.set_xlabel("col"); ax.set_ylabel("row")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    _draw(axes[0], stl_polys, "STL footprints (vectorized)", "#c0392b")
    _draw(axes[1], osm_polys, "OSM footprints (vectorized)", "#2a7a2a")
    fig.suptitle("Vectorized building footprints  (raster mask → simplified polygons)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def building_component_stats(mask: np.ndarray, cell_m: float | None = None,
                              min_blob_px: int = 10) -> dict:
    """
    Connected-component statistics for a binary building mask.

    Parameters
    ----------
    mask      : bool array — building presence
    cell_m    : metres per pixel (if known, areas are also reported in m²)
    min_blob_px : minimum component size to count (noise filter)

    Returns
    -------
    dict with: count, mean_px, median_px, p25_px, p75_px, total_px,
               and *_m2 variants if cell_m is provided.
    """
    try:
        import cv2
        n, _labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)  # skip background
        areas = areas[areas >= min_blob_px]
    except Exception:
        areas = np.array([])

    if len(areas) == 0:
        base = {"count": 0, "mean_px": 0.0, "median_px": 0.0,
                "p25_px": 0.0, "p75_px": 0.0, "total_px": 0}
        if cell_m:
            base.update(mean_m2=0.0, median_m2=0.0, total_m2=0.0)
        return base

    result = {
        "count":     int(len(areas)),
        "mean_px":   float(np.mean(areas)),
        "median_px": float(np.median(areas)),
        "p25_px":    float(np.percentile(areas, 25)),
        "p75_px":    float(np.percentile(areas, 75)),
        "total_px":  int(areas.sum()),
    }
    if cell_m is not None:
        m2 = cell_m ** 2
        result["mean_m2"]   = result["mean_px"]   * m2
        result["median_m2"] = result["median_px"] * m2
        result["total_m2"]  = result["total_px"]  * m2
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
