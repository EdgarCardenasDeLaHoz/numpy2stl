"""Height-comparison figures: 3-panel, difference histogram, missing analysis.

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


# ---------------------------------------------------------------------------
# Public render functions (one per asset PNG)
# ---------------------------------------------------------------------------

def render_corrected_difference_png(out_path: str | Path, report) -> Path | None:
    """
    Per-BUILDING corrected height difference (STL_m − OSM).

    Uses `comparison.building_diff_map`: one difference value per OSM footprint,
    with OSM fill-defaults and fit-outliers excluded and the affine height fit
    applied.  This is the "clean" companion to the per-pixel difference panel —
    it shows where the *buildings that actually match* are over/under-estimated,
    without the street-pixel and fill noise that skews the per-pixel map.

    Left: spatial map (RdBu_r, symmetric, cropped to STL extent).
    Right: histogram of per-building differences (should centre near 0).
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        return None
    comp = report.comparison
    bdm = getattr(comp, "building_diff_map", None)
    if bdm is None or not np.isfinite(bdm).any():
        return None

    valid = np.isfinite(bdm)
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    mr = max(5, int((rows[-1] - rows[0]) * 0.1))
    mc = max(5, int((cols[-1] - cols[0]) * 0.1))
    r0, r1 = max(0, rows[0] - mr), min(bdm.shape[0], rows[-1] + mr + 1)
    c0, c1 = max(0, cols[0] - mc), min(bdm.shape[1], cols[-1] + mc + 1)
    crop = bdm[r0:r1, c0:c1]

    # One sample PER BUILDING (the map repeats a value across each footprint's
    # pixels) — label connected footprints and take one value each.
    try:
        import cv2
        n_lbl, lbl = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)
        vals = np.array([float(bdm[lbl == i][0]) for i in range(1, n_lbl)])
    except Exception:
        vals = bdm[valid]
    n_buildings = int(vals.size)
    vmax = float(np.percentile(np.abs(vals), 95)) or 1.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = axes[0].imshow(crop, origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                        interpolation="none")
    fig.colorbar(im, ax=axes[0], label="STL − OSM (m)", fraction=0.046, pad=0.04)
    axes[0].set_title(f"Corrected per-building difference  ({n_buildings} buildings)\n"
                      f"fills + fit-outliers excluded", fontsize=11)
    axes[0].set_xlabel("col"); axes[0].set_ylabel("row")

    axes[1].hist(vals, bins=40, color="#3a7ebf", alpha=0.85, edgecolor="none")
    axes[1].axvline(0, color="k", ls="--", lw=1.2)
    axes[1].axvline(float(np.median(vals)), color="red", lw=1.5,
                    label=f"median {np.median(vals):+.1f} m")
    axes[1].set_xlabel("STL − OSM per building (m)")
    axes[1].set_ylabel("buildings")
    axes[1].set_title(f"RMSE {comp.rmse:.1f} m   bias {comp.bias:+.1f} m   "
                      f"r {comp.correlation:.3f}", fontsize=11)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_comparison_png(out_path: str | Path, report) -> Path:
    """
    Canonical 3-panel comparison -- cropped and zoomed to the actual overlap.

    Both panels are cropped to the bounding box of valid STL pixels (+10% margin)
    so the STL and OSM are shown at the same geographic scale and pixel extent.
    This makes building-level comparison legible even when the OSM covers a much
    larger area than the STL model.

    Panels (all same crop):
      1 -- STL aligned (height-scaled to metres)
      2 -- OSM building heights  (same colormap + limits as panel 1)
      3 -- Height difference (RdBu_r, symmetric)
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    comp = report.comparison
    stl_m = report.stl_aligned * comp.height_scale_used + comp.height_offset_used

    # -- Compute crop: bounding box of valid STL pixels + 10% margin --
    stl_valid = ~np.isnan(stl_m)
    rows_valid = np.where(stl_valid.any(axis=1))[0]
    cols_valid = np.where(stl_valid.any(axis=0))[0]

    if len(rows_valid) > 0 and len(cols_valid) > 0:
        margin_r = max(5, int((rows_valid[-1] - rows_valid[0]) * 0.10))
        margin_c = max(5, int((cols_valid[-1] - cols_valid[0]) * 0.10))
        r0 = max(0, rows_valid[0] - margin_r)
        r1 = min(stl_m.shape[0], rows_valid[-1] + margin_r + 1)
        c0 = max(0, cols_valid[0] - margin_c)
        c1 = min(stl_m.shape[1], cols_valid[-1] + margin_c + 1)
    else:
        r0, r1 = 0, stl_m.shape[0]
        c0, c1 = 0, stl_m.shape[1]

    crop = np.s_[r0:r1, c0:c1]
    stl_crop = stl_m[crop]
    osm_crop = report.osm_heightmap[crop]
    diff_crop = comp.difference[crop]

    # -- Shared height limits for STL and OSM (direct visual comparison) --
    all_vals = np.concatenate([
        stl_crop[~np.isnan(stl_crop)],
        osm_crop[~np.isnan(osm_crop)],
    ])
    vmin_h = 0.0
    vmax_h = float(np.percentile(all_vals, 99)) if len(all_vals) > 0 else 1.0
    vmax_h = max(vmax_h, 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    cmap_h = plt.get_cmap("viridis").copy()
    cmap_h.set_bad(color="#cccccc")
    cmap_o = plt.get_cmap("viridis").copy()
    cmap_o.set_bad(color="#ffffff")

    # Panel 1 -- STL
    im1 = axes[0].imshow(stl_crop, cmap=cmap_h, vmin=vmin_h, vmax=vmax_h, origin="lower")
    axes[0].set_title("STL Aligned (metres)", fontsize=12)
    fig.colorbar(im1, ax=axes[0], label="Height (m)", fraction=0.046, pad=0.04)

    # Panel 2 -- OSM (same scale)
    im2 = axes[1].imshow(osm_crop, cmap=cmap_o, vmin=vmin_h, vmax=vmax_h, origin="lower")
    axes[1].set_title("OSM Building Heights (m)", fontsize=12)
    fig.colorbar(im2, ax=axes[1], label="Height (m)", fraction=0.046, pad=0.04)

    # Panel 3 -- Difference
    valid_diff = diff_crop[~np.isnan(diff_crop)]
    vmax_d = float(np.percentile(np.abs(valid_diff), 99)) if len(valid_diff) > 0 else 1.0
    vmax_d = max(vmax_d, 0.01)

    cmap_diff = plt.get_cmap("RdBu_r").copy()
    cmap_diff.set_bad(color="#f0f0f0")
    im3 = axes[2].imshow(diff_crop, cmap=cmap_diff, vmin=-vmax_d, vmax=vmax_d, origin="lower")
    fig.colorbar(im3, ax=axes[2], label="STL - OSM (m)", fraction=0.046, pad=0.04)

    if comp.missing_in_osm[crop].any():
        missing_rgba = np.zeros((*diff_crop.shape, 4))
        missing_rgba[comp.missing_in_osm[crop]] = [1, 0.5, 0, 0.4]
        axes[2].imshow(missing_rgba, origin="lower", interpolation="none")
        axes[2].legend(
            handles=[Patch(facecolor="orange", alpha=0.4, label="In STL, not OSM")],
            loc="lower right", fontsize=8,
        )

    axes[2].set_title(
        f"Difference  RMSE={comp.rmse:.1f} m  r={comp.correlation:.3f}",
        fontsize=12,
    )

    for ax in axes:
        ax.set_xlabel("col (OSM pixel)", fontsize=10)
        ax.set_ylabel("row (OSM pixel)", fontsize=10)

    fig.suptitle(
        f"Registration: {report.region_name}  |  "
        f"conf={report.registration.confidence:.3f}  "
        f"scale={report.registration.scale:.4f}x  "
        f"angle={report.registration.angle_deg:.1f}deg",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_difference_hist_png(out_path: str | Path, comp) -> Path:
    """Histogram of height differences over the overlap region."""
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    valid = comp.difference[~np.isnan(comp.difference)]
    fig, ax = plt.subplots(figsize=(7, 4))

    if len(valid) > 0:
        vmax = float(np.percentile(np.abs(valid), 99))
        bins = np.linspace(-vmax, vmax, 61)
        ax.hist(valid, bins=bins, color="#3a7ebf", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="k", lw=1.5, ls="--")
        ax.axvline(comp.bias, color="red", lw=1.5, ls="-",
                   label=f"bias = {comp.bias:+.2f} m")
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No overlap data", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_xlabel("STL - OSM height (m)")
    ax.set_ylabel("Pixel count")
    ax.set_title(
        f"Height difference distribution  "
        f"RMSE={comp.rmse:.2f} m  MAE={comp.mae:.2f} m",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_missing_analysis_png(out_path: str | Path, comp) -> Path:
    """2-panel: missing_in_osm | missing_in_stl coverage maps."""
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pct_new = 100.0 * comp.missing_in_osm.sum() / max(comp.missing_in_osm.size, 1)
    pct_miss = 100.0 * comp.missing_in_stl.sum() / max(comp.missing_in_stl.size, 1)

    axes[0].imshow(comp.missing_in_osm, cmap="Oranges", origin="lower", vmin=0, vmax=1)
    axes[0].set_title(
        f"In STL, not in OSM  ({pct_new:.1f}% of pixels)\n"
        "(new buildings / OSM gaps)",
        fontsize=10,
    )

    axes[1].imshow(comp.missing_in_stl, cmap="Blues", origin="lower", vmin=0, vmax=1)
    axes[1].set_title(
        f"In OSM, not in STL  ({pct_miss:.1f}% of pixels)\n"
        "(outside STL extent / demolished buildings)",
        fontsize=10,
    )

    for ax in axes:
        ax.set_xlabel("col")
        ax.set_ylabel("row")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
