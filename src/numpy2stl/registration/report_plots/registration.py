"""Registration figures: transform summary, scale/rotation sweeps, xcorr, angle hist.

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

def render_transform_summary_png(out_path: str | Path, reg) -> Path:
    """Text summary of the found registration transform."""
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    lines = [
        f"Confidence (ECC cc): {reg.confidence:.4f}  (raw FFT xcorr peak -- search diagnostic, not match quality)",
        f"Spatial scale:       {reg.scale:.4f}x",
        f"Rotation:            {reg.angle_deg:.2f} deg",
        f"Translation (tx):    {reg.transform[0, 2]:.2f} px",
        f"Translation (ty):    {reg.transform[1, 2]:.2f} px",
        f"ECC converged:       {'Yes' if reg.converged else 'No'}",
    ]
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, va="top", ha="left", fontfamily="monospace",
            fontsize=11, transform=ax.transAxes)
    ax.set_title("Registration Transform Parameters", fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_scale_sweep_png(
    out_path: str | Path,
    scale_sweep: list[tuple[float, float]],
    known_scale: float | None = None,
    found_scale: float | None = None,
    area_scale: float | None = None,
    fourier_scale: float | None = None,
) -> Path | None:
    """
    Plot footprint Dice vs spatial scale factor — the scale-determinism graph.

    Three independent scale signals are overlaid so it is obvious how the scale
    was determined and whether the methods agree:
      - blue curve : Dice of footprint masks vs scale (translation re-solved by
        phase correlation at each point) — the empirical landscape
      - orange line: scale derived from the footprint-AREA ratio
      - purple line: scale derived independently by FOURIER profile analysis
    plus the scale actually used (green) and the geometric anchor (red).
    """
    out_path = Path(out_path)
    if not HAS_MPL or not scale_sweep:
        return None

    scales_arr = np.array([row[0] for row in scale_sweep])
    dice_arr   = np.array([row[1] for row in scale_sweep])
    # Optional extra metrics (iou, xcorr) when the sweep carries 4-tuples.
    has_extra  = len(scale_sweep[0]) >= 4
    iou_arr    = np.array([row[2] for row in scale_sweep]) if has_extra else None
    xcorr_arr  = np.array([row[3] for row in scale_sweep]) if has_extra else None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scales_arr, dice_arr, "o-", color="#3a7ebf", ms=4, lw=1.8,
            label="Footprint Dice (edges)")
    if has_extra:
        ax.plot(scales_arr, iou_arr, "s--", color="#2ca02c", ms=3, lw=1.2,
                alpha=0.8, label="Edge IoU")
        ax.plot(scales_arr, xcorr_arr, "^:", color="#777777", ms=3, lw=1.2,
                alpha=0.8, label="Heightmap xcorr")

    # Three independent scale determinations as vertical reference lines.
    if area_scale is not None:
        ax.axvline(area_scale, color="#e08214", ls="-", lw=1.8,
                   label=f"Area-ratio scale  {area_scale:.3f}×")
    if fourier_scale is not None:
        ax.axvline(fourier_scale, color="#8a2be2", ls="-.", lw=1.8,
                   label=f"Fourier scale  {fourier_scale:.3f}×")
    if known_scale is not None:
        ax.axvline(known_scale, color="#cc3333", ls="--", lw=1.4,
                   label=f"Geometric anchor  {known_scale:.3f}×")
    if found_scale is not None:
        ax.axvline(found_scale, color="#1a8c1a", ls=":", lw=2.0,
                   label=f"Scale used  {found_scale:.3f}×")

    # Annotate the peak of the scale-determining metric (heightmap xcorr when
    # available, else Dice) — that is the scale actually adopted.
    if has_extra:
        pk = int(np.argmax(xcorr_arr))
        ax.annotate(
            f"peak xcorr {xcorr_arr[pk]:.3f}\n@ {scales_arr[pk]:.3f}×",
            xy=(scales_arr[pk], xcorr_arr[pk]),
            xytext=(8, 16), textcoords="offset points",
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
        )
    else:
        best_idx = int(np.argmax(dice_arr))
        ax.annotate(
            f"peak Dice {dice_arr[best_idx]:.3f}\n@ {scales_arr[best_idx]:.3f}×",
            xy=(scales_arr[best_idx], dice_arr[best_idx]),
            xytext=(8, -20), textcoords="offset points",
            fontsize=9, color="#333",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
        )

    ax.set_xlabel("Scale (STL px → OSM px)")
    ax.set_ylabel("Match score")
    ax.set_title("Scale determinism — match metrics vs spatial scale\n"
                 "(Dice/IoU/xcorr curves + independent scale estimates)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_xcorr_map_png(
    out_path: str | Path,
    xcorr_map: "np.ndarray | None",
    best_dx: float,
    best_dy: float,
) -> Path | None:
    """
    2-D FFT cross-correlation map at the best rotation, with the found
    translation peak marked.  Hot = high overlap score; the bright spot
    shows where the STL edges best align with the OSM edges.
    """
    out_path = Path(out_path)
    if not HAS_MPL or xcorr_map is None:
        return None

    h, w = xcorr_map.shape
    # Shift so (0,0) is at centre — same convention as the peak decoding
    xcorr_shifted = np.roll(np.roll(xcorr_map, h // 2, axis=0), w // 2, axis=1)

    # Axis labels in pixels (translation range = ±half the image)
    ext = [-w // 2, w // 2, -h // 2, h // 2]

    # Log scale to reveal peak structure (xcorr values are small, log brings out detail)
    # Clip to avoid log(0); use small positive floor
    xcorr_log = np.log(np.maximum(xcorr_shifted, 1e-9))

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(xcorr_log, origin="lower", cmap="hot",
                   extent=ext, aspect="equal")
    fig.colorbar(im, ax=ax, label="log(Cross-correlation)", fraction=0.046, pad=0.04)

    # Mark the found peak
    ax.plot(best_dx, best_dy, "c+", ms=16, mew=2, label=f"Found peak ({best_dx:.0f}, {best_dy:.0f}) px")
    ax.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax.axvline(0, color="white", lw=0.5, alpha=0.4)

    ax.set_xlabel("dx (pixels)")
    ax.set_ylabel("dy (pixels)")
    ax.set_title("Translation cross-correlation map (log scale)\n"
                 "(reveals secondary peaks; bright = best alignment)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_angle_histogram_png(
    out_path: str | Path,
    hist_src: "np.ndarray",
    hist_tgt: "np.ndarray",
    hist_xcorr: "np.ndarray",
    hist_rot_deg: float,
    found_rot: float | None = None,
) -> Path | None:
    """
    Three-panel figure:
      Left  — STL (red) and OSM (blue) line angle histograms overlaid.
               Dominant peaks reveal the street-grid orientation of each dataset.
               When registered, the STL histogram should match the OSM histogram
               after a rotation by `found_rot`.
      Centre — STL histogram rotated by `found_rot` overlaid on OSM.
               If the rotation is correct both curves should align.
      Right  — 1-D circular cross-correlation of the two histograms.
               The peak gives the histogram rotation estimate (translation-invariant).
    """
    out_path = Path(out_path)
    if not HAS_MPL:
        return None

    n_bins = len(hist_src)
    angles = np.linspace(0, 180, n_bins, endpoint=False)

    # Normalise for display
    s = hist_src / (hist_src.max() + 1e-9)
    t = hist_tgt / (hist_tgt.max() + 1e-9)

    # Rotate STL histogram by found_rot to show alignment
    shift_bins = int(round(hist_rot_deg * n_bins / 180.0)) % n_bins
    s_shifted  = np.roll(s, shift_bins)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: angle-distribution histograms (filled bars, length-weighted)
    bw = 180.0 / n_bins
    axes[0].bar(angles, s, width=bw, align="edge", color="#d44", alpha=0.45,
                label="STL edges")
    axes[0].bar(angles, t, width=bw, align="edge", color="#44a", alpha=0.45,
                label="OSM edges")
    axes[0].plot(angles, s, color="#d44", lw=1.0)
    axes[0].plot(angles, t, color="#44a", lw=1.0)
    # Mark each dataset's dominant orientation.
    s_pk = float(angles[int(np.argmax(s))])
    t_pk = float(angles[int(np.argmax(t))])
    axes[0].axvline(s_pk, color="#d44", ls=":", lw=1.4, alpha=0.8,
                    label=f"STL peak {s_pk:.0f}°")
    axes[0].axvline(t_pk, color="#44a", ls=":", lw=1.4, alpha=0.8,
                    label=f"OSM peak {t_pk:.0f}°")
    axes[0].set_xlabel("Line angle (°)")
    axes[0].set_ylabel("Normalised length-weighted count")
    axes[0].set_title("Angle distribution histogram\n(peaks = dominant wall directions)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(0, 180)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: STL rotated by histogram estimate
    axes[1].plot(angles, s_shifted, color="#d44", lw=1.5,
                 label=f"STL rotated {hist_rot_deg:+.1f}°")
    axes[1].plot(angles, t, color="#44a", lw=1.5, label="OSM edges", alpha=0.8)
    if found_rot is not None and abs(found_rot - hist_rot_deg) > 0.5:
        shift2 = int(round(found_rot * n_bins / 180.0)) % n_bins
        axes[1].plot(angles, np.roll(s, shift2), color="#a80", lw=1, ls="--",
                     label=f"STL @ final {found_rot:+.1f}°", alpha=0.7)
    axes[1].set_xlabel("Line angle (°)")
    axes[1].set_title("After applying histogram rotation\n(should overlap OSM if correct)")
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 180)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: cross-correlation
    xcorr_angles = np.linspace(-90, 90, n_bins, endpoint=False)
    xcorr_rolled = np.roll(hist_xcorr, n_bins // 2)
    xcorr_norm   = xcorr_rolled / (xcorr_rolled.max() + 1e-9)
    axes[2].plot(xcorr_angles, xcorr_norm, color="#333", lw=1.5)
    axes[2].axvline(hist_rot_deg if abs(hist_rot_deg) <= 90 else hist_rot_deg - 180,
                    color="#c33", ls="--", lw=1.8,
                    label=f"Histogram peak  {hist_rot_deg:+.1f}°")
    axes[2].set_xlabel("Rotation offset (°)")
    axes[2].set_ylabel("Cross-correlation")
    axes[2].set_title("Histogram cross-correlation\n(translation-invariant rotation estimate)")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Line angle analysis — rotation from dominant building directions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_rot_sweep_png(
    out_path: str | Path,
    rot_sweep: list[tuple[float, float]],
    found_rot: float | None = None,
    hist_rot_deg: float | None = None,
    rot_l1_xcorr: "dict[float, float] | None" = None,
    rot_l2_xcorr: "dict[float, float] | None" = None,
    dice_fine: "dict[float, float] | None" = None,
) -> Path | None:
    """All rotation sweeps in one two-panel figure.

    Left panel  — full range (1° step): Dice / IoU / xcorr context.
    Right panel — zoomed ±6° around selected rotation: L1 xcorr (0.25°),
                  L2 xcorr (0.05°), Dice refinement (0.25°).
    Both panels share vertical marker lines for the histogram estimate and
    the final chosen rotation so the coarse→fine progression is visible.
    When sub-degree data is absent the right panel is omitted (single panel).
    """
    out_path = Path(out_path)
    if not HAS_MPL or not rot_sweep:
        return None

    rots_arr  = np.array([row[0] for row in rot_sweep])
    dice_arr  = np.array([row[1] for row in rot_sweep])
    has_extra = len(rot_sweep[0]) >= 4
    iou_arr   = np.array([row[2] for row in rot_sweep]) if has_extra else None
    xcorr_arr = np.array([row[3] for row in rot_sweep]) if has_extra else None

    has_detail = bool(rot_l1_xcorr or rot_l2_xcorr or dice_fine)

    if has_detail:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(14, 4.5),
            gridspec_kw={"width_ratios": [3, 2]},
        )
    else:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax2 = None

    # ── Left panel: full 1°-step sweep ──────────────────────────────────────
    ax.plot(rots_arr, dice_arr, "o-", color="#e07b20", ms=4, lw=1.8,
            label="Footprint Dice (edges, 1° step)")
    if has_extra:
        ax.plot(rots_arr, iou_arr, "s--", color="#2ca02c", ms=3, lw=1.2,
                alpha=0.7, label="Edge IoU (1° step)")
        ax.plot(rots_arr, xcorr_arr, "^:", color="#888888", ms=3, lw=1.0,
                alpha=0.6, label="Heightmap xcorr (1° step)")

    if hist_rot_deg is not None:
        ax.axvline(hist_rot_deg, color="#9467bd", ls=":", lw=1.5,
                   label=f"Histogram estimate  {hist_rot_deg:.1f}°")
    if found_rot is not None:
        ax.axvline(found_rot, color="#1a8c1a", ls="--", lw=2.0,
                   label=f"Selected  {found_rot:.2f}°")

    best_idx = int(np.argmax(dice_arr))
    ax.annotate(
        f"peak Dice {dice_arr[best_idx]:.3f} @ {rots_arr[best_idx]:.1f}°",
        xy=(rots_arr[best_idx], dice_arr[best_idx]),
        xytext=(8, -20), textcoords="offset points",
        fontsize=8, color="#333",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
    )
    ax.set_xlabel("Rotation (degrees)")
    ax.set_ylabel("Match score")
    ax.set_title("Full sweep — 1° resolution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right panel: sub-degree detail ──────────────────────────────────────
    if ax2 is not None:
        centre = found_rot if found_rot is not None else 0.0
        zoom_hw = 6.0

        if rot_l1_xcorr:
            rk = np.array(sorted(rot_l1_xcorr))
            rv = np.array([rot_l1_xcorr[k] for k in rk])
            # Normalise xcorr to [0,1] over visible window for comparability.
            mask = (rk >= centre - zoom_hw) & (rk <= centre + zoom_hw)
            if mask.sum() >= 2:
                lo, hi = rv[mask].min(), rv[mask].max()
                rv_n = (rv - lo) / (hi - lo + 1e-12)
                ax2.plot(rk[mask], rv_n[mask], ".-", color="#1f77b4",
                         ms=4, lw=1.4, alpha=0.85,
                         label="L1 xcorr (0.25°, norm.)")

        if rot_l2_xcorr:
            rk = np.array(sorted(rot_l2_xcorr))
            rv = np.array([rot_l2_xcorr[k] for k in rk])
            mask = (rk >= centre - zoom_hw) & (rk <= centre + zoom_hw)
            if mask.sum() >= 2:
                lo, hi = rv[mask].min(), rv[mask].max()
                rv_n = (rv - lo) / (hi - lo + 1e-12)
                ax2.plot(rk[mask], rv_n[mask], "o-", color="#17becf",
                         ms=5, lw=2.0,
                         label="L2 xcorr (0.05°, norm.)")

        if dice_fine:
            rk = np.array(sorted(dice_fine))
            rv = np.array([dice_fine[k] for k in rk])
            mask = (rk >= centre - zoom_hw) & (rk <= centre + zoom_hw)
            if mask.sum() >= 2:
                lo, hi = rv[mask].min(), rv[mask].max()
                rv_n = (rv - lo) / (hi - lo + 1e-12)
                ax2.plot(rk[mask], rv_n[mask], "s-", color="#e07b20",
                         ms=5, lw=2.0,
                         label="Dice refine (0.25°, norm.)")

        if hist_rot_deg is not None:
            ax2.axvline(hist_rot_deg, color="#9467bd", ls=":", lw=1.5,
                        label=f"Histogram  {hist_rot_deg:.1f}°")
        if found_rot is not None:
            ax2.axvline(found_rot, color="#1a8c1a", ls="--", lw=2.0,
                        label=f"Selected  {found_rot:.2f}°")

        ax2.set_xlim(centre - zoom_hw, centre + zoom_hw)
        ax2.set_xlabel("Rotation (degrees)")
        ax2.set_ylabel("Normalised score (0=worst, 1=best within window)")
        ax2.set_title(f"Zoom ±{zoom_hw:.0f}° — sub-degree detail\n"
                      "(each series normalised independently)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Rotation sweep (diagnostic) — final rotation is taken from the "
                 "translation-invariant line-angle histogram, not these curves",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
