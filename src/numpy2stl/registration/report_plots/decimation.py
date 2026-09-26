"""Decimation-evaluation figures for the registration report.

Two views of the footprint-preserving mesh simplification (Stage 0):
  - `render_decimation_png`: original heightmap | simplified heightmap | their
    difference (metres), so the user can see exactly what detail was removed and
    confirm the footprint/silhouette is preserved.
  - `render_decimation_curve_png`: the quality/size trade-off — surface deviation
    (Hausdorff, metres) vs faces kept — with the chosen deviation budget marked,
    so the budget can be evaluated against the achievable reduction.

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
    HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None
    HAS_MPL = False

from ._common import render_three_panel


def render_decimation_png(out_path, report) -> Path | None:
    """3-panel: original STL heightmap | simplified | difference (m)."""
    orig = getattr(report, "_stl_heightmap_original", None)
    simp = report.stl_heightmap
    stats = getattr(report, "_simplify_stats", None) or {}
    if orig is None or simp is None or orig.shape != simp.shape:
        return None

    m_per_unit = float(stats.get("m_per_unit", 1.0) or 1.0)
    diff_m = (simp.astype(np.float64) - orig.astype(np.float64)) * m_per_unit

    f0 = stats.get("orig_faces"); f1 = stats.get("simplified_faces")
    haus_m = stats.get("hausdorff_m"); budget = stats.get("deviation_tol_m_metres")
    parts = []
    if f0 and f1:
        parts.append(f"faces {f0:,} → {f1:,} ({100.0 * f1 / f0:.0f}%)")
    if haus_m is not None:
        parts.append(f"max surface deviation {haus_m:.2f} m")
    if budget is not None:
        parts.append(f"budget {budget:.2f} m")
    return render_three_panel(
        out_path, orig, simp, diff_m,
        left_title="Original heightmap", right_title="Simplified heightmap",
        diff_title="Difference (simplified − original)",
        suptitle="Mesh decimation  |  " + "   ".join(parts))


def render_decimation_curve_png(out_path, report) -> Path | None:
    """Surface deviation (m) vs faces kept (%), with the chosen budget marked."""
    out_path = Path(out_path)
    if not HAS_MPL:
        raise ImportError("matplotlib is required for report rendering.")
    sweep = getattr(report, "_decimation_sweep", None)
    if not sweep:
        return None
    stats = getattr(report, "_simplify_stats", None) or {}

    kept = [100.0 * d["ratio"] for d in sweep]
    dev = [d["hausdorff_m"] for d in sweep]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(kept, dev, "o-", color="#3a7ebf", lw=2, label="achievable deviation")
    ax.set_xlabel("faces kept (%)"); ax.set_ylabel("max surface deviation (m)")
    ax.set_title("Decimation trade-off: surface deviation vs faces kept", fontsize=11)
    ax.grid(alpha=0.3)

    budget = stats.get("deviation_tol_m_metres")
    if budget is not None:
        ax.axhline(budget, color="#cc0000", ls="--", lw=1.5,
                   label=f"deviation budget ({budget:.2f} m)")
    chosen = stats.get("face_ratio")
    chosen_h = stats.get("hausdorff_m")
    if chosen is not None and chosen_h is not None:
        ax.plot([100.0 * chosen], [chosen_h], "*", ms=18, color="#e8a000",
                label=f"chosen ({100.0 * chosen:.0f}% faces, {chosen_h:.2f} m)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
