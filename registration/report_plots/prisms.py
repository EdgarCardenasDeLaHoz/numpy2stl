"""Prism-decomposition figure for the registration report.

Shows the STL re-expressed as a sum of extruded prisms (the reverse of the OSM
render): original heightmap | prism-model heightmap | their difference (metres),
annotated with building / prism / layer counts and the achieved deviation.

Part of the report_plots/ subpackage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from ._common import render_three_panel


def render_prism_decomposition_png(out_path, report) -> Path | None:
    """3-panel: original heightmap | prism-model heightmap | difference (m)."""
    out_path = Path(out_path)
    orig = getattr(report, "_stl_heightmap_original", None)
    prism = report.stl_heightmap
    st = getattr(report, "_prism_stats", None) or {}
    if orig is None or prism is None or orig.shape != prism.shape:
        return None

    mpu = float(st.get("m_per_unit", 1.0) or 1.0)
    # The prism model is height-above-ground (from z0=0); the original heightmap is
    # absolute z (includes the base plate).  Normalise each to its own ground level
    # (low percentile) before differencing so the constant base offset doesn't
    # swamp the real shape difference.
    o = orig.astype(np.float64); p = prism.astype(np.float64)
    o0 = float(np.nanpercentile(o, 2)) if np.isfinite(o).any() else 0.0
    p0 = float(np.nanpercentile(p, 2)) if np.isfinite(p).any() else 0.0
    diff_m = ((p - p0) - (o - o0)) * mpu

    parts = []
    if st.get("n_buildings"):
        parts.append(f"{st['n_buildings']} buildings → {st.get('n_prisms', 0)} prisms")
    if st.get("mean_layers"):
        parts.append(f"mean {st['mean_layers']:.1f} layers")
    if st.get("sloped_caps"):
        parts.append(f"{st['sloped_caps']} sloped caps")
    if st.get("hausdorff_m") is not None and np.isfinite(st.get("hausdorff_m", np.nan)):
        parts.append(f"deviation {st['hausdorff_m']:.2f} m (budget {st.get('deviation_tol_m_metres', float('nan')):.2f} m)")
    return render_three_panel(
        out_path, orig, prism, diff_m,
        left_title="Original heightmap", right_title="Prism model (sum of extruded prisms)",
        diff_title="Difference (prism − original)",
        suptitle="Prism decomposition  |  " + "   ".join(parts))
