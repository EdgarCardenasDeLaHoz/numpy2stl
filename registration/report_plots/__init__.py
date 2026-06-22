"""report_plots: PNG render functions for the registration report.

Split into stage-grouped submodules; this faade re-exports every
render_* function so `from ...report_plots import X` keeps working.
"""
from __future__ import annotations

from ._common import _render_single_heightmap, _imshow_heightmap
from .inputs import render_stl_heightmap_png, render_osm_heightmap_png, render_aligned_png
from .masks import render_binarization_png, render_mask_overlay_png, render_matched_buildings_png, render_footprint_rgchannel_png, render_vectorized_png, building_component_stats
from .registration import render_transform_summary_png, render_scale_sweep_png, render_xcorr_map_png, render_angle_histogram_png, render_rot_sweep_png
from .comparison import render_comparison_png, render_difference_hist_png, render_missing_analysis_png, render_corrected_difference_png
from .decimation import render_decimation_png, render_decimation_curve_png
from .prisms import render_prism_decomposition_png

__all__ = [
    "render_decimation_png",
    "render_decimation_curve_png",
    "render_prism_decomposition_png",
    "render_corrected_difference_png",
    "render_vectorized_png",
    "_render_single_heightmap",
    "_imshow_heightmap",
    "render_stl_heightmap_png",
    "render_osm_heightmap_png",
    "render_aligned_png",
    "render_binarization_png",
    "render_mask_overlay_png",
    "render_matched_buildings_png",
    "render_footprint_rgchannel_png",
    "building_component_stats",
    "render_transform_summary_png",
    "render_scale_sweep_png",
    "render_xcorr_map_png",
    "render_angle_histogram_png",
    "render_rot_sweep_png",
    "render_comparison_png",
    "render_difference_hist_png",
    "render_missing_analysis_png",
]
