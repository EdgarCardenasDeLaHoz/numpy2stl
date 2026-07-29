"""HTML report assembly for the registration pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Report stylesheet lives in an external template file so the CSS can be edited
# without wading through Python, and so editors can syntax-highlight it.
_CSS = (Path(__file__).parent / "templates" / "report.css").read_text(encoding="utf-8")

_CONFIDENCE_THRESHOLD_GOOD = 0.5
_CONFIDENCE_THRESHOLD_WARN = 0.2


def write_registration_report(out_dir: str | Path, report) -> Path:
    from .report_plots import (
        render_angle_histogram_png,
        render_binarization_png,
        render_vectorized_png,
        render_xcorr_map_png,
        render_comparison_png,
        render_corrected_difference_png,
        render_difference_hist_png,
        render_footprint_rgchannel_png,
        render_mask_overlay_png,
        render_matched_buildings_png,
        render_missing_analysis_png,
        render_osm_heightmap_png,
        render_rot_sweep_png,
        render_scale_sweep_png,
        render_stl_heightmap_png,
        render_aligned_png,
        render_transform_summary_png,
        render_decimation_png,
        render_decimation_curve_png,
        render_prism_decomposition_png,
    )

    out_dir = Path(out_dir)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing registration report to %s", out_dir)

    asset_paths = {}
    asset_paths["stl_heightmap"]     = render_stl_heightmap_png(assets_dir / "stl_heightmap.png", report.stl_heightmap)
    asset_paths["osm_heightmap"]     = render_osm_heightmap_png(assets_dir / "osm_heightmap.png", report.osm_heightmap)
    asset_paths["stl_aligned"]       = render_aligned_png(assets_dir / "stl_aligned.png", report.stl_aligned, report.osm_heightmap)
    asset_paths["comparison"]        = render_comparison_png(assets_dir / "comparison.png", report)
    asset_paths["corrected_difference"] = render_corrected_difference_png(assets_dir / "corrected_difference.png", report)
    asset_paths["binarization"]      = render_binarization_png(assets_dir / "binarization.png", report)
    asset_paths["mask_overlay"]      = render_mask_overlay_png(assets_dir / "mask_overlay.png", report)
    asset_paths["footprint_rgchannel"] = render_footprint_rgchannel_png(assets_dir / "footprint_rgchannel.png", report)
    asset_paths["matched_buildings"] = render_matched_buildings_png(assets_dir / "matched_buildings.png", report)
    asset_paths["difference_hist"]   = render_difference_hist_png(assets_dir / "difference_hist.png", report.comparison)
    asset_paths["missing_analysis"]  = render_missing_analysis_png(assets_dir / "missing_analysis.png", report.comparison)
    asset_paths["transform_summary"] = render_transform_summary_png(assets_dir / "transform_summary.png", report.registration)
    asset_paths["scale_sweep"] = render_scale_sweep_png(
        assets_dir / "scale_sweep.png", list(report.scale_sweep),
        known_scale=report.known_scale, found_scale=report.registration.scale,
        area_scale=getattr(report, "area_scale", None),
        fourier_scale=getattr(report, "fourier_scale", None))
    asset_paths["rot_sweep"] = render_rot_sweep_png(
        assets_dir / "rot_sweep.png",
        list(getattr(report, "rot_sweep", [])),
        found_rot=report.registration.angle_deg,
        hist_rot_deg=getattr(report, "_hist_rot_deg", None),
        rot_l1_xcorr=getattr(report, "_rot_l1_xcorr", None),
        rot_l2_xcorr=getattr(report, "_rot_l2_xcorr", None),
        dice_fine=getattr(report, "_dice_fine", None),
    )

    _hs = getattr(report, "_hist_src", None)
    _ht = getattr(report, "_hist_tgt", None)
    _hx = getattr(report, "_hist_xcorr", None)
    _hr = getattr(report, "_hist_rot_deg", 0.0)
    # Hough detected-lines / pipeline figures removed — diagnostic-only (rotation
    # is solved from the image gradient, not these lines).
    asset_paths["vectorized"] = render_vectorized_png(
        assets_dir / "vectorized.png", report)

    # Simplification-evaluation figures (only when the mesh was simplified).
    if getattr(report, "_prism_stats", None) is not None:
        asset_paths["prism"] = render_prism_decomposition_png(
            assets_dir / "prism.png", report)
    elif getattr(report, "_stl_heightmap_original", None) is not None:
        asset_paths["decimation"] = render_decimation_png(
            assets_dir / "decimation.png", report)
        if getattr(report, "_decimation_sweep", None):
            asset_paths["decimation_curve"] = render_decimation_curve_png(
                assets_dir / "decimation_curve.png", report)

    asset_paths["xcorr_map"] = render_xcorr_map_png(
        assets_dir / "xcorr_map.png",
        getattr(report, "_xcorr_map", None),
        getattr(report, "_best_dx", 0.0),
        getattr(report, "_best_dy", 0.0),
    )

    asset_paths["angle_hist"] = render_angle_histogram_png(
        assets_dir / "angle_hist.png",
        _hs if _hs is not None else [],
        _ht if _ht is not None else [],
        _hx if _hx is not None else [],
        _hr,
        found_rot=report.registration.angle_deg,
    ) if (_hs is not None and len(_hs) > 0) else None

    html = render_registration_index(report, asset_paths)
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    logger.info("Report written: %s", index_path)
    return index_path


def render_registration_index(report, asset_paths: dict) -> str:
    reg  = report.registration
    comp = report.comparison

    # --- Alignment quality metrics ---
    try:
        from .align import score_alignment
        import numpy as _np
        sc = score_alignment(report.stl_aligned, report.osm_heightmap,
                             _np.array([[1,0,0],[0,1,0]], dtype=float),
                             cell_size_m=getattr(report, "cell_size_m", None))
        edge_iou   = sc["edge_iou"]
        edge_lift  = sc["edge_lift"]
        edge_base  = sc["edge_baseline"]
        foot_iou   = sc["footprint_iou"]
        foot_lift  = sc["footprint_lift"]
        ov_iou     = sc["overlap_iou"]
        ov_prec    = sc["overlap_precision"]
        ov_dice    = sc["overlap_dice"]
    except Exception:
        edge_iou = edge_lift = edge_base = foot_iou = foot_lift = float("nan")
        ov_iou = ov_prec = ov_dice = float("nan")

    # Quality badge — driven by the FILLED footprint-overlap IoU (cropped to the STL
    # extent: the number the footprint_rgchannel.png overlay shows), combined with
    # overlap precision (fraction of the STL model's footprint OSM confirms).  The
    # old gate used edge-IoU lift over random, which is so sparse that a visually
    # broken, mostly-red overlay (e.g. Salzburg, wrong-quadrant −90° flip) still
    # cleared it.  Two factors are required because filled-IoU alone cannot separate
    # a broken city from a good one when both read a similar absolute IoU: precision
    # collapses on the broken fit (large STL-only regions OSM never confirms) while a
    # genuine fit keeps it high.  Calibrated on the 8-city set (Salzburg iou 0.35 /
    # prec 0.45 fails; the next-worst genuine city Bilbao iou 0.365 / prec 0.55 passes).
    if ov_iou >= 0.36 and ov_prec >= 0.50:
        badge_cls, badge_txt = "badge-good", "&#x2705; Genuine match"
        quality_summary = ("The filled building footprints overlap strongly and most of the model's "
                           "footprint is confirmed by OSM — the registration found a real geometric correspondence.")
    elif ov_iou >= 0.28 and ov_prec >= 0.40:
        badge_cls, badge_txt = "badge-warn", "&#x26A0;&#xFE0F; Weak match"
        quality_summary = ("Partial footprint overlap, but below the confident-match threshold "
                           "(filled IoU &ge; 0.36 and precision &ge; 0.50). A large share of the model's "
                           "footprint is unconfirmed by OSM &mdash; inspect the footprint overlay below for misalignment.")
    else:
        badge_cls, badge_txt = "badge-bad", "&#x274C; No match"
        quality_summary = "Filled footprint overlap is near chance. The registration did not find a reliable alignment."

    iou_cls  = "good" if (ov_iou >= 0.36 and ov_prec >= 0.50) else "warn" if (ov_iou >= 0.28 and ov_prec >= 0.40) else "bad"

    # Landmark check block
    lm = getattr(report, "landmark_check", None)
    if lm:
        lm_px   = lm.get("osm_dist_from_center", 0)
        lm_h    = lm.get("osm_mean_h", 0)
        stl_d   = lm.get("stl_dist_from_center", 0)
        lm_html = f"""
<div class="landmark-box">
  <strong>Central landmark check</strong>
  (largest tall building within 80 px of OSM centre &mdash; proxy for City Hall or equivalent)<br>
  <span class="lm-val">OSM location: row&nbsp;{lm.get('osm_row',0):.0f}, col&nbsp;{lm.get('osm_col',0):.0f}
    &nbsp;({lm_px:.0f}&nbsp;px from bbox centre, mean&nbsp;height&nbsp;{lm_h:.0f}&nbsp;m)</span><br>
  <span class="lm-val">Maps to STL: row&nbsp;{lm.get('stl_row',0):.0f}, col&nbsp;{lm.get('stl_col',0):.0f}
    &nbsp;({stl_d:.0f}&nbsp;px from STL centre)</span><br>
  <span class="muted">For a well-centred model, STL distance from centre should be small (&lt;40&nbsp;px = &lt;175&nbsp;m).</span>
</div>"""
    else:
        lm_html = "<p class='muted'>Landmark check not available.</p>"

    # Scale sweep
    scale_img = ('<img class="wide-img" style="max-width:800px" src="assets/scale_sweep.png" alt="Scale sweep" />'
                 if asset_paths.get("scale_sweep") else "<p class='muted'>Scale sweep not available.</p>")

    timing_rows    = _render_timing_rows(report.step_timings)
    timing_bar_html = _render_timing_bars(report.step_timings)

    anchor_note = (f"Physical anchor: <code>1 / osm_margin = 1/1.5 = {report.known_scale:.3f}</code> "
                   f"&mdash; the OSM frame is 1.5&times; the STL footprint so a building spans "
                   f"exactly {report.known_scale:.3f}&times; as many OSM pixels as STL pixels."
                   if report.known_scale else "Scale was searched (no physical anchor).")
    scale_anchor_note = (f"(<code>1 / osm_margin = {report.known_scale:.3f}</code> when known)."
                         if report.known_scale else "(No physical anchor).")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_esc(report.region_name)} — Registration report</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">

<!-- ================================================================ HEADER -->
<h1>
  {_esc(report.region_name)}
  <span class="quality-badge {badge_cls}">{badge_txt}</span>
</h1>
<p class="muted">STL: <code>{_esc(report.stl_file)}</code></p>
<p style="margin-top:8px">{quality_summary}</p>

<!-- =========================================================== HOW IT WORKS -->
<details class="rsec" open>
  <summary>How the registration works</summary>
  <p>Both the 3D model and the OpenStreetMap data are converted to 2D building
     presence masks, then their outlines are compared. Heights are not used &mdash;
     they are too noisy to register directly. The outline patterns are distinctive
     enough to find the correct rotation, scale, and translation.</p>
  <div class="process-steps">
    <div class="step"><span class="step-num">🗺</span><span class="step-label">STL → heightmap</span>Project 3D mesh from above</div>
    <div class="step"><span class="step-num">🏢</span><span class="step-label">Detect buildings</span>Top-hat filter removes terrain; threshold gives building mask</div>
    <div class="step"><span class="step-num">✏️</span><span class="step-label">Extract edges</span>1-px outline of each building footprint — sparse, distinctive signal</div>
    <div class="step"><span class="step-num">🔍</span><span class="step-label">FFT global search</span>FFT cross-correlation finds best translation at each rotation; L0/L1/L2 sweep</div>
    <div class="step"><span class="step-num">🎯</span><span class="step-label">ECC refinement</span>ECC on building distance fields fine-tunes the alignment</div>
    <div class="step"><span class="step-num">📐</span><span class="step-label">Compare</span>Building-to-building height statistics over the overlap</div>
  </div>
</details>

{_decimation_section(report, asset_paths)}

<!-- ========================================================= BUILDING DETECTION -->
<details class="rsec" open>
  <summary>Building detection &amp; registration search</summary>
  <p>Each heightmap is converted to a binary building mask, then reduced to its
     1-pixel outline (edges). The STL mask uses a morphological top-hat to separate
     height-above-local-terrain from the base plate and terrain variation.
     The OSM mask is any pixel with a rasterized building footprint.
     Coverage is matched so both datasets have similar edge density.</p>
  <img class="wide-img" src="assets/binarization.png" alt="Binarization panel" />

  <h3 style="margin-top:20px">Building component statistics</h3>
  {_render_component_stats_table(report)}

  <h3 style="margin-top:20px">Transform parameters</h3>
  <p class="muted">Monospace summary of all numeric registration outputs — confidence,
     scale, rotation, and translation as applied.</p>
  {'<img class="wide-img" style="max-width:560px" src="assets/transform_summary.png" alt="Transform summary" />'
   if asset_paths.get("transform_summary") else "<p class='muted'>Transform summary not available.</p>"}

  <h3 style="margin-top:20px">Rotation from gradient-orientation histograms</h3>
  <p>Rotation is solved from the <em>image gradient</em> (Sobel on the terrain-removed
     height field), not from segmentation — wall orientations show as peaks in the
     gradient-angle histogram.  The STL→OSM rotation is the circular shift that aligns
     the two histograms (1-D FFT cross-correlation), which is <em>translation- and
     scale-invariant</em>.  The 90° grid ambiguity is resolved with a bias toward 0°,
     overridden only when a rotated orientation correlates clearly better in height
     (or a manual rotation is supplied).</p>
  {'<img class="wide-img" src="assets/angle_hist.png" alt="Gradient angle histograms" />'
   if asset_paths.get("angle_hist") else "<p class='muted'>Angle histogram not available.</p>"}

  <h3 style="margin-top:20px">Vectorized footprints</h3>
  <p>Building masks traced into simplified polygons (contour + Douglas&ndash;Peucker) &mdash;
     the same polygon form OSM uses, so STL footprints are directly comparable and exportable.</p>
  {'<img class="wide-img" src="assets/vectorized.png" alt="Vectorized footprints" />'
   if asset_paths.get("vectorized") else "<p class='muted'>Vectorization not available.</p>"}

  <h3 style="margin-top:20px">Translation cross-correlation map</h3>
  <p>2-D FFT cross-correlation at the best rotation.  Each pixel is the edge
     overlap score if the STL were shifted by (dx, dy).  The bright spot is the
     found translation; a compact isolated peak means a reliable result.</p>
  {'<img class="wide-img" style="max-width:600px" src="assets/xcorr_map.png" alt="XCorr map" />'
   if asset_paths.get("xcorr_map") else "<p class='muted'>Cross-correlation map not available.</p>"}

  <h3 style="margin-top:20px">Rotation sweep — all stages</h3>
  <p>Left: full ±12° survey at 1° resolution showing Dice, Edge IoU, and heightmap
     cross-correlation.  Right: zoomed ±6° view overlaying the L1 xcorr (0.25° step),
     L2 xcorr (0.05° step), and Dice refinement (0.25° step) so the sub-degree
     selection is visible.  Purple dotted line = histogram estimate; green dashed
     line = final chosen rotation.  Each right-panel series is normalised
     independently to [0,1] for comparability.</p>
  {'<img class="wide-img" src="assets/rot_sweep.png" alt="Rotation sweep" />'
   if asset_paths.get("rot_sweep") else "<p class='muted'>Rotation sweep not available.</p>"}

  <h3 style="margin-top:20px">Scale determination</h3>
  <p>Edge-mask IoU vs spatial scale at the best rotation.
     The red dashed line is the physical anchor
     {scale_anchor_note}
     The peak should coincide with the anchor when the model is correctly sized.</p>
  {scale_img}
</details>

<!-- ========================================================= REGISTRATION RESULT -->
<details class="rsec" open>
  <summary>Registration result</summary>
  <table style="margin-bottom:12px">
    <tr><th>Scale</th>
        <td>{reg.scale:.5f}&times;
            <span class="muted">&nbsp;({anchor_note})</span></td></tr>
    <tr><th>Rotation</th><td>{reg.angle_deg:.2f}&deg;</td></tr>
    <tr><th>Translation (tx&nbsp;/&nbsp;ty)</th>
        <td>{reg.transform[0,2]:.1f}&nbsp;/&nbsp;{reg.transform[1,2]:.1f}&nbsp;px</td></tr>
    <tr><th>Filled footprint-overlap IoU</th>
        <td class="{iou_cls}"><b>{ov_iou:.3f}</b>
            &nbsp;<span class="muted">precision {ov_prec:.2f} (share of STL footprint OSM confirms),
            Dice {ov_dice:.3f}
            &mdash; the gate metric (cropped to the STL extent; matches the footprint overlay).
            {'&#x2705; genuine' if (ov_iou >= 0.36 and ov_prec >= 0.50) else '&#x26A0;&#xFE0F; weak' if (ov_iou >= 0.28 and ov_prec >= 0.40) else '&#x274C; no match'}
            </span></td></tr>
    <tr><th>Edge IoU</th>
        <td><span class="muted"><b>{edge_iou:.3f}</b> vs random baseline {edge_base:.3f}
            &mdash; lift&nbsp;<b>{edge_lift:.1f}&times;</b>
            &mdash; 1-px outlines; too sparse to gate on (kept for reference)</span></td></tr>
    <tr><th>Filled-mask IoU (full frame)</th>
        <td><span class="muted">{foot_iou:.3f} (lift {foot_lift:.1f}&times;)
            &mdash; over the whole frame; diluted by the empty margin around the STL</span></td></tr>
    <tr><th>Height scale (Z)</th>
        <td>{comp.height_scale_used:.4f}&times; STL units &rarr; metres
            <span class="muted">&nbsp;(auto-estimated from building overlap)</span></td></tr>
  </table>
</details>

<!-- ========================================================= ALIGNMENT EVIDENCE -->
<details class="rsec" open>
  <summary>Alignment evidence &mdash; footprint overlay</summary>
  <p>The primary alignment check: where do the two building footprint masks agree?
     <b style="color:#cc0">Yellow</b> = both datasets place a building here (good).
     <b style="color:#c00">Red</b> = STL model has a building, OSM does not.
     <b style="color:#090">Green</b> = OSM has a building, STL does not.
     Dense yellow means genuine alignment.</p>
  <img class="wide-img" src="assets/footprint_rgchannel.png" alt="Footprint overlay" />

  <h3 style="margin-top:20px">Footprint agreement map</h3>
  <p><b style="color:#2a7a2a">Green</b> = both agree (building present in STL and OSM).
     <b style="color:orange">Orange</b> = STL only.
     <b style="color:#3070d8">Blue</b> = OSM only.
     Orange and blue are expected at boundaries; large orange regions inside the
     model extent indicate misalignment or buildings missing from OSM.</p>
  <img class="wide-img" style="max-width:760px" src="assets/mask_overlay.png" alt="Footprint agreement" />

  <h3 style="margin-top:20px">Landmark check</h3>
  {lm_html}
</details>

<!-- ========================================================= HEIGHT COMPARISON -->
<details class="rsec" open>
  <summary>Height comparison</summary>
  <div class="caveat">
    <strong>Heights are compared as height-above-terrain.</strong>
    The STL stores absolute elevation; a morphological top-hat removes terrain so the
    residual is each building&rsquo;s height above local ground, fit to OSM metres by an
    affine fit (<code>stl_m = scale&middot;stl + offset</code>).  The offset absorbs a
    constant base-plate bias.  Pearson <em>r</em> / Spearman &rho; (segmentation-independent)
    are the most reliable height-agreement signals; RMSE/MAPE are sensitive to the per-building
    height variance that remains after the single-scalar fit.
  </div>

  <h3>3-panel comparison (STL aligned | OSM | difference)</h3>
  <p class="muted">Both panels are cropped to the STL&rsquo;s valid extent. STL heights
     are shown scaled by the auto-estimated Z factor ({comp.height_scale_used:.2f}&times;).
     The difference panel shows where heights agree (white) and where they differ (red/blue).</p>
  <img class="wide-img" src="assets/comparison.png" alt="3-panel comparison" />

  <h3 style="margin-top:20px">Corrected per-building difference</h3>
  <p class="muted">One difference value per building footprint (median STL height-above-terrain
     vs OSM), with OSM fill-defaults and fit-outliers excluded and the affine height fit applied.
     This isolates the buildings that genuinely match &mdash; the histogram should centre near zero.</p>
  {'<img class="wide-img" src="assets/corrected_difference.png" alt="Corrected per-building difference" />'
   if asset_paths.get("corrected_difference") else "<p class='muted'>Corrected difference not available.</p>"}

  <table style="margin-top:12px">
    <tr><th>Building-to-building RMSE</th><td>{comp.rmse:.1f}&nbsp;m
        <span class="muted"> (limited by terrain; not reliable)</span></td></tr>
    <tr><th>Bias</th>
        <td class="{'bad' if abs(comp.bias)>10 else 'warn' if abs(comp.bias)>4 else 'good'}">{comp.bias:+.1f}&nbsp;m
        <span class="muted">&nbsp;positive = STL reports taller than OSM</span></td></tr>
    <tr><th>Height correlation (Pearson r)</th>
        <td class="{'good' if comp.correlation>0.5 else 'warn' if comp.correlation>0.2 else 'bad'}">{comp.correlation:.3f}
        <span class="muted">&nbsp;(height-above-terrain vs OSM height-above-ground)</span></td></tr>
    <tr><th>Rank correlation (Spearman &rho;)</th>
        <td class="{'good' if comp.rank_correlation>0.5 else 'warn' if comp.rank_correlation>0.2 else 'bad'}">{comp.rank_correlation:.3f}
        <span class="muted">&nbsp;robust to outliers; measures relative height ranking</span></td></tr>
    <tr><th>Median height error (MAPE)</th>
        <td class="{'good' if comp.mape<25 else 'warn' if comp.mape<50 else 'bad'}">{comp.mape:.1f}%</td></tr>
    <tr><th>Height ratio (STL/OSM)</th>
        <td>{comp.height_ratio_mean:.3f} &plusmn; {comp.height_ratio_std:.3f}
        <span class="muted">&nbsp;1.0 = perfect scale</span></td></tr>
    <tr><th>Footprint IoU (registration quality)</th>
        <td class="{'good' if comp.footprint_iou>0.12 else 'warn' if comp.footprint_iou>0.06 else 'bad'}">{comp.footprint_iou:.3f}
        <span class="muted">&nbsp;union overlap of STL vs OSM footprints (Dice {comp.dice_score:.3f})</span></td></tr>
    <tr><th>Building pixel overlap</th>
        <td>{comp.n_overlap:,}&nbsp;px &nbsp;({comp.coverage_pct:.1f}% of OSM building pixels)</td></tr>
  </table>

  <div class="figures" style="margin-top:16px">
    <figure>
      <img src="assets/matched_buildings.png" alt="Matched buildings" width="800" />
      <figcaption>Per-building height scatter (left) and error map (right) for OSM footprints
        that overlap the STL.  A good result clusters points on the dashed y=x line.</figcaption>
    </figure>
    <figure>
      <img src="assets/difference_hist.png" alt="Height difference histogram" width="560" />
      <figcaption>Distribution of STL&nbsp;&minus;&nbsp;OSM height differences over matched
        building pixels. Centred near zero = good bias; narrow = good RMSE.</figcaption>
    </figure>
    <figure>
      <img src="assets/missing_analysis.png" alt="Missing coverage" width="700" />
      <figcaption>Left: buildings in STL with no OSM footprint (new construction or OSM gaps).
        Right: OSM buildings outside the STL extent or not captured by the model.</figcaption>
    </figure>
  </div>

  <h3 style="margin-top:20px">Individual heightmaps</h3>
  <div class="figures">
    <figure>
      <img src="assets/stl_heightmap.png" alt="STL heightmap" width="380" />
      <figcaption>STL projection (model units, before registration). Gray = no mesh data.</figcaption>
    </figure>
    <figure>
      <img src="assets/osm_heightmap.png" alt="OSM heightmap" width="380" />
      <figcaption>OSM building heights (metres). White = no building footprint.</figcaption>
    </figure>
    <figure>
      <img src="assets/stl_aligned.png" alt="STL aligned vs OSM" width="700" />
      <figcaption>STL warped into OSM pixel space (left) vs OSM (right) &mdash; same coordinate frame.</figcaption>
    </figure>
  </div>
</details>

<!-- ============================================================== TIMING -->
<details class="rsec" open>
  <summary>Pipeline timing</summary>
  <table class="timing-table">
    <tr><th>Step</th><th>Duration (s)</th><th></th></tr>
    {timing_rows}
  </table>
  {timing_bar_html}
</details>

<footer>Generated by numpy2stl.registration</footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return (str(s).replace("&","&amp;").replace("<","&lt;")
            .replace(">","&gt;").replace('"',"&quot;"))


def _prism_section(report, asset_paths: dict) -> str:
    """HTML for the prism-decomposition section (empty unless simplify_mode='prism')."""
    st = getattr(report, "_prism_stats", None)
    if st is None or asset_paths.get("prism") is None:
        return ""
    haus = st.get("hausdorff_m", float("nan"))
    budget = st.get("deviation_tol_m_metres", float("nan"))
    return f'''
<!-- ========================================================= PRISM DECOMPOSITION -->
<details class="rsec" open>
  <summary>Prism decomposition (STL as a sum of prisms)</summary>
  <p>The STL is re-expressed as a <strong>sum of extruded prisms</strong> &mdash; the reverse of
     how OSM footprints are rendered into a model.  Each building is segmented and built as a
     <em>wedding-cake stack</em> of nested prisms (one footprint per height layer, spaced by the
     deviation budget), and the topmost roof is capped by a fitted plane (a slanted top) where it
     is planar within budget.  This makes the STL the same kind of geometry as an OSM-extruded
     model, with footprints clean by construction.  Model scale: {st.get('m_per_unit', 1.0):.2f} m per unit.</p>
  <table>
    <tr><th>Buildings</th><td>{st.get('n_buildings', 0)}</td></tr>
    <tr><th>Prisms</th><td>{st.get('n_prisms', 0)} (mean {st.get('mean_layers', 0):.1f} layers/building)</td></tr>
    <tr><th>Sloped (planar) caps</th><td>{st.get('sloped_caps', 0)}</td></tr>
    <tr><th>Max surface deviation</th><td>{haus:.2f} m (budget {budget:.2f} m)</td></tr>
  </table>
  <h3 style="margin-top:20px">Original vs prism model</h3>
  <p>Left: the rasterized original heightmap.  Middle: the prism model (sum of extruded prisms).
     Right: their difference in metres &mdash; small, footprint-aligned residuals confirm the
     building massing is captured by stacked prisms.</p>
  <img class="wide-img" src="assets/prism.png" alt="Prism decomposition before/after/difference" />
</details>'''


def _decimation_section(report, asset_paths: dict) -> str:
    """HTML for the mesh-simplification evaluation section (empty when the mesh
    was not simplified)."""
    if getattr(report, "_prism_stats", None) is not None:
        return _prism_section(report, asset_paths)
    st = getattr(report, "_simplify_stats", None)
    if st is None or asset_paths.get("decimation") is None:
        return ""
    f0 = st.get("orig_faces", 0); f1 = st.get("simplified_faces", 0)
    pct = 100.0 * f1 / f0 if f0 else 0.0
    haus_m = st.get("hausdorff_m", float("nan"))
    budget = st.get("deviation_tol_m_metres", float("nan"))
    mpu = st.get("m_per_unit", 1.0)
    curve = (f'''<h3 style="margin-top:20px">Decimation trade-off curve</h3>
  <p>Surface deviation (Hausdorff, metres) achievable at each face-reduction level,
     with the chosen deviation budget and operating point marked &mdash; use it to
     judge how aggressive the budget can be before the building shape drifts.</p>
  <img class="wide-img" style="max-width:760px" src="assets/decimation_curve.png" alt="Decimation curve" />'''
             if asset_paths.get("decimation_curve") else "")
    return f'''
<!-- ========================================================= MESH SIMPLIFICATION -->
<details class="rsec" open>
  <summary>Mesh simplification (decimation)</summary>
  <p>Before segmentation the mesh is <strong>decimated as aggressively as possible
     while its surface stays within a metres deviation budget</strong> (feature-preserving
     quadric edge collapse, binary-searched against the symmetric Hausdorff distance to the
     original).  This strips triangle density and sub-budget roof clutter while keeping each
     building's height and footprint, so the rasterized heightmap segments into cleaner
     footprints.  Model scale: {mpu:.2f} m per mesh unit.</p>
  <table>
    <tr><th>Faces</th><td>{f0:,} &rarr; {f1:,} ({pct:.0f}% kept)</td></tr>
    <tr><th>Max surface deviation</th><td>{haus_m:.2f} m (budget {budget:.2f} m)</td></tr>
  </table>
  <h3 style="margin-top:20px">Original vs simplified heightmap</h3>
  <p>The left two panels are the rasterized heightmaps before and after decimation; the right
     panel is their difference in metres.  A difference concentrated on rooftops with a clean,
     near-zero footprint edge confirms detail was removed without moving the building outline.</p>
  <img class="wide-img" src="assets/decimation.png" alt="Decimation before/after/difference" />
  {curve}
</details>'''


def _render_component_stats_table(report) -> str:
    """
    Compute and render a building-component area comparison table for STL vs OSM.

    OSM cell size is derived from the known_scale anchor and the STL footprint bounds.
    STL cell size is derived directly from the STL heightmap bounds.
    """
    try:
        from .align import building_mask
        from .report_plots import building_component_stats

        stl_mask = building_mask(report.stl_heightmap, source="stl",
                                  cell_size_m=getattr(report, "cell_size_m", None))
        osm_mask = building_mask(report.osm_heightmap, source="osm")

        # Cell sizes in metres:
        # STL: the heightmap covers the full mesh XY bounding box at its resolution
        #   cell_m_stl = stl_footprint_m / resolution
        # OSM: the bbox is stl_footprint_m * (1/known_scale) per side
        #   cell_m_osm = osm_footprint_m / resolution
        h, w = report.stl_heightmap.shape
        # Recover STL footprint from the tallest-building scale anchor in cities.py
        # We don't have it directly, but we can estimate from known_scale:
        #   osm_bbox_m = stl_footprint_m / known_scale
        # Without stl_footprint_m we can't compute absolute m², so mark as unavailable
        # unless we have it from report metadata (currently not stored).
        # Fallback: compute px² only and add a note.
        stl_stats = building_component_stats(stl_mask)
        osm_stats = building_component_stats(osm_mask)

        def _fmt(s, label):
            if s["count"] == 0:
                return f"<tr><th>{label}</th><td colspan='5'>No buildings detected</td></tr>"
            return (
                f"<tr><th>{label}</th>"
                f"<td>{s['count']:,}</td>"
                f"<td>{s['mean_px']:.0f}</td>"
                f"<td>{s['median_px']:.0f}</td>"
                f"<td>{s['p25_px']:.0f} – {s['p75_px']:.0f}</td>"
                f"<td>{s['total_px']:,}</td>"
                f"</tr>"
            )

        ratio_mean = (stl_stats["mean_px"] / osm_stats["mean_px"]
                      if osm_stats["mean_px"] > 0 else float("nan"))
        ratio_note = (f"STL mean is {ratio_mean:.2f}× OSM mean. "
                      f"{'Larger than expected — may include terrain blobs.'  if ratio_mean > 1.5 else ''}"
                      f"{'Smaller than expected — threshold may be too high.' if ratio_mean < 0.5 else ''}"
                      if not (ratio_mean != ratio_mean) else "")

        return f"""
<p class="muted">Component areas are in pixels² (px²). To convert to m²,
multiply by the cell area: OSM ≈ 18.3 m²/px (4.28 m/px) at 512-px 2.2 km frame;
STL ≈ 8.1 m²/px (2.85 m/px) at 512-px 1.46 km model.</p>
<table>
  <tr><th>Dataset</th><th>Buildings</th><th>Mean (px²)</th>
      <th>Median (px²)</th><th>IQR (px²)</th><th>Total area (px²)</th></tr>
  {_fmt(stl_stats, "STL")}
  {_fmt(osm_stats, "OSM")}
</table>
{'<p class="muted" style="margin-top:6px">' + ratio_note + '</p>' if ratio_note else ''}"""
    except Exception as exc:
        return f"<p class='muted'>Component stats unavailable: {exc}</p>"


def _render_timing_rows(step_timings: list, hide_below_pct: float = 5.0) -> str:
    if not step_timings:
        return "<tr><td colspan='3'>No timing data</td></tr>"
    total = sum(t for name, t in step_timings if not name.startswith("↳"))
    rows = []
    hidden_t = 0.0
    hidden_n = 0
    for name, t in step_timings:
        is_sub = name.startswith("↳")
        pct = 100.0 * t / total if total > 0 else 0
        bar_w = int(pct * 1.5)
        # Hide small steps (top-level and sub-steps) to keep the table readable;
        # their time is rolled into an "(other small steps)" row.
        if pct < hide_below_pct:
            hidden_t += t if not is_sub else 0.0
            hidden_n += 1 if not is_sub else 0
            continue
        if is_sub:
            rows.append(
                f"<tr style='font-size:12px;color:#888'>"
                f"<td style='padding-left:28px'>{_esc(name)}</td>"
                f"<td>{t:.2f}</td>"
                f"<td><span class='timing-bar' style='width:{bar_w}px;background:#85b4d9'></span>"
                f"&nbsp;{pct:.0f}%</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{_esc(name)}</td><td>{t:.2f}</td>"
                f"<td><span class='timing-bar' style='width:{bar_w}px'></span>"
                f"&nbsp;{pct:.0f}%</td></tr>"
            )
    if hidden_n > 0:
        hpct = 100.0 * hidden_t / total if total > 0 else 0
        rows.append(
            f"<tr style='color:#aaa'><td><em>(other small steps &lt;{hide_below_pct:.0f}% each, "
            f"n={hidden_n})</em></td><td>{hidden_t:.2f}</td><td>&nbsp;{hpct:.0f}%</td></tr>"
        )
    rows.append(f"<tr><th>Total</th><th>{total:.2f}</th><th></th></tr>")
    return "\n    ".join(rows)


def _render_timing_bars(step_timings: list) -> str:
    if not step_timings:
        return ""
    top_level = [(n, t) for n, t in step_timings if not n.startswith("↳")]
    sub_level  = [(n, t) for n, t in step_timings if n.startswith("↳")]
    top_sorted = sorted(top_level, key=lambda x: x[1], reverse=True)[:8]
    if sub_level:
        combined, inserted = [], False
        for item in top_sorted:
            combined.append(item)
            if not inserted and item == max(top_sorted, key=lambda x: x[1]):
                for s in sub_level:
                    combined.append(s)
                inserted = True
        entries = combined
    else:
        entries = top_sorted

    all_t  = [t for _, t in entries]
    max_t  = max(all_t) if all_t else 1
    bar_max_w = 280
    rows = []
    for name, t in entries:
        is_sub = name.startswith("↳")
        w = max(2, int(t / max_t * bar_max_w))
        color  = "#85b4d9" if is_sub else "#3a7ebf"
        indent = 16 if is_sub else 0
        rows.append(
            f'<tr><td style="font-size:12px;padding:2px 6px;padding-left:{6+indent}px;'
            f'{"color:#999" if is_sub else ""}">{_esc(name)}</td>'
            f'<td><svg width="{bar_max_w}" height="16">'
            f'<rect x="{indent}" y="1" width="{max(2,w-indent)}" height="14" fill="{color}"/></svg>'
            f'&nbsp;<span style="font-size:11px">{t:.2f}s</span></td></tr>'
        )
    return f'<table style="margin-top:8px;border:none">{"".join(rows)}</table>'
