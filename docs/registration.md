# numpy2stl — City STL Registration

Register a city 3D mesh against OpenStreetMap building data to find geographic
alignment and compare dataset quality.

> **Architecture & module map:** see
> [`registration/docs/ARCHITECTURE.md`](../registration/docs/ARCHITECTURE.md) for the
> subpackage layout, the pipeline stages, the key algorithm decisions, and the
> hardcode/assumption audit (how to run on a city not in the built-in config).

**Works on any city.** Cities not in the built-in config geocode their centre
automatically; pass `tallest_m=` or `scale_m_per_unit=` (CLI `--tallest-m` /
`--scale-m-per-unit`) for a tight, well-scaled OSM fetch, or `center=` / `--center
LAT,LON` to set the downtown point explicitly.

---

## Quick Start

```python
from numpy2stl.registration import register_city_stl
from numpy2stl.registration.html_report import write_registration_report

# Full pipeline + write HTML report to ./report/
report = register_city_stl(
    stl_file="philadelphia.stl",
    city_name="Philadelphia, PA, USA",
    resolution=512,
    height_scale=None,    # auto-estimate STL units → metres
    out_dir="./report",   # writes index.html + assets/
)

# report is a CityRegistrationReport dataclass
print(report.registration.confidence)   # raw FFT xcorr peak amplitude — an internal
                                         # search diagnostic, NOT a match-quality score
                                         # (no defined ceiling; see "Reading the metrics")
print(report.comparison.match_score)    # 0–1 composite match-quality score; the
                                         # number to actually look at (see below)
print(report.registration.scale)        # spatial scale factor found
print(report.registration.angle_deg)    # rotation found (degrees)
print(f"RMSE: {report.comparison.rmse:.1f} m")
print(f"Bias: {report.comparison.bias:+.1f} m  (+ = STL taller than OSM)")
print(f"Coverage: {report.comparison.coverage_pct:.1f}%")
```

---

## Step-by-Step Usage

```python
from numpy2stl.stl2numpy import mesh_to_heightmap
from numpy2stl.applications.cities import get_osm_building_heightmap
from numpy2stl.registration.align import register, apply_transform
from numpy2stl.registration.compare import compare

# 1. STL → 2D heightmap (arbitrary coordinates, unknown scale)
stl = mesh_to_heightmap("city.stl", resolution=512, projection="max")

# 2. OSM building heights for the same city
osm = get_osm_building_heightmap("Philadelphia, PA, USA", resolution=512)
# Or use a hardcoded bbox: get_osm_building_heightmap((40.06, 39.86, -74.95, -75.28))

# 3. Register: find translation + rotation + scale
reg = register(stl["heightmap"], osm["heightmap"], max_scale_ratio=5.0)
print(reg["transform"])          # 2×3 affine matrix
print(reg["confidence"])         # ECC correlation coefficient

# 4. Warp STL into OSM pixel space
stl_aligned = apply_transform(
    stl["heightmap"], reg["transform"],
    output_shape=osm["heightmap"].shape,
)

# 5. Compare heights (auto-estimate STL unit → metres scaling)
comp = compare(stl_aligned, osm["heightmap"], height_scale=None)
print(comp.rmse, comp.bias, comp.coverage_pct)

# 6. Write report manually
from numpy2stl.registration.html_report import write_registration_report
from numpy2stl.registration.types import (
    CityRegistrationReport, RegistrationResult, ComparisonResult
)
```

---

## Named Cities

```python
from numpy2stl.applications.cities import get_philadelphia_heightmap

osm = get_philadelphia_heightmap(resolution=512)
```

Adding more cities: copy the `get_philadelphia_heightmap` pattern with a
hardcoded `(N, S, E, W)` bbox, or use any city name string:

```python
from numpy2stl.applications.cities import get_osm_building_heightmap

osm = get_osm_building_heightmap("Seattle, WA, USA", resolution=512)
```

Currently named wrappers: `get_philadelphia_heightmap`, `get_new_york_heightmap`,
`get_chicago_heightmap`, `get_boston_heightmap`.

---

## Parameters Reference

### `register_city_stl(stl_file, city_name, ...)`

| Parameter | Default | Description |
|---|---|---|
| `stl_file` | — | Path to city STL (no geographic metadata required) |
| `city_name` | — | City name string or `(N, S, E, W)` bbox tuple |
| `resolution` | `1024` | Output grid size for the comparison/report heightmaps (square). The registration **search** is fixed at `REGISTER_RES=512` and resolution-independent, so raising this only sharpens the footprint/agreement images (more agreement pixels), without changing the transform or the search cost. |
| `default_height` | `10.0` | Fallback OSM building height (metres) |
| `levels_to_meters` | `3.5` | OSM `building:levels` × this = metres |
| `max_scale_ratio` | `5.0` | Maximum spatial scale search range |
| `height_scale` | `None` | STL model units → metres. `None` = auto-estimate. |
| `stl_z_axis` | `2` | Which mesh axis is elevation (default 2 = Z) |
| `out_dir` | `None` | If given, writes HTML report here |
| `simplify_mesh` | `False` | Alias for `simplify_mode="decimate"`. |
| `simplify_mode` | `"off"` | `"decimate"` = footprint-preserving quadric decimation; `"prism"` = decompose the STL into a sum of extruded prisms (reverse of the OSM render) for the comparison. Registration always uses the original mesh. |
| `simplify_tol_m` | `3.5` | Surface-deviation budget (metres) for `simplify_mesh`. Larger = more detail removed. |
| `save_simplified` | `None` | Write the simplified mesh here (STL/3MF/OBJ by extension). |
| `regularize_footprints` | `False` | Watershed-split merged buildings + snap footprint polygons to rectilinear edges. |
| `refine_polygons` | `False` | Polygon-matched ICP fine-tuning of the transform (only when footprint Dice > 0.95). |
| `decimation_curve` | `False` | Add the decimation trade-off curve (deviation vs faces kept) to the report. Re-decimates a few times; adds ~30 s. |
| `registration_method` | `"raster"` | `"raster"` (gradient + xcorr + edge-IoU) or `"polygon"` (match building footprints directly; auto-falls back to raster when match confidence is low). |
| `free_scale` | `False` | Un-lock scale: refine within ±10% of the geometric anchor. The anchor is geometrically exact, so this usually drifts — off by default. |

### `register(source, target, ...)`

| Parameter | Default | Description |
|---|---|---|
| `max_scale_ratio` | `5.0` | Maximum spatial scale search range (source may be up to 5× larger or smaller than target) |
| `ecc_motion_type` | `MOTION_AFFINE` | OpenCV motion model. AFFINE handles scale + rotation + translation. |
| `ecc_iterations` | `1000` | ECC maximum iterations |
| `ecc_eps` | `1e-6` | ECC convergence threshold |
| `gaussian_blur_for_ecc` | `2.0` | Pre-ECC blur sigma — **do not set to 0**, ECC fails without smoothing |

### `compare(stl_aligned, osm, height_scale=None)`

| Parameter | Description |
|---|---|
| `height_scale` | Multiply STL values by this to get metres. `None` = auto from median ratio of overlap region. Pass an explicit value if you know the STL units (e.g., `0.001` if STL is in mm). |

---

## Output Structure

`register_city_stl()` returns a `CityRegistrationReport` frozen dataclass:

```
report.region_name          str
report.stl_file             str
report.stl_heightmap        ndarray (rows, cols)
report.osm_heightmap        ndarray (rows, cols)
report.stl_aligned          ndarray — STL warped into OSM pixel space
report.step_timings         list[(step_name, wall_seconds)]

report.registration         RegistrationResult
  .transform                ndarray (2,3) affine warp matrix
  .confidence               float 0–1 (ECC correlation coefficient)
  .scale                    float spatial scale found
  .angle_deg                float rotation found (degrees)
  .converged                bool

report.comparison           ComparisonResult
  .difference               ndarray (rows, cols) — STL_m − OSM, NaN outside overlap
  .overlap_mask             ndarray bool
  .missing_in_osm           ndarray bool — in STL, not in OSM
  .missing_in_stl           ndarray bool — in OSM, not in STL
  .rmse                     float metres
  .mae                      float metres
  .bias                     float metres (positive = STL buildings taller)
  .correlation              float Pearson r
  .coverage_pct             float %
  .n_overlap                int
  .height_scale_used        float
```

---

## HTML Report Output

`write_registration_report(out_dir, report)` produces:

```
out_dir/
├── index.html                # Summary page with all stats + embedded figures
└── assets/
    ├── comparison.png        # 3-panel: STL | OSM | difference (canonical figure)
    ├── decimation.png        # (simplify_mode=decimate) original | simplified | difference (m)
    ├── decimation_curve.png  # (decimation_curve) surface deviation vs faces kept, budget marked
    ├── prism.png             # (simplify_mode=prism) original | prism model | difference (m)
    ├── binarization.png      # 2x3: STL/OSM heightmap -> binary mask -> edges
    ├── mask_overlay.png      # Footprint agreement (green/orange/blue)
    ├── matched_buildings.png # Per-building height scatter + error map
    ├── stl_heightmap.png     # STL projection (viridis, NaN=gray)
    ├── osm_heightmap.png     # OSM building heights (viridis, NaN=white)
    ├── stl_aligned.png       # STL aligned vs OSM side-by-side
    ├── difference_hist.png   # Histogram of height differences
    ├── missing_analysis.png  # 2-panel: missing_in_osm | missing_in_stl
    └── transform_summary.png # Registration transform parameters
```

The `index.html` includes:
- Registration parameters table (edge IoU + lift, projection, scale, rotation)
- 3-panel comparison image
- **Binarization panel** — heightmaps → binary masks → edges (the registration signal)
- Footprint agreement overlay (green = both, orange = STL only, blue = OSM only)
- Matched-buildings height scatter + per-building error map
- Detail plots (difference histogram, missing analysis)
- Pipeline timing table + bar chart

---

## Algorithm Notes

> Full detail, the rationale for each decision, and a survey of standard
> image-registration techniques we could adopt are in
> [`registration/docs/ARCHITECTURE.md`](../registration/docs/ARCHITECTURE.md).
> This is the short version.

The pipeline is a **similarity registration** (uniform scale + rotation +
translation, no shear) of two height rasters — the STL and the OSM building-height
raster — done **resolution-independently** (the search runs at a fixed 512 grid;
the transform is scaled to the output resolution).

1. **Rotation — from the image gradient.** Sobel on the Gaussian-high-pass height
   field gives a magnitude-weighted orientation histogram (`gradient_angle_histogram`);
   the STL→OSM rotation is the histogram cross-correlation peak. This is
   *segmentation-independent* (no masks/Hough lines). The 90° grid ambiguity is
   resolved with a **bias toward 0°**, overridden only by a clearly-better height
   correlation, or by a manual `--rotation`.
2. **Scale — raw-height xcorr peak.** A scale sweep picks the scale that maximises
   the normalized height cross-correlation (not footprint overlap, which inflates).
3. **Translation — FFT cross-correlation peak.**
4. **Refine — ECC** on building signed-distance fields, projected to a similarity
   transform (shear removed) with the scale pinned.
5. **Compare — per building.** STL height-above-terrain vs OSM height-above-ground,
   aggregated per OSM footprint by **p95** (tip height, matching OSM's single stated
   height), with fill-default and outlier footprints excluded, fit by
   `stl_m = scale·stl + offset`.

### Reading the metrics

- **Footprint Dice** (≈0.99) confirms the footprints overlap — but it's a weak
  signal (dense masks overlap regardless), useful mainly to catch gross failure.
- **Per-building height correlation (Pearson r, Spearman ρ)** is the real quality
  signal now that heights are compared per-building at the tip (p95).
- **Height ratio ≈ 1.0** means the STL→metres scale matches OSM tip heights.
- **Rotation should be ≈0°** for north-aligned grid cities; a confident large value
  (e.g. Denver ≈45°) reflects a genuinely rotated grid.

### height_scale semantics

The STL Z axis is in model units. `height_scale` converts to metres
(`stl_metres = stl_aligned × height_scale`). Because STL heights are noisy, this
calibration is approximate. The spatial XY scale is found by the registration
search; `height_scale` only affects the Z comparison in `compare()`.

### Known limitation: local distortion

A single global transform aligns the centre well but can leave residual drift at
the edges of the frame (the STL may have a non-uniform distortion relative to the
flat OSM map). A piecewise / non-rigid refinement is a possible future step.

---

## OSM Data Quality Notes

OSM building coverage for Philadelphia:
- **Center City**: excellent footprint + `height`/`building:levels` coverage
- **Row-house neighborhoods**: good footprints, sparse height tags → falls back
  to `default_height=10.0 m`

Buildings in `comparison.missing_in_osm` represent either new construction or OSM
data gaps. The high-detail STL source is the authoritative record for those areas.

---

## Installation

```bash
pip install numpy2stl[registration]
# Installs: osmnx, geopandas, rasterio, opencv-python, scikit-image, scipy
```

All registration dependencies are already installed in the project venv.
