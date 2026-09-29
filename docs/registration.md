# numpy2stl — city STL registration (usage)

Register a city 3D mesh against a building-height raster (normally OpenStreetMap) to
find the alignment and compare the two datasets' heights.

- **Design, stages, why each choice, assumption audit:**
  [registration ARCHITECTURE.md](../src/numpy2stl/registration/docs/ARCHITECTURE.md).
- **Library overview:** [numpy2stl README](../README.md).

## Who does what

- numpy2stl is geo-free: `register_city_stl(stl_file, reference, ...)` takes a
  `numpy2stl.registration.ReferenceSource`, or a `StaticReference` over arrays you
  already have. It never fetches OSM.
- map2stl wraps it for real cities:
  - `city2stl.registration.register_city_stl(stl_file, city_name, ...)` builds an
    `OSMReference` (over `city2stl.osm_raster`) and calls numpy2stl.
  - OSM rasters: `city2stl.osm_raster.get_osm_building_heightmap`,
    `get_osm_semantic_masks`, `get_philadelphia_heightmap`.
  - CLI: `python -m city2stl.registration.scripts.run_registration --stl PATH --region "City, ST"`
    (also `benchmark_micropolitan`, `robustness_test`).
- **Any city works** through the map2stl wrapper: cities outside its built-in config
  geocode their centre. Pass `tallest_m=` or `scale_m_per_unit=` (CLI `--tallest-m`,
  `--scale-m-per-unit`) for a tight, well-scaled OSM fetch, or `center=` (`--center LAT,LON`)
  to set the downtown point.

## Quick start

With map2stl (fetches OSM):

```python
from city2stl.registration import register_city_stl

report = register_city_stl(
    stl_file="philadelphia.stl",
    city_name="Philadelphia, PA, USA",
    resolution=512,
    height_scale=None,     # auto-estimate STL units -> metres
    out_dir="./report",    # index.html + assets/
)
print(report.comparison.match_score)   # 0–1 composite match quality: the number to read
print(report.registration.scale, report.registration.angle_deg)
print(f"RMSE {report.comparison.rmse:.1f} m, bias {report.comparison.bias:+.1f} m, "
      f"coverage {report.comparison.coverage_pct:.1f}%")
```

- `report.registration.confidence` is the raw FFT cross-correlation peak: a search
  diagnostic with no defined ceiling, **not** a match-quality score.

numpy2stl only (reference raster in hand; row 0 = south, NaN = no building):

```python
from numpy2stl.registration import StaticReference, register_city_stl

ref = StaticReference(osm_heightmap, cell_size_m=2.0, name="philadelphia")
report = register_city_stl("philadelphia.stl", ref, resolution=512, out_dir="./report")
```

## Step by step

```python
from numpy2stl.stl2numpy import mesh_to_heightmap
from numpy2stl.registration import register, apply_transform, compare
from city2stl.osm_raster import get_osm_building_heightmap   # map2stl

# 1. STL -> heightmap (arbitrary units); registration rasters are row 0 = south
stl = mesh_to_heightmap("city.stl", resolution=512, projection="max", row0="south")

# 2. Reference building heights (or any (rows, cols) array, NaN = no building)
osm = get_osm_building_heightmap("Philadelphia, PA, USA", resolution=512)

# 3. Similarity transform (scale + rotation + translation)
reg = register(stl["heightmap"], osm["heightmap"], max_scale_ratio=5.0)
print(reg["transform"], reg["scale"], reg["angle_deg"])

# 4. Warp the STL into reference pixel space
aligned = apply_transform(stl["heightmap"], reg["transform"],
                          output_shape=osm["heightmap"].shape)

# 5. Compare heights (auto-fit STL units -> metres)
comp = compare(aligned, osm["heightmap"], height_scale=None)
print(comp.rmse, comp.bias, comp.coverage_pct)
```

- The report writer is `numpy2stl.registration.write_registration_report(out_dir, report)`;
  it takes a full `CityRegistrationReport`, so in practice use `register_city_stl`.

## Parameters

### `register_city_stl` (numpy2stl)

| Parameter | Default | Description |
|---|---|---|
| `stl_file` | — | City STL / 3MF / OBJ; no geographic metadata needed |
| `reference` | — | `ReferenceSource` (map2stl `OSMReference`, or `StaticReference`) |
| `resolution` | `1024` | Output grid for comparison / report images. The **search** always runs at `REGISTER_RES` (512), so this sharpens images only; transform and search cost are unchanged |
| `max_scale_ratio` | `5.0` | Spatial scale search range |
| `height_scale` | `None` | STL units -> metres; `None` = auto-fit |
| `stl_z_axis` | `2` | Mesh axis that is elevation |
| `out_dir` | `None` | Report folder; default `Code/_reports/{region}/`; `False` = no files |
| `forced_rotation` / `forced_scale` | `None` | Pin rotation (deg) / scale; CLI `--rotation` |
| `height_source` | `"osm"` | `"lidar"`: per-footprint median of `reference.ndsm()` (falls back to reference heights) |
| `simplify_mode` | `"off"` | `"decimate"` (footprint-preserving quadric) or `"prism"` (sum of extruded prisms); feeds comparison only, registration uses the original mesh |
| `simplify_mesh` | `False` | Alias for `simplify_mode="decimate"` |
| `simplify_tol_m` | `3.5` | Deviation budget (m) for simplification |
| `save_simplified` | `None` | Write the simplified mesh (STL / 3MF / OBJ by extension) |
| `regularize_footprints` | `False` | Watershed-split merged buildings + rectilinear footprint snapping |
| `refine_polygons` | `False` | Polygon-ICP fine-tune, only when footprint Dice > 0.95 |
| `decimation_curve` | `False` | Add the deviation-vs-faces-kept curve to the report (~30 s) |
| `registration_method` | `"raster"` | `"polygon"`: match footprints directly; auto-falls back to raster at low confidence |
| `free_scale` | `False` | Refine scale within ±10% of the geometric anchor; usually drifts, so off |
| `center_search` | `"never"` | `"auto"` / `"always"`: score the reference's candidate frames. Off because the lock gate does not yet tell right from wrong centres |
| `region_name` | `reference.name` | Report title and default output folder |

### map2stl `city2stl.registration.register_city_stl` extras

- `city_name` (name or `(N, S, E, W)` bbox), `center`, `tallest_m`,
  `scale_m_per_unit`, `default_height` (10 m), `levels_to_meters` (3.5) build the OSM
  reference; everything else passes through to numpy2stl.

### `register(source, target, ...)`

| Parameter | Default | Description |
|---|---|---|
| `max_scale_ratio` | `5.0` | Scale search range |
| `known_scale` | `None` | Geometric scale anchor; the search refines around it |
| `scale_search` | `0.35` | Half-width of the scale refinement |
| `cell_size_m` | `None` | Metres per pixel (metre-based kernels) |
| `forced_rotation` | `None` | Skip rotation search |
| `source_mask` / `source_exclude_mask` | `None` | STL building mask / cells to treat as non-building (vegetation, hillside, water) |
| `free_scale` | `False` | As above |

- Returns a dict: `transform` (2×3), `confidence`, `scale`, `angle_deg`, `converged`,
  `n_iterations`, plus diagnostic sweeps for the report.

### `compare(stl_aligned, osm, height_scale=1.0, height_offset=None, height_agg="p95")`

- `height_scale`: multiply STL values by this for metres (default 1.0 = already metres).
  `None` least-squares fits `stl_m = scale·stl + offset` over the overlap; pass a value
  if you know the units (`0.001` for mm). `register_city_stl` passes `None` by default.
  - *Why an intercept:* it absorbs base-plate / terrain-estimate bias that a
    through-origin fit would push into the slope.
- `height_agg`: per-footprint aggregate of the STL (`"p95"`, `"median"`, `"max"`, `"pNN"`).
  p95 because OSM stores one tip height per footprint.

## Output: `CityRegistrationReport`

```
report.region_name, .stl_file, .step_timings [(step, seconds)]
report.stl_heightmap / .osm_heightmap / .stl_aligned   (rows, cols)
report.cell_size_m, .osm_bbox, .stl_building_mask

report.registration  RegistrationResult
  .transform (2,3)  .scale  .angle_deg  .confidence (raw xcorr peak)  .converged  .projection

report.comparison    ComparisonResult
  .match_score (0–1 composite) + .match_score_components
  .footprint_iou  .dice_score  .overlap_iou  .overlap_precision  .edge_lift  .height_corr
  .rmse  .mae  .bias (+ = STL taller)  .correlation (Pearson)  .rank_correlation (Spearman)  .mape
  .coverage_pct  .n_overlap  .height_scale_used  .height_offset_used
  .difference / .building_diff_map   STL_m − OSM, NaN outside overlap
  .missing_in_osm / .missing_in_stl  bool masks
```

- Use `footprint_iou` (union IoU), not `dice_score`, as the registration-quality number.

## HTML report

`index.html` plus `assets/*.png` inside `out_dir`, sections in pipeline order:

- Inputs: `stl_heightmap`, `osm_heightmap`, `stl_aligned`.
- Registration: `transform_summary`, `binarization` (heightmaps -> masks -> edges, the
  registration signal), `angle_hist`, `scale_sweep`, `rot_sweep`, `xcorr_map`, `vectorized`.
- Agreement: `mask_overlay` (green both / orange STL only / blue OSM only),
  `footprint_rgchannel`, `matched_buildings` (per-building scatter + error map).
- Heights: `comparison` (STL | OSM | difference), `corrected_difference`,
  `difference_hist`, `missing_analysis`.
- Simplification (when enabled): `decimation`, `decimation_curve`, `prism`.
- Timing table.

## Reading the metrics

- **`match_score`** is the summary number (overlap IoU, overlap precision, edge lift,
  height correlation combined).
- **Footprint Dice ≈ 0.99** only confirms gross overlap: dense masks overlap regardless.
- **Per-building r / ρ** is the real height-quality signal (per building, at p95).
- **Height ratio ≈ 1.0** means the STL -> metres scale matches OSM tip heights.
- **Rotation ≈ 0°** for north-aligned grids; a confident large value (Denver ≈ 45°) is a
  genuinely rotated grid.

## Known limitations

- One global similarity transform: the centre aligns well, edges can drift if the STL
  is non-uniformly distorted relative to the flat map.
- Rotation auto-detection needs a dominant grid; radial cities (Paris) need
  `forced_rotation` / `--rotation`, off-centre tiles need `center=`.
- OSM height tags are sparse outside downtowns (Philadelphia row-house blocks fall back
  to `default_height` = 10 m); those footprints are excluded from the height fit.
  `missing_in_osm` marks new construction or OSM gaps.

## Installation

```bash
pip install -e ".[registration]"   # matplotlib, rasterio, opencv-python, scikit-image
pip install -e ".[mesh]"           # trimesh (mesh_to_heightmap)
```

- OSM fetching (osmnx, geopandas) is map2stl's dependency, not numpy2stl's.
- The 3D Maps venv (`~/.venvs/map2stl`) has everything.
