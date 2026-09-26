# Registration pipeline — architecture

How a city STL is aligned to OpenStreetMap building data, the module layout, and
the audit of hardcoded assumptions that affect reuse on other cities/STLs.

## Module map

`align.py` and `report_plots.py` were split (2025 refactor) from single 2000-/1000-line
files into subpackages.  Each subpackage's `__init__.py` is a **façade** that re-exports
every public name, so `from numpy2stl.registration.align import X` and
`from numpy2stl.registration.report_plots import render_X` keep working unchanged.

```
registration/
  __init__.py        thin façade — re-exports register_city_stl + public API
  pipeline.py        register_city_stl orchestrator + stage helpers
                     (_simplify_stage, _run_registration, _run_comparison)
  config.py          RegistrationConfig + named tuning constants (see audit)
  compare.py         height comparison (affine fit stl_m = scale*stl + offset)
  types.py           frozen dataclasses (RegistrationResult, ComparisonResult, …)
  html_report.py     HTML assembly (sections in pipeline order)
  align/
    transform.py     preprocess, apply_transform, matrix helpers
    segmentation.py  terrain_residual, building_mask, building_edges, split_touching_buildings,
                     vectorize_buildings (+ rectilinear regularization)
    lines.py         gradient_angle_histogram + rotation_from_angle_histograms (gradient-only;
                     the legacy Hough line detector was removed)
    scale.py         estimate_scale (area + Fourier) — report diagnostics only
    metrics.py       tolerant IoU, Dice, SDF, score_alignment
    global_search.py register_global (gradient rotation + edge-IoU refine + xcorr translation)
    register.py      register() — thin wrapper over register_global (legacy ECC path removed)
    polygon_register.py  register_polygons — polygon point-pattern registration (RANSAC + z-score)
    polygon_icp.py   refine_registration_polygons (boundary ICP, gated Dice>0.95)
    ecc.py           refine_transform, discover_projection
  report_plots/
    _common.py       shared heightmap renderers + render_three_panel (decimation/prism figures)
    inputs.py        STL/OSM heightmaps, aligned side-by-side
    masks.py         binarization, mask overlay, matched buildings, vectorized footprints
    registration.py  transform summary, scale/rotation sweeps, xcorr, angle hist
    comparison.py    3-panel comparison, difference histogram, missing analysis
    decimation.py / prisms.py   simplification evaluation figures
```

## Method evaluation & assumptions (stress-tested)
The raster registration's foundational assumptions were tested empirically:
- **Scale = 1/osm_margin** (geometric) — **holds**: building footprints span ~1.00× the model's
  render extent in every tested model, so render-extent = footprint and the anchor is exact.  It
  breaks only if a model carries a frame/border beyond its buildings, or a different `osm_margin`.
- **Model centred on the city point** — **holds** (centroid within ±0.06 of centre).  Breaks for
  off-centre / cropped tiles → pass `--center`.
- **Rotation** needs a dominant grid (gradient histogram + 0°-bias, edge-IoU refinement); radial
  cities (Paris) need `--rotation`.  **Translation** needs the base-plate anchor in the heightmap.
- **Segmentation**: the STL only raises ~20–40 % of the frame as buildings (triangle threshold),
  vs OSM ~55 % — short buildings sit at the base level and can't be thresholded out without
  re-flooding.  This granularity gap is the main residual limitation.

### Polygon registration (`registration_method="polygon"`, opt-in)
`register_polygons` matches STL↔OSM building footprints directly (RANSAC over centroid pairs,
scale pinned to the geometric anchor, descriptor-pruned candidates, **z-score significance gate**),
removing the grid/base-plate/segmentation fragilities in principle.  **Benchmark finding**: it
recovers a known similarity exactly on synthetic data and rejects unrelated inputs, but on real
Philadelphia it falls back to raster (z≈2.3, only 49/150 inliers) — the STL/OSM **granularity gap**
plus dense-OSM random-baseline prevent significant consensus.  Raster stays the default; the
fallback is automatic and safe.

## Pipeline stages (and the report sections that mirror them)

The whole thing is a **similarity registration** (uniform scale + rotation + translation;
no shear) of two single-channel rasters — an STL height field and an OSM building-height
raster — followed by a height comparison.

0. **(Optional) mesh simplification** (`processing/building_simplify.py`, `simplify_mode`):
   - **`"decimate"`** — footprint-preserving quadric edge-collapse with feature-preservation
     flags, binary-searched to the **most aggressive reduction whose symmetric Hausdorff to the
     original stays ≤ a metres deviation budget** (`simplify_tol_m`, converted to mesh units via
     the model scale).  Plain decimation collapses tall thin buildings; the preserving variant
     keeps height + footprint.  The simplified mesh is rasterized for all downstream stages and
     can be saved.  Decimation evaluation figures: original/simplified/difference + a deviation-
     vs-faces-kept trade-off curve (`--decimation-curve`).
   - **`"prism"`** — decompose the STL into a **sum of extruded prisms** (`prism_decompose`), the
     *reverse* of the OSM render: segment buildings, build each as a wedding-cake stack of nested
     prisms (one footprint per budget-spaced height layer), and cap the top with a fitted plane
     (slanted) where planar within budget (`make_sloped_prism_solid`).  The prism model is saved
     (STL = merged, 3MF = prism soup) and **feeds the comparison/segmentation only** — registration
     stays on the original mesh (the prism heightmap's sharp/quantized content perturbs the scale
     estimate, so it is decoupled; the transform applies equally since footprints share world
     coords).  Tall complex buildings can exceed the budget where `max_layers` caps the stack.
   - Stage-2 roof flattening (`flatten_roof_clutter`) is available for heightmap callers.
1. **STL → heightmap** — `mesh_to_heightmap(..., isotropic=True)`: square-pixel render
   (true aspect, NaN-padded to square) so a world-square building is pixel-square,
   matching the isotropic OSM raster.  Interior NaN holes filled; exterior padding kept NaN.
2. **OSM fetch** — `get_osm_building_heightmap` (per-footprint heights from `height` tag →
   `levels`×3.5 → `default_height`) + `get_osm_semantic_masks` (vegetation/water, to exclude
   trees/rivers from the STL building mask).  Optional `height_source="lidar"` replaces tag
   heights with measured 3DEP nDSM per footprint (`applications/lidar.py`).
3. **Resolution-independent search** — the registration runs at a fixed `REGISTER_RES`
   (512); the found transform's linear part (scale+rotation) is a pixel ratio, so only the
   translation is scaled to the output grid.  Identical transform at any output resolution;
   a 1k output costs the same search as 512.  Comparison/segmentation use the full output res.
4. **Rotation** — coarse from the **image gradient** (`gradient_angle_histogram`: Sobel on the
   Gaussian-high-pass height field → magnitude-weighted orientation histogram), cross-
   correlated between STL and OSM (`rotation_from_angle_histograms`); a **second pass at 0.1°/bin**
   fine-tunes it (kept only if within 2° of the coarse value).  The 90° grid ambiguity is
   resolved among {θ, θ±90, θ+180} by a **bias toward 0°**.  Then an **edge-IoU refinement**
   sweeps rotation ±`_ROT_IOU_WINDOW` (20°) of that estimate — re-solving translation per angle —
   and adopts the footprint **edge-IoU peak only if it beats the histogram pose by `_ROT_IOU_MARGIN`
   (0.02)**.  This catches the case where the heightmap gradient locks onto roof/clutter
   orientation and misses the true building-grid rotation (Philadelphia: histogram ~0° but the
   grid is tilted ~9–12° → edge-IoU recovers it, edge-IoU 0.33→0.63); the margin guard leaves
   already-correct cities untouched (Boston: gain 0 → kept at 0°).  A manual `forced_rotation`
   (`--rotation`) overrides everything.
5. **Scale** — a post-hoc sweep; the scale used is the **peak of the raw-height cross-
   correlation** (segmentation-independent), not the Dice/IoU peak (mask-dependent).
6. **Translation** — normalized FFT cross-correlation peak (`_xcorr_best`).
7. **Refine** — ECC on signed-distance fields, **projected to a similarity transform**
   (no shear/anisotropy) and constrained to **preserve the peak scale**.
7b. **(Optional) polygon ICP** — `refine_polygons=True`, gated on footprint **Dice > 0.95**
   (the coarse registration must already be good): vectorize + match STL↔OSM building
   polygons by centroid, then ICP-align their boundary points to solve a small corrective
   similarity (`align/polygon_icp.py`).  Accepted only if it lowers the point-to-boundary
   RMSE (a guard against drift), then the transform is re-warped and re-compared.  No-ops on
   already-optimal registrations (the residual is genuine shape disagreement, not misalignment).
8. **Compare** — STL height-above-terrain vs OSM height-above-ground.  Per **building**
   (not per pixel): aggregate the STL residual inside each OSM footprint by **p95** (tip
   height, to match OSM's single stated/tip height — median under-reports spires);
   exclude OSM **fill-default** footprints (the modal height, e.g. 10 m) and **residual
   outliers**; affine fit `stl_m = scale·stl + offset` (offset absorbs base-plate bias).
   Footprints are also vectorized (hi-res adaptive `triangle` threshold + Douglas–Peucker)
   for the polygon view.

### Key algorithm decisions (why, so they aren't re-litigated)
- **Resolution-independent search** (`REGISTER_RES`): searching at the output resolution
  made 1k 4× slower and gave slightly different transforms; fixing the search grid makes the
  transform identical across resolutions and the search cost constant.
- **Rotation from the image GRADIENT, not segmentation/Hough lines**: segmentation quality
  (blobby masks on irregular cities) corrupted the line-angle histogram.  The gradient is
  read straight off the height field, so rotation is decoupled from segmentation — which, with
  scale+translation already from height xcorr, makes the *whole* registration
  segmentation-independent (segmentation only serves the comparison/report).
- **Gaussian high-pass before the gradient (not a morphological top-hat)**: the square
  morphological kernel injects axis-aligned artefacts whose orientation flips with a 1–2 px
  kernel change (base swung ±14°); the isotropic Gaussian high-pass has no directional bias.
- **0° bias for the 90°/orientation ambiguity**: footprint/height xcorr is 90°-ambiguous on
  grids and a weak disambiguator (raw terrain correlates spuriously), so default to the
  near-0 orientation; only a clearly-better height correlation (or `--rotation`) overrides.
- **Square-pixel rendering**: a non-square model forced into a square grid shears the
  registration (uniform scale can't undo anisotropy) and inflates the reported scale.
- **Scale from xcorr peak, not Dice/IoU**: Dice/IoU peak at an inflated scale because larger
  footprints overlap more — a segmentation artifact.  Raw-height xcorr is immune.
- **p95 per-building height**: OSM stores one *tip* height per footprint; the STL captures
  full geometry (broad roof + thin spire).  Median returns the roof and under-reports towers;
  p95 matches OSM's tip semantics (height ratio → ~1.0).
- **Landmark check is informational only** — it assumes the model is centred on the city
  landmark, which is false for off-centre tiles; it must never drive scale/rotation.
- **Known limitation**: rotation auto-detection needs a dominant orthogonal grid.  Irregular
  (Boston — fixed by the 0° bias) and radial (Paris) cities need `--rotation`/`--center`.
  Hough line detection is now diagnostic-only (rotation uses the gradient); the figures were
  removed from the report.

## Hardcode / assumption audit (reuse on other cities / STLs)

| Assumption / value | Where | Risk on a new city/STL | Override |
|---|---|---|---|
| City centre + tallest building | `_CITY_CONFIG` (cities.py) | Only 4 cities pre-listed | `center=`, `tallest_m=`, or `scale_m_per_unit=` on `register_city_stl` / CLI `--center --tallest-m`; else the centre is **geocoded** automatically |
| Model centred on city centre | bbox + landmark check | Off-centre tiles misregister | use the complete single-piece model; pass `center=` |
| `z_max` ⇒ tallest building | scale anchor | Cropped models lack the tallest tower | pass `tallest_m`/`scale_m_per_unit` |
| OSM frame = 1.5× footprint | `DEFAULT_OSM_MARGIN` (config.py) | — | `config.osm_margin` |
| Building threshold = p50 | `DEFAULT_BUILDING_THRESHOLD_PCT` | Sparse/suburban models may want lower | config |
| Terrain kernel = 80 m | `DEFAULT_TERRAIN_KERNEL_M` | Very wide blocks absorbed into terrain | config |
| Scale sweep ±0.45 | `DEFAULT_SCALE_SWEEP_HALFWIDTH` | True scale far from prior | config |
| Search resolution = 512 | `REGISTER_RES` (config.py) | — (transform is resolution-independent) | config |
| Output/comparison resolution = 1024 | `resolution` arg | Higher = sharper footprint/agreement images (more agreement px), slightly slower; does NOT change the transform | `resolution=`, CLI `--resolution` |
| Mesh simplification off | `simplify_mesh` | Cleaner footprints when on; needs trimesh+pymeshlab; adds ~1 min decimation | `simplify_mesh=True`, `--simplify-mesh` |
| Deviation budget = 3.5 m | `simplify_tol_m` / `DEFAULT_SIMPLIFY_TOL_M` | Larger = more aggressive detail removal | `simplify_tol_m=`, `--simplify-tol-m` |
| Footprint regularization off | `regularize_footprints` | Watershed-splits merged blocks + rectilinearizes polygons | `regularize_footprints=True`, `--regularize-footprints` |
| Polygon ICP off | `refine_polygons` | Fine-tunes the transform via matched building polygons (gated Dice>0.95) | `refine_polygons=True`, `--refine-polygons` |
| Auto-rotation needs a grid | gradient histogram | Irregular/radial cities (Boston/Paris) | `--rotation DEG` |
| p95 tip height | `height_agg` (compare.py) | Models where roof≠tip differ | `height_agg="median"/"max"/"pNN"` |
| Metres per degree = 111320 | `M_PER_DEG_LAT` (config.py) | Fine for mid-latitudes | — |

All tuning constants live in `config.py`; pass a `RegistrationConfig` to override.

## Improvement opportunities — standard image-registration techniques

Our pipeline is hand-rolled (gradient-histogram rotation + xcorr scale-sweep + FFT
translation + ECC).  Standard techniques from the registration literature map cleanly onto
its weak spots:

| Technique | What it is | Where it would help us |
|---|---|---|
| **Fourier–Mellin / log-polar** | Recover rotation **and** scale together from the magnitude spectrum (rotation → angular shift, scale → log-radius shift), then phase-correlate for translation. | The *textbook* method for our similarity problem and **needs no street grid** — could fix the non-grid cities and the 90° ambiguity in one step.  **Prototyped** in `align/fourier_mellin.py` (see below). |
| **Mutual information (MI)** | A similarity metric that maximises statistical dependence, not intensity equality — the standard objective for **multi-modal** registration. | STL absolute height vs OSM building height is genuinely multi-modal; MI (or normalized MI) is a better orientation/translation score than raw-height Pearson xcorr, which is why our height-corr disambiguator is weak.  Use MI to score the 4 orientation candidates and as the ECC objective. |
| **Phase correlation** | Cross-power spectrum normalized by magnitude → a sharp delta at the translation; far more robust to contrast/outliers than raw xcorr. | Replace `_xcorr_best`'s raw correlation for translation; sharper, less biased by a few tall buildings.  (We previously preferred raw for *binary* masks, but for the height field phase correlation should win.) |
| **Feature-based + RANSAC** | Detect keypoints (corners/ORB), match descriptors, fit the transform with RANSAC. | Robust to partial overlap and non-grid layouts; building corners are good features.  A RANSAC similarity fit would give an orientation-agnostic alternative to the histogram for irregular cities. |
| **Multi-scale (Gaussian/Laplacian pyramid) coarse-to-fine** | Register on a coarse level, propagate, refine on finer levels. | We already cap the search at 512; a true pyramid would make the search both faster and more robust to local minima, and naturally supports the scale-sweep coarse-to-fine trim. |
| **Robust intensity models (Huber/Tukey) in ECC** | Down-weight outlier pixels in the correlation objective. | Our ECC overfits the periodic grid (we patch it by projecting out shear); a robust loss would reduce that pull directly. |

**Highest-leverage:** (1) **Fourier–Mellin** for a grid-free, principled rotation+scale that
could fix Boston/Paris automatically, and (2) **mutual information** as the multi-modal score
for orientation disambiguation and refinement.  Both are well-supported by OpenCV/scikit-image
and would replace bespoke heuristics with established methods.

### Fourier–Mellin prototype — findings (`align/fourier_mellin.py`)
A clean implementation (Hanning window + Reddy–Chatterji high-pass emphasis + log-polar phase
correlation; the 180° spectrum ambiguity broken by overlap correlation; an optional
`refine_overlap` loop that masks both images to their mutual overlap and re-estimates the
residual).  Validate with `python -m numpy2stl.registration.scripts.fourier_mellin_prototype`
(synthetic recovery + random-crop test, no network) or `--stl … --region …` (vs the production
path).  Unit-tested in `TestFourierMellin` (11 tests).

- **Concept proven.**  On synthetic city rasters it recovers a known rotation+scale to **<0.1°
  / <0.01 scale**, **identically on a grid and an irregular (random-orientation) layout** — the
  grid-free property the gradient histogram lacks; it would in principle fix Boston/Paris.
- **Partial overlap is NOT the blocker.**  The random-crop test (transform a base, keep only a
  random sub-window, recover the transform) recovers to **<0.3° / <0.005 scale / <0.5 px even at
  ~5–14 % overlap**, grid and irregular alike.  So a small shared extent is fine *as long as the
  visible content is the same and the rest is zero*.  The `refine_overlap` loop sharpens the
  hardest crops (e.g. 0.17°→0.06°) and is cheap.
- **The real blocker is cross-modality + foreign content.**  On Philadelphia the production path
  gives rot ≈0° / scale ≈0.98; Fourier–Mellin converges — stably across raw, high-passed,
  overlap-refined, and mask inputs — on a *wrong* pose (rot ≈+9° / scale ≈0.68).  Its
  phase-corr **response stays ≈0.14 vs 0.7–1.0 on synthetic**, correctly self-flagging.  The
  cause is NOT the partial extent (handled above) but that the non-overlap region holds
  *different real buildings* and the two rasters are *cross-modal* (STL = terrain + full 3-D
  geometry, OSM = flat-topped footprints).  High-pass lifts the pose overlap-corr (0.43→0.48) but
  not the response; masks don't help.
- **Recommended path:** use the response as a **confidence gate** — keep the gradient path as the
  default and fall back to Fourier–Mellin only for flat-histogram (non-grid) cities *and only
  when its response clears a threshold* (synthetic ≈0.7; real Philadelphia ≈0.14 ⇒ correctly
  rejected).  To make it trustworthy on real data the next step targets cross-modality: a common
  representation (edge / gradient-magnitude maps) or a **mutual-information** log-polar score —
  not more overlap cropping.

## Tests / verification
`numpy2stl/tests/test_registration.py` imports through the façades, so it exercises the
split.  Run `pytest numpy2stl/tests/test_registration.py` (45 tests).  Reference run:
register `Philadelphia, PA_L.3mf` — expect **scale 0.724, rotation ≈0°, Dice ≈0.999,
r ≈0.58 (p95), bias ≈0, shear 0**, and the **same scale/rotation at 512 and 1024** (only the
translation and comparison detail change with resolution).  Boston (`--region "Boston, MA,
USA"`) should give rotation ≈0° (0° bias) and r ≈0.6; Denver/Paris need `--center` and often
`--rotation`.
