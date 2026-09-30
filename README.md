# numpy2stl

Turn NumPy arrays and 2-D geometry into watertight triangle meshes (STL / OBJ / 3MF),
process those meshes (lossless simplification, error-bounded decimation, booleans,
jigsaw cutting), turn meshes back into rasters, and register a city mesh against a
reference height raster.

- **Geo-free by design.** No module fetches map data or converts lon/lat to metres
  (see [Geo-free](#geo-free-no-map-fetching)). map2stl builds on it; it never imports map2stl.
- **Everything returns plain arrays.** Meshes are `(vertices (N, 3) float64, faces (M, 3) int)`;
  rasters are 2-D `ndarray`s with NaN for "no data".
- **Docs in this repo**
  - This README: overview, module map, API, design choices.
  - [docs/registration.md](docs/registration.md): city-STL registration, usage.
  - [registration ARCHITECTURE.md](src/numpy2stl/registration/docs/ARCHITECTURE.md):
    registration design, stages, assumption audit.
  - [docs/archive/](docs/archive/): June 2026 code analysis, reorganisation proposal/checklist and the
    stl2numpy planning and config Q&A (history only).

## Installation

Not on PyPI; install editable from a checkout (src layout, `pyproject.toml`).

```bash
git clone https://github.com/EdgarCardenasDeLaHoz/numpy2stl.git
cd numpy2stl
pip install -e .            # core: numpy, shapely>=2.1, scipy, triangle
pip install -e ".[dev]"     # + pytest, pytest-benchmark, ruff
pip install -e ".[all]"     # tools, boolean, geo, mesh, registration extras
```

- In the 3D Maps project it is installed editable into the shared `~/.venvs/map2stl`
  venv by map2stl's `scripts/setup-venv.ps1`.
- `shapely>=2.1` is a core pin because `shapely.constrained_delaunay_triangles` (GEOS)
  is what `core.extrude` and `processing.simplify` triangulate with.

| Extra | Packages | Needed for |
|---|---|---|
| (core) | numpy, shapely, scipy, triangle | `core`, `raster.fill`, `processing.decimate.heightfield_tin` |
| `tools` | opencv-python, scikit-image | `utils.image.rescale` / `resize_max`, `raster.segment` / `vectorize` (cv2 guarded, skimage for thresholds) |
| `boolean` | pymeshlab, manifold3d | `processing.boolean`, and **any** `numpy2stl.processing.*` import (see below) |
| `mesh` | trimesh, rtree | `stl2numpy`, `io.load_trimesh`, `processing.decimate.decimate_to_tolerance`, `verify` |
| `geo` | rasterio, h5py | `raster.burn_polygons` uses rasterio when installed |
| `registration` | matplotlib, rasterio, opencv-python, scikit-image | `registration` |
| `viz` | matplotlib, napari | `utils.visualization` |
| `all` | tools, boolean, geo, mesh, registration | everything except viz |

- `src/numpy2stl/processing/__init__.py` re-exports `processing.boolean`, which imports pymeshlab at
  module level, so importing any `numpy2stl.processing.*` module needs the `boolean` extra.
  The project venv has it; a minimal install that only wants `decimate` does not.

## Quick start

```python
import numpy as np
from numpy2stl import array_to_mesh, Solid, write3MF

z = np.random.rand(100, 100) * 50                 # heightmap, 1 px = 1 unit
vertices, faces = array_to_mesh(z, solid=True)    # grid top + walls + base
Solid((vertices, faces)).save_stl("terrain.stl")
write3MF("terrain.3mf", {"terrain": (vertices, faces)})
```

Adaptive (error-bounded) solid, then a lossless flat-region pass:

```python
from numpy2stl.core.heightfield import tin_solid
from numpy2stl.processing.simplify import simplify_mesh_surfaces

v, f = tin_solid(z, max_error=0.1, mm_per_px=0.5, floor=0.0)   # row 0 of z = north
f = simplify_mesh_surfaces(v, f)                               # same geometry, fewer faces
```

Masked terrain (e.g. drop the sea):

```python
v, f = array_to_mesh(terrain, mask_val=0, floor_val=0, solid=True)   # cells <= 0 excluded
```

## Module map

```
src/numpy2stl/
  __init__.py        top-level re-exports (core + io writers + utils.image), get_logger
  _paths.py          CACHE_ROOT / REPORTS_ROOT (env NUMPY2STL_CACHE / NUMPY2STL_REPORTS)
  core/              arrays / polygons -> meshes (numpy, shapely, scipy only)
    generate.py      array_to_mesh (grid solid), array2faces, polygon_to_prism/_complex, perimeter_to_walls
    heightfield.py   tin_solid (adaptive TIN top + walls + flat bottom)
    extrude.py       prism, prisms, close_surface, orient_ccw (indexed, watertight)
    solid.py         Solid, vertices_to_index, triangles_to_facets, get_open_edges, validate_object, get_surfaces
    polygon.py       perimeters: get_ordered_perimeter, perimeter_to_2D, get_area/orientation, triangulate_polygon, rotate_3D
  processing/        mesh -> mesh
    simplify.py      simplify_mesh_surfaces (lossless flat-region retriangulation)
    decimate.py      heightfield_tin, heightfield_tin_budget, decimate_to_tolerance
    boolean.py       to_manifold, from_manifold, union, cut_jigsaw, mesh_volume
    extrusion.py     trimesh-based prism helpers (make_prism_solid, make_sloped_prism_solid, ...)
    verify.py        check_model_status, diagnose_mesh, find_mesh_issues
    building_simplify/  city-mesh LOD: simplify_building_mesh, decimation_sweep,
                        flatten_roof_clutter, prism_decompose
  raster/            generic 2-D raster helpers (no geo, no registration)
    burn.py          burn_polygons (polygons -> raster)
    fill.py          fill_nan
    segment.py       terrain_residual, hill_relief_mask, building_mask, split_touching_buildings, building_edges
    vectorize.py     vectorize_buildings (mask -> polygons)
  stl2numpy/         mesh -> arrays: mesh_to_heightmap, mesh_to_voxels, mesh_to_pointcloud,
                     slice_mesh, rasterize_slice, get_mesh_properties, detect_orientation, decimate_mesh
  io/                writeSTL, write3MF, writeOBJ, load_mesh, load_trimesh
  applications/      puzzle.py (jigsaw); cities.py / lidar.py are ImportError tombstones
  registration/      city STL <-> reference raster alignment + comparison + HTML report
  utils/             image.py (rescale, resize_max), visualization.py (matplotlib / napari)
```

## Core: arrays and outlines to solids

- **`core.generate.array_to_mesh(A, mask_val=None, solid=True, floor_val=None)`**
  - One quad (two triangles) per pixel; `solid=True` adds walls along the mask
    boundary and a base at `floor_val` (default: `mask_val`).
  - `mask_val`: cells `<= mask_val` are left out (holes, coastline).
  - Use it when every pixel matters or for small grids; use `tin_solid` for large DEMs.
- **`core.heightfield.tin_solid(z, max_error, mm_per_px=1.0, floor=0.0, seed_step=8)`**
  - Top = `processing.decimate.heightfield_tin` (within `max_error` of every pixel),
    then `close_surface` to a flat bottom at `floor`.
  - Row 0 of `z` is north: pixel `(i, j)` -> `x = j*s`, `y = (H-1-i)*s`.
  - The top surface's vertices are the first `len(vertices) // 2` rows of the result.
- **`core.extrude`** (indexed, watertight, outward-facing; boolean-ready)
  - `prism(poly, z0, z1)`: closed prism over a shapely polygon, holes kept, using only
    the outline's own vertices; `None` for zero height / degenerate input.
  - `prisms(polys, z0, z1)`: same result for many polygons, one GEOS call per step.
    - *Why:* per-call shapely overhead dominated a 25k-building city (~20 s vs ~2 s).
  - `close_surface(top, faces, bottom_z)`: solid from an upward-facing open surface;
    walls along its boundary down to `bottom_z(xy)`, the surface copied there reversed.
  - `orient_ccw(xy, faces)`: rewind faces CCW in xy (normals +z).
  - *Why a second extrusion module:* `core.generate.polygon_to_prism` returns triangle
    soup; booleans (manifold3d) need an indexed closed manifold.
- **`core.solid`**
  - `Solid((vertices, faces))` or `Solid(raw_triangles)` (auto-indexed); methods
    `save_stl(filename, ascii=False)`, `validate_object()` (logs degenerate faces / open edges, returns bool), `simplify()`.
  - `vertices_to_index(triangles)`: dedupe `(M, 3, 3)` soup to indexed form with the
    `np.unique` void-view trick (the old profile had this at 60–65% of `array_to_mesh` time).
  - `triangles_to_facets`, `get_open_edges` (boundary edges), `get_surfaces`
    (connected coplanar groups), `validate_object`.
- **`core.polygon`**: ordered perimeters from edges, 3-D perimeter -> 2-D projection,
  signed area / orientation, collinear-point removal, constrained polygon triangulation.

## Processing: mesh to mesh

- **Lossless simplification: `processing.simplify.simplify_mesh_surfaces(vertices, faces)`**
  - Replaces each planar, edge-connected patch with the fewest triangles covering the
    same polygon. No vertex moves or is added; volume, area and boundary unchanged;
    a watertight input stays watertight.
  - Returns new `faces` indexing the *input* vertices.
  - Triangulates with GEOS constrained Delaunay (`shapely.constrained_delaunay_triangles`):
    rings, holes and pinch vertices handled; no Triangle dependency.
  - *Why lossless first:* heightmap solids (1 px = 1 mm) shrink by orders of magnitude
    on plateaus, flat sea, walls and base, while slopes keep full detail. Used by
    map2stl's lossless export pass.
- **Error-bounded reduction: `processing.decimate`**
  - `heightfield_tin(z, max_error, seed_step=8, max_iter=80, max_vertices=None)`
    -> `(pixel indices into z.ravel(), triangles)`.
    - Seeds: every border pixel + a lattice every `seed_step` px; then repeatedly
      inserts local-peak pixels whose error exceeds `max_error`.
    - *Why a raster TIN, not surface decimation:* the bound is **exact at every pixel**
      and vertices stay on pixel centres; border pixels are all kept, so model edges
      and side walls are exact. Each pass re-rasterises only the changed triangles.
  - `heightfield_tin_budget(...)` -> `(idx, tris, bound_reached)`: with
    `max_vertices`, tightens the bound in `ladder`^k stages and keeps the last that fits
    (a preview budget at the cost of ~one refinement).
  - `decimate_to_tolerance(mesh, deviation_tol)` -> `(mesh, kept_ratio, hausdorff)`:
    binary-searches pymeshlab's feature-preserving quadric decimation for the smallest
    face ratio whose sampled symmetric Hausdorff stays within `deviation_tol`.
    For arbitrary (imported building / city) meshes, where no raster exists.
- **Booleans: `processing.boolean`** (extra `boolean`)
  - `to_manifold(v, f, strict=True)` / `from_manifold(m)`: indexed mesh <-> manifold3d
    `Manifold` (float64); `strict` raises on a non-closed input.
  - `union(meshes)` -> `(Manifold or None, n_rejected)`: batch union of the valid
    closed meshes; invalid ones are counted, not fatal.
  - `cut_jigsaw(v, f, cutters, engine="manifold", max_loss=0.01)`: intersect a model
    with each cutter prism -> `{key: (v, f)}`.
    - Raises if the pieces miss more than `max_loss` of the volume beyond the clearance
      gaps (catches cutters that do not cover the model or overlap).
    - *Why manifold3d by default:* exact and fast; pymeshlab is the fallback engine.
- **Building LOD: `processing.building_simplify`** (used by registration)
  - `simplify_building_mesh` / `decimate_to_tolerance`: footprint-preserving
    decimation to a metres budget; `decimation_sweep` for the trade-off curve.
  - `prism_decompose`: STL -> sum of extruded (optionally sloped-top) prisms.
  - `flatten_roof_clutter`: level per-building roof noise within the budget.
- **Verification: `processing.verify`**: `check_model_status(models, repair=True)`,
  `diagnose_mesh`, vectorised hole / winding checks.

## Raster helpers: `numpy2stl.raster`

- `burn_polygons(polys, shape, transform=None, bounds=None, values=1.0, mode="max")`
  - Polygons (shapely, GeoJSON-like, or `(K, 2)` rings) -> raster; holes stay unburnt.
  - `mode="max" | "sum" | "set"`; `bounds=` is north-up (row 0 = max y).
- `fill_nan(arr, method="nearest", interior_only=False)`: fill NaN cells.
- `segment`: `terrain_residual` (height above local terrain), `hill_relief_mask`,
  `building_mask`, `split_touching_buildings` (watershed gaps), `building_edges`.
  - `building_mask(source="osm")` is `~isnan(heightmap)`: never `nan_to_num` a
    reference heightmap before segmentation, or every cell becomes a building.
- `vectorize_buildings(mask, simplify_frac=0.02, regularize=False)`: mask -> polygons.
- *Why here, not in registration:* they are generic raster primitives also used by
  map2stl.

## Mesh back to arrays: `numpy2stl.stl2numpy`

- **`mesh_to_heightmap(mesh_or_path, resolution=None, projection="max", ..., method="bin", cell_size=None, row0="north")`**
  - The one mesh -> heightmap implementation for numpy2stl and map2stl.
  - `method="bin"`: sample the surface and bin per cell (fast, any mesh).
    `method="raycast"`: one vertical ray per cell centre (exact top, no empty cells
    over the mesh; needs trimesh's ray backend).
  - `row0="north"` (default): row 0 = the mesh's max-y edge (image convention).
    `row0="south"` is exactly `np.flipud` of it; registration and the building
    simplifier pass it because their rasters are row 0 = south.
  - `isotropic=True` renders square pixels, NaN-padded to a square canvas (required for
    registration against an isotropic raster).
  - Pass `resolution` or `cell_size`, not both. File inputs are disk-cached.
  - Returns a dict: `heightmap`, `bounds`, `resolution`, `cell_size`, `projection`,
    `method`, `row0`.
- Others: `mesh_to_voxels`, `mesh_to_pointcloud`, `slice_mesh`, `rasterize_slice`,
  `get_mesh_properties`, `detect_orientation`, `decimate_mesh`.

## I/O: `numpy2stl.io`

- `writeSTL(facets, file_name, ascii=False)`: low-level; takes facets
  (`triangles_to_facets`). Prefer `Solid.save_stl`.
- `write3MF(file_name, {name: (v, f)}, compresslevel=1)`: one object per entry.
  - *Why streamed:* the XML is written into the zip in chunks with plain string
    formatting (vertices to 0.1 µm); an ElementTree node per vertex took ~30 s for a
    3 M-triangle city. `compresslevel=1` is the fastest zlib level and still ~4x smaller.
- `writeOBJ(file_name, {name: (v, f)})`: all meshes in one OBJ.
- `load_trimesh(path)`: STL/OBJ/3MF/PLY as one `trimesh.Trimesh` (scenes concatenated);
  `load_mesh(path)` -> `(vertices, faces)`.

## Puzzles: `numpy2stl.applications.puzzle`

- Frame: x along `width_x` (columns), y along `width_y` (rows), 1 unit = 1 mm; piece
  `r{row}c{col}` with row 0 at min y. Non-uniform grids via `col_edges` / `row_edges`.
- `jigsaw_outlines(...)` -> `{key: Polygon}`; `make_jigsaw_cutters(...)` -> prism
  cutters for `processing.boolean.cut_jigsaw`.
  - `knob_shape`: `"classic"` (rounded knob on a neck), `"dovetail"`, `"rectangular"`
    (default); `knob_width`, `knob_depth`, `clearance` (default 0.3 mm), `border`.
- `heightfield_pieces(z, outlines, max_error, mm_per_px=1.0, floor=0.0)`: pieces meshed
  straight from the heightfield, one closed solid per outline.
  - *Why:* the fast path for terrain-only puzzles, no boolean engine. Walls follow the
    exact outline (knobs, clearance), not the pixel grid; the top is the same TIN the
    whole-model export uses (within `max_error`).
- `underside_marks` + `engrave_underside`: piece id and north arrow engraved in the
  bottom, mirrored to read from below.
- `plate_layout(pieces, bed_w, bed_h)`: shelf-pack pieces onto print beds at z = 0.
- Legacy: `make_puzzle_model`, `make_puzzle_pts`, `make_puzzle_piece`, `make_base_border`.

Example (model already built):

```python
import numpy2stl.processing.boolean as boolean
from numpy2stl.applications.puzzle import make_jigsaw_cutters

cutters = make_jigsaw_cutters(200, 144, cols=4, rows=3, knob_width=12, knob_depth=6,
                              clearance=0.3, z0=-1, z1=100, knob_shape="classic")
pieces = boolean.cut_jigsaw(vertices, faces, cutters)   # {"r0c0": (v, f), ...}
```

## Registration: `numpy2stl.registration`

- Aligns a city STL's heightmap to a reference building-height raster (similarity
  transform), compares heights per building, writes an HTML report.
- `register_city_stl(stl_file, reference, ...)`: `reference` implements
  `ReferenceSource` (building heightmap with `cell_size_m`, optional semantic masks,
  nDSM and candidate frames), or is a `StaticReference` over arrays you already have.
- The OSM-backed wrapper (`city2stl.registration.register_city_stl(stl, "City, ST", ...)`,
  `OSMReference`) and the CLI scripts live in map2stl.
- Usage: [docs/registration.md](docs/registration.md).
  Design: [ARCHITECTURE.md](src/numpy2stl/registration/docs/ARCHITECTURE.md).

## Geo-free: no map fetching

- numpy2stl never fetches map data and never converts lon/lat to metres.
- `tests/test_geo_free.py` fails if any module (function-local imports included)
  imports `osmnx`, `requests`, `urllib`, `httpx`, `overpy`, `pdal`, `py3dep`, or
  map2stl / city2stl / geo2stl.
- *Why:* keeps the library installable and testable offline, and gives one owner
  (map2stl) for network code, caching and projections (F-ARCH decision, 2026-09-26).
- Where the moved code went:

| Was (numpy2stl) | Now (map2stl) |
|---|---|
| `applications.cities` (`get_osm_building_heightmap`, `get_osm_semantic_masks`, `get_city_bbox`, `get_city_center_point`, `estimate_bbox_from_stl`, `tight_bbox_from_extent`, `derive_scale_m_per_unit`, `get_philadelphia_heightmap`) | `city2stl.osm_raster` |
| `applications.lidar.get_ndsm` | `city2stl.height.providers.lidar_3dep_ept.get_ndsm` |
| `register_city_stl(stl, "City, ST")` / bbox tuple, `center=`, `tallest_m=`, `scale_m_per_unit=`, `default_height=`, `levels_to_meters=` | `city2stl.registration.register_city_stl` (same arguments) |
| `registration.center_search.find_best_city_center` | `find_best_target` over `OSMReference.candidate_targets` |
| `python -m numpy2stl.registration.scripts.{run_registration,benchmark_micropolitan,robustness_test}` | `python -m city2stl.registration.scripts.…` |

- numpy2stl cannot import map2stl, so nothing forwards: the old `applications.cities` /
  `.lidar` modules were deleted (2026-09-30) after one release as ImportError tombstones;
  passing a city name to `register_city_stl` raises `TypeError`.

## Utilities

- `utils.image.rescale(im, max_size=600, height=20, base=10, clip=None)`: resize and
  remap an elevation image to print height (extra `tools`); `resize_max`.
- `utils.visualization`: matplotlib `draw_3D_vertices` / `set_limits_3D`, `render_models_napari(models)`.
- Logging: standard `logging` under the `numpy2stl` logger
  (`logging.getLogger("numpy2stl").setLevel(logging.INFO)`); nothing prints.
- Paths (`_paths.py`): caches default to `registration/runs/` in the package tree,
  reports to the shared gitignored `Code/_reports/`; override with
  `NUMPY2STL_CACHE` / `NUMPY2STL_REPORTS`.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # default: -m "not integration and not slow"
pytest -m slow                           # slow tests
pytest tests/benchmarks --benchmark-only # benchmarks
ruff check --fix .                       # same rules as map2stl/ruff.toml
```

- Notebooks: `Generate STL from array.ipynb`, `Demo Make Solid from 2D array.ipynb`.
- License: MIT. Author: Edgar Cardenas De La Hoz.
