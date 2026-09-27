# numpy2stl

Convert NumPy arrays and geometric shapes into 3D mesh files (STL/OBJ/3MF) for 3D printing and visualization.

## Features

- **2D Array → 3D Mesh**: Transform elevation data into watertight solids with automatic walls and base
- **Multi-format Export**: Save as STL (binary/ASCII), OBJ, or 3MF
- **Polygon Processing**: Extrude 2D polygons into 3D prisms or complex shapes
- **Mesh Operations**: Simplification, validation, and boolean operations
- **Optional Modules**: Advanced features for geographic data, puzzle generation, and mesh manipulation
- **City STL Registration**: Align a city 3D model to a building-height raster (e.g. from OpenStreetMap) and compare heights — see [docs/registration.md](docs/registration.md) and [registration/docs/ARCHITECTURE.md](registration/docs/ARCHITECTURE.md)

## Installation

numpy2stl is not published to PyPI; install it from a checkout. Packaging is
defined in `pyproject.toml` (src layout, `src/numpy2stl/`).

```bash
git clone https://github.com/EdgarCardenasDeLaHoz/numpy2stl.git
cd numpy2stl
pip install -e .            # core: numpy, shapely, scipy, triangle
pip install -e ".[dev]"     # + pytest, pytest-benchmark, ruff
pip install -e ".[all]"     # tools, boolean, geo, mesh, registration extras
```

Within the 3D Maps project it is installed editable into the shared
`~/.venvs/strm2stl` venv by strm2stl's `scripts/setup-venv.ps1`.

## Quick Start

### Basic Usage: Elevation Array → STL

```python
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Create elevation data (e.g., terrain, heightmap)
elevation = np.random.rand(100, 100) * 50

# Convert to watertight mesh
vertices, faces = array_to_mesh(elevation, solid=True)

# Save as STL
solid = Solid((vertices, faces))
solid.save_stl("terrain.stl")
```

### With Masking (e.g., Ocean Floor)

```python
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Create terrain with water (negative values = ocean)
terrain = np.random.rand(200, 200) * 100 - 20

# Mask out underwater regions
vertices, faces = array_to_mesh(
    terrain, 
    mask_val=0,      # Exclude values <= 0
    floor_val=0,     # Base at sea level
    solid=True
)

solid = Solid((vertices, faces))
solid.save_stl("coastal_terrain.stl")
```

### Export Multiple Models (3MF/OBJ)

```python
from numpy2stl import array_to_mesh, write3MF, writeOBJ

models = {}
for i, data in enumerate(terrain_arrays):
    models[f"region_{i}"] = array_to_mesh(data, solid=True)

# Save all models in one file
write3MF("regions.3mf", models)
writeOBJ("regions.obj", models)
```

## Core API

### Main Functions

#### `array_to_mesh(A, mask_val=None, solid=True, floor_val=None)`
Convert a 2D elevation array into a 3D mesh.

**Parameters:**
- `A`: 2D numpy array (elevation data)
- `mask_val`: Exclude values ≤ this threshold (default: include all)
- `solid`: Add walls and bottom for watertight solid (default: True)
- `floor_val`: Z-coordinate of base (default: same as mask_val)

**Returns:** `(vertices, faces)` tuple

---

#### `Solid(triangles)`
Mesh container with validation and export methods.

**Methods:**
- `save_stl(filename, ascii=False)`: Export to STL format
- `validate_object()`: Check for manifold errors
- `simplify()`: Reduce face count while preserving shape

---

#### `writeSTL(facets, filename, ascii=False)`
Low-level STL writer (requires facet format).

#### `write3MF(filename, models_dict)`
Export multiple meshes to 3MF format.

**Parameters:**
- `models_dict`: `{"name": (vertices, faces), ...}`

#### `writeOBJ(filename, models_dict)`
Export multiple meshes to OBJ format.

---

### Utilities

#### `rescale(im, max_size=600, height=20, base=10, clip=None)`
Rescale elevation data for 3D printing dimensions.

**Requires:** the `tools` extra (`pip install -e ".[tools]"`)

#### `vertices_to_index(triangles)`
Convert raw triangle facets to indexed vertex representation.

#### `triangles_to_facets(triangles)`
Convert indexed mesh to STL facet format (with normals).

## Advanced Usage

### Polygon Extrusion

```python
from numpy2stl import polygon_to_prism, Solid, write3MF
import numpy as np

# Define 2D polygon vertices with heights
vertices = np.array([
    [0, 0, 10],
    [10, 0, 10],
    [10, 10, 15],
    [0, 10, 12]
])

# Extrude to base at z=0
triangles = polygon_to_prism(vertices, base_val=0)
solid = Solid(triangles)
solid.save_stl("extruded_polygon.stl")
```

### Mesh Simplification

```python
import numpy2stl.processing.simplify as simp

# Lossless: flat regions are retriangulated with the fewest triangles that
# cover exactly the same polygon. Same volume, area and boundary; no vertex
# moves or is added; a watertight input stays watertight. Returns faces that
# index the input vertices.
simplified_faces = simp.simplify_mesh_surfaces(vertices, faces)
```

### Boolean Operations

```python
# Requires the boolean extra: pip install -e ".[boolean]"
import numpy2stl.processing.boolean as boolean

from numpy2stl.applications.puzzle import make_jigsaw_cutters

# 200 x 144 mm model -> 4 x 3 interlocking pieces keyed "r{row}c{col}" (row 0 = min y)
cutters = make_jigsaw_cutters(200, 144, cols=4, rows=3, knob_width=12, knob_depth=6,
                              clearance=0.3, z0=-1, z1=100)
# manifold3d intersection; raises if the pieces miss more than the clearance gaps
pieces = boolean.cut_jigsaw(vertices, faces, cutters)
```

### Visualization

```python
# Requires the viz extra: pip install -e ".[viz]"
import numpy2stl.utils.visualization as view

models = {"terrain": (vertices, faces)}
view.render_models_napari(models)  # Opens interactive 3D viewer
```

## Optional Modules

All optional modules must be imported explicitly:

```python
import numpy2stl.processing.simplify as simp       # Mesh simplification (scipy/shapely)
import numpy2stl.processing.boolean as boolean     # Boolean operations (pymeshlab/manifold3d)
import numpy2stl.applications.puzzle as puzzle     # Puzzle piece generation (trimesh)
import numpy2stl.utils.visualization as view       # 3D visualization (matplotlib/napari)
import numpy2stl.raster as raster                  # mask/polygon/NaN-fill helpers (cv2, rasterio optional)
from numpy2stl.stl2numpy import mesh_to_heightmap  # mesh → heightmap (trimesh)
```

`numpy2stl.raster` holds the generic 2-D raster helpers: `terrain_residual`,
`building_mask`, `building_edges`, `split_touching_buildings` (segment),
`vectorize_buildings` (mask → polygons), `burn_polygons` (polygons → raster,
`mode="max"|"sum"|"set"`, holes kept, `bounds=` is north-up) and `fill_nan`.
`mesh_to_heightmap(mesh_or_path, resolution | cell_size=, method="bin"|"raycast",
row0="north"|"south")` is the one mesh → heightmap conversion; row 0 defaults to the
mesh's max-y edge (image convention). Pass `row0="south"` for the min-y edge first, as
the registration pipeline does (its rasters are row 0 = south).

### Geo-free: no map fetching

numpy2stl never fetches map data and never converts lon/lat to metres; a static
test (`tests/test_geo_free.py`) fails if any module imports `osmnx`, `requests`,
`pdal` or strm2stl. Registration takes the reference side as an input:
`register_city_stl(stl_file, reference)` where `reference` implements
`numpy2stl.registration.ReferenceSource` (building heightmap with `cell_size_m`,
optional vegetation/water/bridge masks and nDSM, candidate frames for the centre
search), or is a `StaticReference` over arrays you already have.

The OSM and 3DEP lidar code moved to strm2stl:

| Was (numpy2stl) | Now (strm2stl) |
|---|---|
| `applications.cities` (`get_osm_building_heightmap`, `get_osm_semantic_masks`, `get_city_bbox`, `get_city_center_point`, `estimate_bbox_from_stl`, `tight_bbox_from_extent`, `derive_scale_m_per_unit`, `get_philadelphia_heightmap`) | `city2stl.osm_raster` |
| `applications.lidar.get_ndsm` | `city2stl.height.providers.lidar_3dep_ept.get_ndsm` |
| `register_city_stl(stl, "City, ST")` / bbox tuple, `center=`, `tallest_m=`, `scale_m_per_unit=`, `default_height=`, `levels_to_meters=` | `city2stl.registration.register_city_stl` (same arguments) |
| `registration.center_search.find_best_city_center` | `find_best_target` over `OSMReference.candidate_targets` |
| `python -m numpy2stl.registration.scripts.{run_registration,benchmark_micropolitan,robustness_test}` | `python -m city2stl.registration.scripts.…` |

numpy2stl cannot import strm2stl, so the old modules are not forwarding shims:
for one release, importing `numpy2stl.applications.cities` / `.lidar` (or the old
names from `numpy2stl.applications`) raises an `ImportError` naming the new
location, and passing a city name to `register_city_stl` raises a `TypeError`
saying the same.

### Installation by Feature

Extras are declared in `pyproject.toml` under `[project.optional-dependencies]`.

| Feature | Install Command | Use Case |
|---------|----------------|----------|
| Core mesh generation, simplification | `pip install -e .` | Array processing, triangulation |
| Image rescaling | `pip install -e ".[tools]"` | DEM preprocessing (opencv, scikit-image) |
| Boolean operations | `pip install -e ".[boolean]"` | Combine/cut meshes |
| Puzzle / trimesh helpers | `pip install -e ".[mesh]"` | Puzzle pieces, extrusion |
| City STL registration | `pip install -e ".[registration]"` | Raster alignment + height comparison (OSM fetching is strm2stl's) |
| Visualization | `pip install -e ".[viz]"` | View meshes |
| Everything | `pip install -e ".[all]"` | All features (no viz) |

## Logging

numpy2stl uses Python's standard logging module. Configure verbosity as needed:

```python
import logging

# Show all numpy2stl debug messages
logging.basicConfig(level=logging.DEBUG)

# Or configure just numpy2stl
logging.getLogger('numpy2stl').setLevel(logging.INFO)
```

## Examples

See the notebooks in the parent project:
- `Generate STL from array.ipynb`: Basic elevation mesh generation
- `Demo Make Solid from 2D array.ipynb`: Watertight solid creation

## Requirements

- Python 3.11+
- NumPy >= 1.24.0
- Shapely >= 2.0.0

Optional dependencies are declared as extras in `pyproject.toml`.

## Contributing

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests with coverage
pytest tests/ -v --cov=numpy2stl --cov-report=html

# Run benchmarks
pytest tests/benchmark/ -v --benchmark-only
```

### Code Style

This project uses Black and Ruff for formatting:

```bash
# Format code
black .

# Lint
ruff check --fix .
```

## License

MIT License (see LICENSE file)

## Related Projects

- **strm2stl**: Geographic terrain mesh generation from SRTM/DEM data
- **geo2stl**: Geographic data processing for 3D mapping

## Credits

Created by Edgar Cardenas De La Hoz

## Changelog

### v0.1.0 (2024)
- Improved dependency management with optional extras
- Added logging support (replaced print statements)
- Fixed import issues for better module organization
- Python 3.11+ requirement
- Better error messages for missing optional dependencies
- Conditional imports for heavy dependencies

### v0.0.1
- Initial release
- Basic array to mesh conversion
- STL/OBJ/3MF export
- Polygon processing
