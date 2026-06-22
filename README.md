# numpy2stl

Convert NumPy arrays and geometric shapes into 3D mesh files (STL/OBJ/3MF) for 3D printing and visualization.

## Features

- **2D Array → 3D Mesh**: Transform elevation data into watertight solids with automatic walls and base
- **Multi-format Export**: Save as STL (binary/ASCII), OBJ, or 3MF
- **Polygon Processing**: Extrude 2D polygons into 3D prisms or complex shapes
- **Mesh Operations**: Simplification, validation, and boolean operations
- **Optional Modules**: Advanced features for geographic data, puzzle generation, and mesh manipulation
- **City STL Registration**: Align a city 3D model to OpenStreetMap buildings and compare heights — see [docs/registration.md](docs/registration.md) and [registration/docs/ARCHITECTURE.md](registration/docs/ARCHITECTURE.md)

## Installation

### Basic Installation
```bash
pip install numpy2stl
```

### With Optional Features
```bash
# Core functionality (scipy-based array processing)
pip install numpy2stl[core]

# Image processing tools
pip install numpy2stl[tools]

# Full recommended set
pip install numpy2stl[full]

# Development tools
pip install numpy2stl[dev]
```

### Development Mode
```bash
git clone https://github.com/EdgarCardenasDeLaHoz/numpy2stl.git
cd numpy2stl
pip install -e .[dev]
```

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

**Requires:** `pip install numpy2stl[tools]`

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
# Requires: pip install numpy2stl[simplify]
import numpy2stl.simplify as simp

# Reduce face count by 50%
simplified_faces = simp.simplify_mesh_surfaces(vertices, faces, min_faces=10)
```

### Boolean Operations

```python
# Requires: pip install numpy2stl[boolean]
import numpy2stl.boolean as boolean

# Cut puzzle pieces from base mesh
pieces = boolean.cut_puzzle_pieces(base_model, puzzle_cutters)
```

### Visualization

```python
# Requires: pip install numpy2stl[visualization]
import numpy2stl.view as view

models = {"terrain": (vertices, faces)}
view.render_models_napari(models)  # Opens interactive 3D viewer
```

## Optional Modules

All optional modules must be imported explicitly:

```python
import numpy2stl.simplify as simp    # Mesh simplification (scipy)
import numpy2stl.boolean as boolean  # Boolean operations (pymeshlab/manifold3d)
import numpy2stl.puzzle as puzzle    # Puzzle piece generation (trimesh)
import numpy2stl.view as view        # 3D visualization (matplotlib/napari)
```

### Installation by Feature

| Feature | Install Command | Use Case |
|---------|----------------|----------|
| Core mesh generation | `pip install numpy2stl[core]` | Array processing, triangulation |
| Image rescaling | `pip install numpy2stl[tools]` | DEM preprocessing |
| Mesh simplification | `pip install numpy2stl[simplify]` | Reduce file size |
| Boolean operations | `pip install numpy2stl[boolean]` | Combine/cut meshes |
| Validation | `pip install numpy2stl[validation]` | Check mesh quality |
| Visualization | `pip install numpy2stl[visualization]` | View meshes |
| Everything | `pip install numpy2stl[full]` | All features (no viz) |

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

Optional dependencies are specified in `setup.py` extras_require.

## Contributing

### Running Tests

```bash
# Install development dependencies
pip install -e .[dev]

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

## API Changes

### Deprecated

- `numpy2stl()` function → Use `array_to_mesh()` instead

The old function name still works but will show a deprecation warning.

## License

MIT License (see LICENSE file)

## Related Projects

- **strm2stl**: Geographic terrain mesh generation from SRTM/DEM data
- **geo2stl**: Geographic data processing for 3D mapping

## Credits

Created by Edgar Cardenas De La Hoz

## Changelog

### v0.1.0 (2024)
- Improved dependency management with extras_require
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
