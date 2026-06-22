# stl2numpy - Feature Planning

## Core Concept
**Reverse operation of numpy2stl**: Take 3D mesh files (STL/OBJ/3MF) and convert them back to NumPy arrays and other reduced/normalized representations.

---

## Primary Use Cases

### 1. **Analysis & Reverse Engineering**
- Analyze existing 3D printed models
- Extract elevation data from terrain STLs
- Convert CAD models to processable arrays
- Quality control of manufactured parts

### 2. **Data Normalization**
- Convert arbitrary 3D formats to standard arrays
- Prepare meshes for machine learning
- Create training datasets from 3D models
- Standardize different mesh formats

### 3. **Modification Pipeline**
- STL → array → modify → STL
- Extract height map → edit → regenerate mesh
- Batch processing of 3D models

---

## Proposed Features

### Category A: Basic Conversions

#### 1. **mesh_to_heightmap** (Primary Feature)
Convert 3D mesh to 2D elevation array:
```python
from stl2numpy import mesh_to_heightmap

# Load STL and convert to height map
heightmap = mesh_to_heightmap("terrain.stl", resolution=500)
# Returns: 500x500 array of heights (top surface only)
```

**Parameters**:
- `resolution` - Output grid size (e.g., 500x500)
- `projection` - 'top' (default), 'bottom', 'max', 'min'
- `fill_method` - How to handle gaps: 'nearest', 'linear', 'idw'
- `bounds` - Manual XY bounds or auto-detect
- `orientation` - Auto-detect up axis or specify

**Challenges**:
- Multiple Z values at same XY (overhangs, caves)
- Missing data (gaps in mesh)
- Mesh orientation detection

#### 2. **mesh_to_voxels** 
Convert to 3D volumetric array:
```python
voxels = mesh_to_voxels("model.stl", resolution=(100, 100, 100))
# Returns: 100x100x100 boolean array (inside=True, outside=False)
```

**Use cases**:
- Volume calculations
- 3D convolutions/analysis
- Support structure detection
- Solid/hollow detection

#### 3. **mesh_to_pointcloud**
Extract raw vertex positions:
```python
points = mesh_to_pointcloud("model.stl", sample_density=1000)
# Returns: Nx3 array of point positions
```

**Options**:
- Sample faces uniformly
- Use existing vertices only
- Add normals (Nx6 array)

---

### Category B: Slicing & Cross-Sections

#### 4. **slice_mesh**
Extract 2D cross-sections:
```python
slices = slice_mesh("model.stl", z_levels=[0, 5, 10, 15])
# Returns: List of 2D contours at each Z level
```

**Use cases**:
- Generate print layers
- Analyze model cross-sections
- Create elevation contours
- Path planning

#### 5. **rasterize_slice**
Convert slice to 2D mask array:
```python
mask = rasterize_slice("model.stl", z=10, resolution=500)
# Returns: 500x500 boolean array (inside polygon = True)
```

---

### Category C: Mesh Analysis

#### 6. **get_mesh_properties**
Extract mesh statistics:
```python
props = get_mesh_properties("model.stl")
# Returns: {
#   'volume': 1234.5,
#   'surface_area': 567.8,
#   'bounds': {'x': (0, 100), 'y': (0, 100), 'z': (0, 50)},
#   'center_of_mass': (50, 50, 25),
#   'num_vertices': 10000,
#   'num_faces': 20000,
#   'is_watertight': True,
#   'num_holes': 0
# }
```

#### 7. **detect_orientation**
Auto-detect mesh orientation:
```python
orientation = detect_orientation("model.stl")
# Returns: {'up_axis': 'z', 'front_axis': 'y', 'rotation_needed': None}
```

---

### Category D: Mesh Reduction

#### 8. **decimate_mesh**
Reduce triangle count:
```python
reduced = decimate_mesh("model.stl", target_faces=5000)
# Or: target_ratio=0.5 (reduce to 50% of original)
```

#### 9. **remesh_uniform**
Create uniform triangle distribution:
```python
uniform_mesh = remesh_uniform("model.stl", target_edge_length=1.0)
```

---

### Category E: Advanced Conversions

#### 10. **mesh_to_sdf**
Signed distance field:
```python
sdf = mesh_to_sdf("model.stl", resolution=(100, 100, 100))
# Returns: 3D array where values = distance to surface (negative inside)
```

**Use cases**:
- Implicit surface representation
- Collision detection
- Morphological operations

#### 11. **mesh_to_density**
Sample mesh as density field:
```python
density = mesh_to_density("model.stl", resolution=100, kernel_size=3)
# Returns: Smooth density representation
```

---

## Architecture Design

### Module Structure
```
stl2numpy/
├── __init__.py          # Main exports
├── io.py                # Load STL/OBJ/3MF files
├── heightmap.py         # mesh_to_heightmap, projection methods
├── voxelize.py          # mesh_to_voxels, mesh_to_sdf
├── slice.py             # slice_mesh, rasterize_slice
├── pointcloud.py        # mesh_to_pointcloud, sampling
├── analysis.py          # get_mesh_properties, detect_orientation
├── reduction.py         # decimate_mesh, remesh_uniform
└── utils.py             # Helper functions
```

### Dependencies
**Required**:
- `numpy` - Array operations
- `trimesh` - Mesh loading and operations (best Python mesh library)

**Optional**:
- `scipy` - Interpolation, distance calculations
- `open3d` - Advanced voxelization, point cloud ops
- `pyvista` - Visualization and slicing
- `scikit-image` - Rasterization
- `pymeshlab` - Decimation, remeshing

---

## Key Decisions Needed

### 1. **Handling Overhangs**
When projecting to 2D heightmap, what to do when multiple Z values exist at same XY?

**Options**:
- `projection='max'` - Take highest point (default, most common)
- `projection='min'` - Take lowest point
- `projection='mean'` - Average all points
- `projection='closest'` - Closest to specified Z
- `raise_error` - Fail if overhangs detected

### 2. **Gap Filling**
When mesh doesn't cover entire grid:

**Options**:
- `fill_method='nearest'` - Use nearest mesh point
- `fill_method='linear'` - Linear interpolation
- `fill_method='idw'` - Inverse distance weighting
- `fill_method='nan'` - Leave as NaN (user handles)

### 3. **Mesh Orientation**
How to determine "up" direction:

**Options**:
- Auto-detect (longest dimension, gravity test)
- User-specified axis
- Use metadata if available
- Interactive prompt with visualization

### 4. **Performance vs Quality**
Trade-offs for large meshes:

**Strategies**:
- Subsample mesh before rasterization
- Use spatial indexing (octree, KD-tree)
- Parallel processing for voxelization
- Progressive refinement

---

## Example Workflows

### Workflow 1: Reverse Engineering
```python
from stl2numpy import mesh_to_heightmap
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Load existing STL and extract heightmap
heightmap = mesh_to_heightmap("terrain_scan.stl", resolution=500)

# Smooth the data
from scipy.ndimage import gaussian_filter
smoothed = gaussian_filter(heightmap, sigma=2)

# Regenerate mesh with smoothing
vertices, faces = array_to_mesh(smoothed, solid=True)
solid = Solid((vertices, faces))
solid.save_stl("terrain_smoothed.stl")
```

### Workflow 2: Analysis Pipeline
```python
from stl2numpy import get_mesh_properties, mesh_to_heightmap

# Analyze model
props = get_mesh_properties("model.stl")
print(f"Volume: {props['volume']:.2f} mm³")
print(f"Print material estimate: {props['volume'] * 1.2:.2f} g")

# Extract elevation for visualization
heightmap = mesh_to_heightmap("model.stl", resolution=1000)

import matplotlib.pyplot as plt
plt.imshow(heightmap, cmap='terrain')
plt.colorbar(label='Height (mm)')
plt.savefig("heightmap.png")
```

### Workflow 3: Batch Normalization
```python
from stl2numpy import mesh_to_heightmap
import glob

# Process all STL files in a directory
for stl_file in glob.glob("models/*.stl"):
    heightmap = mesh_to_heightmap(stl_file, resolution=256)
    output = stl_file.replace(".stl", ".npy")
    np.save(output, heightmap)
    print(f"Converted {stl_file} → {output}")
```

---

## Implementation Priority

### Phase 1: Core Functionality (MVP)
1. ✨ `mesh_to_heightmap` - Primary feature
2. ✨ `get_mesh_properties` - Analysis
3. ✨ Basic I/O (load STL/OBJ)

### Phase 2: Advanced Conversions
4. `mesh_to_voxels` - 3D arrays
5. `mesh_to_pointcloud` - Sampling
6. `slice_mesh` - Cross-sections

### Phase 3: Utilities
7. `detect_orientation` - Auto-detection
8. `decimate_mesh` - Reduction
9. Gap filling strategies

### Phase 4: Advanced Features
10. `mesh_to_sdf` - Signed distance
11. `mesh_to_density` - Density fields
12. Performance optimizations

---

## Design Decisions (RESOLVED) ✅

**See `stl2numpy_CONFIG.md` for detailed answers**

1. **Primary use case**: All 3 (terrain, general processing, ML) - start simple with array conversion
2. **Target mesh sizes**: Large (> 100k faces) - optimize for performance
3. **Overhangs handling**: Take max (highest point) - `projection='max'` default
4. **Gap filling**: Leave as NaN (user interpolates later if needed)
5. **Output resolution**: Auto-detect from mesh density, cap at 1000x1000

**Default behavior**:
```python
heightmap = mesh_to_heightmap("model.stl")  
# - Auto-detect resolution (capped at 1000x1000)
# - Take highest point for overhangs
# - NaN for points outside mesh
# - Return dict with metadata
```

## Remaining Questions

4. **Integration with numpy2stl?**
   - Seamless round-trip conversion?
   - Shared utilities?
   - Separate or integrated CLI?

5. **Visualization?**
   - Built-in preview functions?
   - External tool integration?
   - Export to image formats?

---

## Success Criteria

A successful stl2numpy module should:
- ✅ Load common 3D formats (STL, OBJ, 3MF)
- ✅ Convert to height map in < 5 seconds for typical meshes
- ✅ Handle edge cases (overhangs, gaps, orientation)
- ✅ Produce arrays compatible with numpy2stl for round-trip
- ✅ Provide useful mesh analysis tools
- ✅ Clear error messages for invalid inputs
- ✅ Memory-efficient for large meshes

---

## Next Steps

1. Review and refine feature list
2. Prioritize features (MVP vs future)
3. Choose dependency strategy (trimesh vs open3d vs custom)
4. Create initial module structure
5. Implement Phase 1 (heightmap + properties)
6. Write tests for edge cases
7. Document with examples
8. Integrate with numpy2stl workflow
