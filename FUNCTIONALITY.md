# numpy2stl - Standalone Library Functionality

## Core Concept
**numpy2stl converts 2D NumPy arrays into 3D triangular meshes** (vertices + faces), with utilities for mesh manipulation and export to common 3D file formats.

---

## Main Functions

### 1. **Array-to-Mesh Conversion** (`array_to_mesh`)
Takes a 2D array where each value represents a height (Z-coordinate):

```python
import numpy as np
from numpy2stl import array_to_mesh

# 2D array: each value is a height
heights = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5]
])

vertices, faces = array_to_mesh(heights)
# vertices: Nx3 array of (x, y, z) coordinates
# faces: Mx3 array of triangle indices
```

**Options**:
- `solid=True` - Add walls and a bottom (makes it watertight/printable)
- `solid=False` - Just the top surface
- `mask_val` - Exclude values below threshold (creates holes)
- `floor_val` - Set bottom height independently

### 2. **Polygon Operations** (`polygon.py`)
- **`triangulate_polygon`** - Convert polygon outlines into triangle meshes
- **`rotate_3D`** - Rotate 3D points from one orientation to another
- **`get_ordered_perimeter`** - Extract ordered boundary edges from a mesh
- **`get_perimeter_angles`** - Calculate angles along polygon perimeters

### 3. **Mesh Utilities** (`solid.py`)
- **`vertices_to_index`** - Optimize mesh by deduplicating vertices
  ```python
  # Before: 1000 triangles = 3000 vertices (many duplicates)
  # After: 500 unique vertices + face indices
  vertices, faces = vertices_to_index(triangles)
  ```
- **`get_open_edges`** - Find boundary edges (non-manifold detection)
- **`triangles_to_facets`** - Convert triangles to STL facet format (with normals)
- **`validate_object`** - Check if mesh is watertight
- **`get_surfaces`** - Split mesh into connected components

### 4. **Export Functions** (`save.py`)
- **`writeSTL`** - Save as STL (binary or ASCII)
- **`write3MF`** - Save as 3MF (ZIP-based format)
- **`writeOBJ`** - Save as Wavefront OBJ

```python
from numpy2stl import Solid

solid = Solid((vertices, faces))
solid.save_stl("output.stl")          # Binary STL
solid.save_stl("output.stl", ascii=True)  # ASCII STL
```

### 5. **Image Tools** (`tools.py`)
Requires opencv:
- **`rescale`** - Resize + remap values to height range + percentile clipping
- **`resize_max`** - Resize image preserving aspect ratio

### 6. **Mesh Simplification** (`simplify.py`)
Requires scipy + triangle:
- **`simplify_mesh_surfaces`** - Reduce triangle count
- **`simplify_surface`** - Retriangulate planar surfaces
- Extract and remesh complex surfaces

### 7. **Boolean Operations** (`boolean.py`)
Requires pymeshlab or manifold3d:
- Union, difference, intersection of meshes
- Mesh healing and repair

### 8. **Mesh Generation** (`generate.py`)
- **`array2faces`** - Delaunay triangulation of masked 2D array
- **`perimeter_to_walls`** - Generate vertical walls from boundary
- **`polygon_to_complex`** - Extrude polygon to 3D shape
- **`polygon_to_prism`** - Create simple prism from polygon

### 9. **Verification** (`verify.py`)
Requires trimesh:
- Check mesh watertightness
- Validate STL files
- Detect and report mesh issues

---

## How It Works

### The Pipeline

```
2D Array → Triangulation → Vertex Indexing → STL Export
```

1. **Input**: 2D NumPy array (any values)
2. **Triangulation**: Creates triangular mesh from grid points
3. **Solidification** (optional): Adds walls + bottom
4. **Optimization**: Deduplicates vertices
5. **Export**: Writes STL/OBJ/3MF file

### Key Algorithms

**Vertex Deduplication**:
- Uses "void-view trick" to speed up `np.unique()`
- Reduces memory by ~70% (triangles share vertices)
- Optimized for large meshes (500k+ vertices)

**Delaunay Triangulation**:
- Uses scipy's `Delaunay` or Triangle library
- Handles masked regions (holes)
- Generates optimal triangle distribution

**Wall Generation**:
- Finds mesh boundary edges
- Orders them into closed loops
- Extrudes downward to create walls

---

## Use Cases (Generic)

1. **Data Visualization**
   - Turn any matrix into 3D surface
   - Visualize functions: z = f(x, y)
   - Statistical data surfaces

2. **CAD/Manufacturing**
   - Generate meshes from mathematical functions
   - Create parametric 3D shapes
   - Export for CNC/3D printing

3. **Scientific Computing**
   - Convert simulation results to 3D
   - Export computational geometry
   - Mesh generation for FEA preprocessing

4. **Game Development**
   - Procedural terrain generation
   - Height-based mesh creation
   - Quick prototyping of 3D environments

---

## Class: `Solid`

Wrapper class for mesh data:

```python
solid = Solid((vertices, faces))
# or
solid = Solid(raw_triangles)  # auto-indexes

solid.save_stl("file.stl")
solid.validate_object()  # Check for issues
```

---

## Dependencies

**Required**: numpy, shapely  
**Optional**:
- scipy (triangulation, simplification)
- opencv (image rescaling)
- triangle (advanced triangulation)
- matplotlib (visualization)
- pymeshlab/manifold3d (boolean ops)
- trimesh (validation)

---

## Technical Details

**Performance**:
- 100x100 array: ~30ms
- 500x500 array: ~760ms  
- 1000x1000 array: ~3.5s

**Bottleneck**: Vertex deduplication (60-65% of time)

**Output formats**:
- STL: Industry standard for 3D printing
- OBJ: Text-based, widely supported
- 3MF: Modern alternative to STL

**Mesh quality**:
- Generates watertight manifold meshes
- Suitable for 3D printing
- Proper normal calculation
- Handles non-convex shapes

---

## Summary

numpy2stl is a **mesh generation and manipulation library** that:
- Converts 2D arrays → 3D meshes
- Triangulates polygons
- Optimizes vertex data
- Exports to standard 3D formats
- Provides mesh analysis tools

It's essentially a **2D-to-3D converter with mesh utilities**, useful for anyone who needs to turn grid data into 3D geometry.

---

## Quick Examples

### Example 1: Simple Surface
```python
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Create any 2D function
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))  # Ripple pattern

# Convert to mesh and save
vertices, faces = array_to_mesh(Z, solid=True)
solid = Solid((vertices, faces))
solid.save_stl("ripple.stl")
```

### Example 2: Masked Region (Donut)
```python
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Create array with center cutout
arr = np.ones((100, 100)) * 10
center = 50
radius_outer = 40
radius_inner = 20

# Create donut shape
for i in range(100):
    for j in range(100):
        dist = np.sqrt((i - center)**2 + (j - center)**2)
        if dist < radius_inner or dist > radius_outer:
            arr[i, j] = -1  # Mask these areas

vertices, faces = array_to_mesh(arr, mask_val=0, floor_val=0)
solid = Solid((vertices, faces))
solid.save_stl("donut.stl")
```

### Example 3: Optimize Existing Mesh
```python
import numpy as np
from numpy2stl import vertices_to_index

# You have raw triangles with duplicate vertices
raw_triangles = np.array([...])  # Nx3x3 array

# Deduplicate vertices
vertices, faces = vertices_to_index(raw_triangles)

print(f"Reduced from {len(raw_triangles) * 3} to {len(vertices)} vertices")
```

### Example 4: Multiple Export Formats
```python
from numpy2stl import Solid, write3MF, writeOBJ

vertices, faces = array_to_mesh(data)

# Option 1: Using Solid class
solid = Solid((vertices, faces))
solid.save_stl("model.stl")

# Option 2: Direct export functions
models = {"model1": (vertices, faces)}
write3MF("model.3mf", models)
writeOBJ("model.obj", models)
```
