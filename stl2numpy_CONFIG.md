# stl2numpy - User Preferences & Configuration

**Date**: 2026-06-07  
**Project**: numpy2stl/stl2numpy module

---

## Design Decisions (Answered)

### 1. Primary Use Case
**Answer**: All 3 and more - versatile functionality
- ✅ Terrain analysis (height maps from terrain STLs)
- ✅ General 3D model processing
- ✅ ML training data preparation
- ✅ Plus additional use cases as they emerge

**Starting Point**: Simple array conversion (reverse of array_to_mesh)

---

### 2. Typical Mesh Sizes
**Answer**: Large meshes (to be safe)
- Plan for: > 100k triangles
- Performance target: Handle large meshes efficiently
- Memory consideration: May need streaming or chunking for very large files

---

### 3. Handling Overhangs/Caves
**Question**: When multiple Z values exist at same XY coordinate:

**Answer**: Take max (highest point)
- `projection='max'` as default behavior
- Most intuitive for terrain and general use
- User can override if needed for special cases

**Implementation**:
```python
heightmap = mesh_to_heightmap("model.stl", projection='max')  # default
heightmap = mesh_to_heightmap("model.stl", projection='min')  # override
heightmap = mesh_to_heightmap("model.stl", projection='mean') # average
```

---

### 4. Gap Filling Strategy
**Question**: When mesh doesn't cover entire grid:

**Answer**: Leave as NaN
- Especially for points outside the mesh boundaries
- Allows user to interpolate afterward if needed
- Clear distinction between "no data" vs "zero height"
- Prevents automatic assumptions about missing data

**Implementation**:
```python
heightmap = mesh_to_heightmap("model.stl")  # NaN for gaps
# User can then choose interpolation:
from scipy.interpolate import griddata
# ... fill NaNs as needed
```

**Future option** (Phase 2):
```python
heightmap = mesh_to_heightmap("model.stl", fill_method='nearest')  # optional
heightmap = mesh_to_heightmap("model.stl", fill_method='linear')
```

---

### 5. Default Output Resolution
**Answer**: Match input mesh density, but cap at 1000x1000

**Logic**:
- Analyze mesh bounding box and face density
- Calculate appropriate grid resolution
- Cap at 1000x1000 to prevent memory issues
- User can override with explicit `resolution` parameter

**Implementation**:
```python
# Auto-detect (capped at 1000x1000)
heightmap = mesh_to_heightmap("model.stl")

# Explicit resolution
heightmap = mesh_to_heightmap("model.stl", resolution=512)
heightmap = mesh_to_heightmap("model.stl", resolution=(512, 512))

# Very high resolution (user's responsibility)
heightmap = mesh_to_heightmap("model.stl", resolution=2000, allow_large=True)
```

**Resolution calculation algorithm**:
```python
def calculate_resolution(mesh):
    bounds = mesh.bounds  # Get XYZ bounds
    xy_size = bounds[1][:2] - bounds[0][:2]  # X, Y dimensions
    
    # Estimate face density
    face_density = len(mesh.faces) / (xy_size[0] * xy_size[1])
    
    # Calculate grid size based on density
    suggested = int(np.sqrt(face_density)) * 10
    
    # Cap at 1000
    resolution = min(suggested, 1000)
    
    return resolution
```

---

## Additional Questions to Consider

### 6. Mesh Orientation Detection
**Question**: How to determine which axis is "up"?

**Options**:
- Auto-detect (analyze bounds, gravity, face normals)
- User-specified axis ('z', 'y', 'x')
- Prompt with visualization

**Suggested default**: Auto-detect with fallback to Z-axis

---

### 7. Output Format Preferences
**Question**: What NumPy array properties?

**Suggested**:
- dtype: `float64` (match numpy2stl)
- NaN handling: Use `np.nan` (not masked arrays)
- Coordinate system: Preserve original mesh coordinates

---

### 8. Error Handling
**Question**: What to do when:
- Invalid STL file?
- Mesh has holes?
- Mesh is not oriented correctly?

**Suggested**:
- Raise clear exceptions
- Provide validation function
- Warn but continue if non-critical

---

### 9. Metadata Preservation
**Question**: Should we preserve mesh metadata?

**Options**:
- Return just array
- Return dict: `{'heightmap': array, 'bounds': ..., 'resolution': ..., 'units': ...}`
- Return custom class with attributes

**Suggested**: Return dict for flexibility

---

### 10. CLI Integration
**Question**: Should stl2numpy have CLI tools?

**Example**:
```bash
stl2numpy convert terrain.stl --output terrain.npy --resolution 500
stl2numpy analyze model.stl --properties
```

**Later consideration** (Phase 3)

---

## Summary of Configuration

```python
# Default behavior based on user preferences
DEFAULT_CONFIG = {
    'projection': 'max',           # Take highest point for overhangs
    'fill_method': None,           # Leave gaps as NaN
    'resolution': 'auto',          # Auto-detect, cap at 1000x1000
    'max_resolution': 1000,        # Safety cap
    'orientation': 'auto',         # Auto-detect up axis
    'output_format': 'dict',       # Return dict with metadata
    'dtype': np.float64,           # Match numpy2stl
    'warn_on_issues': True,        # Warn about non-watertight, etc.
}
```

---

## Next Steps

1. ✅ User preferences documented
2. Update planning doc with decisions
3. Create initial module structure
4. Implement Phase 1: mesh_to_heightmap + get_mesh_properties
5. Write tests with example STL files
6. Document usage with examples

---

## Open Questions (For Future)

- [ ] Should we support 16-bit heightmaps for memory efficiency?
- [ ] Parallel processing for very large meshes?
- [ ] Support for colored meshes (preserve vertex colors)?
- [ ] Integration with numpy2stl CLI?
- [ ] Export to other formats (GeoTIFF, HDF5)?
