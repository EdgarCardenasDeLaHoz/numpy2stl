# numpy2stl Reorganization Proposal

## ✅ DECISION: Option A Confirmed

**Status**: Ready for implementation in next session (requires fresh token budget)
**Selected**: Functional Hierarchy (Option A)
**Timeline**: ~3.5 hours implementation
**Breaking Changes**: None (backward compatible with deprecation wrappers)

## Current Structure Analysis

### Current Files (13 modules at root)

| File | Lines | Purpose | Category |
|------|-------|---------|----------|
| **generate.py** | 233 | Core mesh generation (array_to_mesh) | 🔵 Core |
| **solid.py** | 364 | Mesh utilities, vertex deduplication, Solid class | 🔵 Core |
| **polygon.py** | 185 | 2D polygon operations, triangulation, perimeter ordering | 🔵 Core |
| **save.py** | 115 | File format writers (STL, OBJ, 3MF) | 💾 IO |
| **simplify.py** | 304 | Mesh simplification algorithms | 🔧 Processing |
| **boolean.py** | 94 | PyMeshLab boolean operations (union, intersection) | 🔧 Processing |
| **verify.py** | 81 | Mesh validation and repair (with trimesh) | 🔧 Processing |
| **tools.py** | 62 | Image utilities (resize, rescale) | 🛠️ Utilities |
| **view.py** | 94 | Visualization (matplotlib 3D plotting) | 📊 Visualization |
| **oceans.py** | 276 | **Domain-specific: Ocean/terrain processing** | 🌊 Apps |
| **puzzle.py** | 291 | **Domain-specific: Puzzle piece generation** | 🧩 Apps + 🔧 Processing |
| **__init__.py** | 87 | Package entry point | - |
| **setup.py** | - | Package configuration | - |

---

## Problems with Current Structure

1. **Flat namespace**: All 13 modules at root → hard to navigate
2. **Mixing concerns**: 
   - Core functionality (generate, solid, polygon) mixed with
   - Optional features (boolean, verify) mixed with
   - Domain apps (oceans, puzzle)
3. **Import confusion**: All imports from top level → naming collisions
4. **No clear boundaries**: Hard to know what's essential vs optional
5. **Future growth**: Adding stl2numpy will make root even more cluttered

---

## Proposed Organization

### Option A: Functional Hierarchy (RECOMMENDED)

```
numpy2stl/
├── __init__.py              # Main exports from core/
├── setup.py
├── README.md
├── FUNCTIONALITY.md
├── requirements.txt
│
├── core/                    # ⭐ Essential functionality
│   ├── __init__.py         # Export: array_to_mesh, Solid, etc.
│   ├── generate.py         # Mesh generation from arrays
│   ├── solid.py            # Mesh utilities, vertex deduplication
│   └── polygon.py          # 2D geometry operations
│
├── io/                      # 💾 File format handling
│   ├── __init__.py         # Export: writeSTL, writeOBJ, write3MF, loadSTL
│   ├── writers.py          # STL/OBJ/3MF export (current save.py)
│   └── readers.py          # NEW: STL/OBJ import (for stl2numpy)
│
├── processing/              # 🔧 Advanced mesh operations
│   ├── __init__.py
│   ├── simplify.py         # Mesh simplification
│   ├── boolean.py          # Boolean operations (requires pymeshlab)
│   ├── verify.py           # Validation and repair (requires trimesh)
│   └── extrusion.py        # 3D extrusion utilities (from puzzle.py)
│
├── stl2numpy/               # 🔄 NEW: Reverse operations (STL → arrays)
│   ├── __init__.py
│   ├── heightmap.py        # STL → 2D elevation array
│   ├── voxels.py           # STL → 3D voxel grid
│   ├── analysis.py         # Mesh property extraction
│   └── slicing.py          # Cross-section analysis
│
├── utils/                   # 🛠️ Helper functions
│   ├── __init__.py
│   ├── image.py            # Image tools (resize, rescale) from tools.py
│   └── visualization.py    # Plotting helpers from view.py
│
├── applications/            # 🎯 Domain-specific use cases
│   ├── __init__.py
│   ├── oceans.py           # Ocean/terrain processing
│   └── puzzle.py           # Puzzle piece generation
│
└── tests/
    ├── test_core/
    ├── test_io/
    ├── test_processing/
    ├── test_stl2numpy/
    └── benchmarks/
```

**Import Examples**:
```python
# Core functionality (most common)
from numpy2stl import array_to_mesh, Solid, writeSTL

# Advanced operations
from numpy2stl.processing import simplify_mesh, union_meshes

# New stl2numpy module
from numpy2stl.stl2numpy import mesh_to_heightmap, analyze_mesh

# Domain apps
from numpy2stl.applications import process_ocean_terrain
```

---

### Option B: Minimal Restructure (Conservative)

```
numpy2stl/
├── __init__.py
├── setup.py
│
├── generate.py             # Keep at root (core)
├── solid.py                # Keep at root (core)
├── polygon.py              # Keep at root (core)
├── save.py                 # Keep at root (core)
│
├── ops/                    # NEW: Advanced operations
│   ├── simplify.py
│   ├── boolean.py
│   └── verify.py
│
├── stl2numpy/              # NEW: Reverse functionality
│   └── [as above]
│
├── apps/                   # NEW: Domain-specific
│   ├── oceans.py
│   └── puzzle.py
│
└── utils/                  # NEW: Helpers
    ├── tools.py
    └── view.py
```

**Pros**: Less disruptive, keeps core at root
**Cons**: Still somewhat cluttered, doesn't scale well

---

## Detailed Rationale

### Why Option A is Better

1. **Clear Mental Model**: 
   - `core/` = always needed
   - `processing/` = optional advanced features
   - `stl2numpy/` = reverse operations
   - `applications/` = specific use cases

2. **Dependency Isolation**:
   - Core has minimal deps (numpy, shapely)
   - Processing requires heavy libs (pymeshlab, trimesh)
   - Users only install what they need

3. **Scalability**:
   - Easy to add new modules (stl2numpy is just one folder)
   - Applications folder can grow without cluttering

4. **Documentation**:
   - Folder structure documents the architecture
   - New users immediately understand what's core vs optional

5. **Import Clarity**:
   - `from numpy2stl import X` → core functionality
   - `from numpy2stl.processing import Y` → advanced
   - `from numpy2stl.stl2numpy import Z` → reverse ops

---

## Migration Strategy

### Phase 1: Create New Structure (Non-Breaking)
1. Create new folders: `core/`, `io/`, `processing/`, `utils/`, `applications/`
2. **Copy** (not move) files to new locations
3. Add `__init__.py` to each folder with appropriate exports
4. Keep original files at root as **deprecated wrappers**

**Example wrapper** (root `generate.py`):
```python
"""
DEPRECATED: Import from numpy2stl.core.generate instead.
This module will be removed in v2.0.0.
"""
import warnings
from .core.generate import *

warnings.warn(
    "Importing from numpy2stl.generate is deprecated. "
    "Use 'from numpy2stl.core import array_to_mesh' or 'from numpy2stl import array_to_mesh'",
    DeprecationWarning,
    stacklevel=2
)
```

### Phase 2: Update Main __init__.py
```python
# numpy2stl/__init__.py

# Core exports (users import from top level)
from .core.generate import array_to_mesh, array2faces
from .core.solid import Solid, vertices_to_index, triangles_to_facets
from .core.polygon import (
    get_ordered_perimeter,
    triangulate_polygon,
    polygon_to_complex,
)

# IO exports
from .io.writers import writeSTL, writeOBJ, write3MF

# Convenience: also expose submodules
from . import core, io, processing, stl2numpy, utils, applications

__all__ = [
    # Core
    "array_to_mesh",
    "Solid",
    "vertices_to_index",
    # ... (full list)
    
    # Submodules
    "core",
    "io",
    "processing",
    "stl2numpy",
]
```

### Phase 3: Update Documentation
- Update README with new import examples
- Add migration guide
- Update FUNCTIONALITY.md with new structure

### Phase 4: Deprecation Timeline
- **v1.0.0**: New structure, old imports work with warnings
- **v1.5.0**: Warnings become louder, document removal date
- **v2.0.0**: Remove root-level deprecated files

---

## Backward Compatibility

### What Still Works
```python
# Old imports (with deprecation warning)
from numpy2stl import array_to_mesh        # ✅ Works
from numpy2stl.generate import array2faces # ✅ Works (via wrapper)
import numpy2stl.solid                      # ✅ Works (via wrapper)

# New imports (preferred)
from numpy2stl import array_to_mesh        # ✅ Same as above!
from numpy2stl.core.generate import array2faces  # ✅ Direct
import numpy2stl.core.solid                # ✅ Direct
```

### What Breaks (Only Internal Imports)
```python
# If oceans.py does: from . import generate
# Must change to: from .core import generate
# (We fix this in Phase 1)
```

---

## stl2numpy Integration

### Folder Contents (Proposed)

```
stl2numpy/
├── __init__.py             # Main exports
├── config.py               # Configuration (resolution, projection, gaps)
├── loaders.py              # STL/OBJ file loading (wraps trimesh)
├── heightmap.py            # mesh_to_heightmap()
├── voxels.py               # mesh_to_voxels()
├── pointcloud.py           # mesh_to_pointcloud()
├── slicing.py              # slice_mesh_at_z(), get_cross_section()
├── analysis.py             # get_mesh_properties(), get_bounds()
├── reduction.py            # decimate_mesh(), simplify_to_budget()
└── advanced.py             # mesh_to_sdf(), mesh_to_density_field()
```

**Key Exports**:
```python
from numpy2stl.stl2numpy import (
    mesh_to_heightmap,      # Main function (Phase 1)
    get_mesh_properties,    # Analysis helper (Phase 1)
    mesh_to_voxels,         # Phase 2
    slice_mesh_at_z,        # Phase 2
    # ... etc
)
```

**Symmetry with numpy2stl**:
- `numpy2stl.array_to_mesh()` → `stl2numpy.mesh_to_heightmap()`
- `numpy2stl.Solid` → `stl2numpy.MeshAnalyzer` (maybe?)
- `numpy2stl.writeSTL()` → `stl2numpy.load_stl()` (via io.readers)

---

## File Mappings

### Detailed Move Plan

| Current File | New Location | Rationale |
|--------------|--------------|-----------|
| `generate.py` | `core/generate.py` | Core mesh generation |
| `solid.py` | `core/solid.py` | Essential mesh utilities |
| `polygon.py` | `core/polygon.py` | Core 2D operations |
| `save.py` | `io/writers.py` | File output |
| `simplify.py` | `processing/simplify.py` | Optional advanced feature |
| `boolean.py` | `processing/boolean.py` | Requires pymeshlab |
| `verify.py` | `processing/verify.py` | Requires trimesh |
| `tools.py` | `utils/image.py` | Helper utilities |
| `view.py` | `utils/visualization.py` | Plotting helpers |
| `oceans.py` | `applications/oceans.py` | Domain-specific |
| `puzzle.py` | **SPLIT**: `processing/extrusion.py` + `applications/puzzle.py` | General tools + domain-specific |

### New Files to Create

| File | Purpose |
|------|---------|
| `io/readers.py` | STL/OBJ loading for stl2numpy |
| `processing/extrusion.py` | Extrusion utilities split from puzzle.py |
| `stl2numpy/[8 files]` | Reverse operations module |
| `core/__init__.py` | Core exports |
| `io/__init__.py` | IO exports |
| `processing/__init__.py` | Processing exports |
| `utils/__init__.py` | Utils exports |
| `applications/__init__.py` | App exports |

---

## Implementation Plan

### Step 1: Create Folder Structure (15 min)
```bash
mkdir core io processing stl2numpy utils applications
touch core/__init__.py io/__init__.py processing/__init__.py
touch stl2numpy/__init__.py utils/__init__.py applications/__init__.py
```

### Step 2: Copy Core Files (15 min)
```bash
cp generate.py core/
cp solid.py core/
cp polygon.py core/
```

### Step 3: Copy & Rename Others (25 min)
```bash
cp save.py io/writers.py
cp simplify.py boolean.py verify.py processing/
cp tools.py utils/image.py
cp view.py utils/visualization.py
cp oceans.py applications/

# Special: Split puzzle.py
# - Extract extrusion functions to processing/extrusion.py
# - Keep puzzle-specific functions in applications/puzzle.py
```

### Step 4: Create __init__.py Files (30 min)
- Write exports for each submodule
- Create deprecation wrappers at root

### Step 5: Update Main __init__.py (20 min)
- Import from new locations
- Preserve all public exports

### Step 6: Fix Internal Imports (30 min)
- Update relative imports in moved files
- Test that everything still works

### Step 7: Update Tests (20 min)
- Reorganize test files to match new structure
- Update test imports

### Step 8: Documentation (1 hour)
- Update README with new structure
- Create migration guide
- Update FUNCTIONALITY.md

**Total Time**: ~3.5 hours (including puzzle.py split)

---

## Decision Points ✅ ALL RESOLVED

### Question 1: Which Option? ✅
- **✅ Option A** (Functional Hierarchy) - **CONFIRMED**
- ~~Option B~~ (Minimal Restructure) - Not selected

### Question 2: Deprecation Strategy ✅
- ~~Aggressive~~: Remove old files in 3 months
- **✅ Conservative**: Keep wrappers for 1-2 major versions - **CONFIRMED**
- ~~Permanent~~: Never remove, just discourage

**Timeline**:
- v0.9.0: Wrappers with `DeprecationWarning`
- v1.0.0: Upgrade to `FutureWarning` (3-6 months)
- v2.0.0: Remove wrappers (6-12 months)

### Question 3: stl2numpy Placement ✅
- **✅ Inside numpy2stl** (as `numpy2stl.stl2numpy`) - **CONFIRMED**
- ~~Separate package~~ (as `stl2numpy`) - Not selected

### Question 4: When to Reorganize? ✅
- **✅ Now** (before building stl2numpy) - **CONFIRMED** (Next session)
- ~~After~~ (stl2numpy first, reorganize later) - Not selected

### Question 5: puzzle.py Handling? ✅
- **✅ Split**: `processing/extrusion.py` + `applications/puzzle.py` - **CONFIRMED**
- ~~Keep in processing/~~ - Not selected
- ~~Keep in applications/~~ - Not selected

---

## Risks & Mitigation

### Risk 1: Breaking Existing Code
**Mitigation**: 
- Keep wrappers at root
- All current imports still work
- Deprecation warnings guide users

### Risk 2: Larger Project (strm2stl) Breaks
**Mitigation**:
- Audit all imports from strm2stl to numpy2stl
- Update before deprecating
- Test integration

### Risk 3: Confusion During Transition
**Mitigation**:
- Clear documentation
- Migration guide with examples
- Version number signals change (bump to 1.0.0)

---

## Recommendation

**Go with Option A** for these reasons:
1. Clean slate for stl2numpy integration
2. Scales to future growth
3. Clear separation of concerns
4. Modern package structure
5. Time investment pays off long-term

**Timeline**:
1. Reorganize now (3 hours)
2. Build stl2numpy in clean structure (5-10 hours)
3. Deprecate old structure over 6 months
4. Remove wrappers in v2.0.0

**Special Case - puzzle.py Split**:
- Split into two files to separate concerns
- `processing/extrusion.py` gets reusable geometry tools:
  - `make_hollow_prism_solid()` - Create hollow 3D shape from 2D polygon
  - `make_hollow_cap()` - Create ring geometry (outer + inner offset)
  - `extrude_solid_polygon()` - Extrude 2D polygon to 3D solid
  - `make_prism_solid()` - Create prism from 2D points
  - `prism_wall_vertices()` - Generate wall triangles for extrusion
  - `robust_triangulate()` - Constrained Delaunay triangulation
- `applications/puzzle.py` keeps puzzle-specific functions:
  - `make_puzzle_pts()` - Generate puzzle piece shape points
  - `make_puzzle_piece()` - Create interlocking puzzle geometry
  - `make_puzzle_model()` - High-level puzzle generation
  - `make_base_border()` - Puzzle borders
  - Imports from `processing.extrusion`

**Next Steps**:
1. ✅ User confirmed Option A
2. ✅ User confirmed splitting puzzle.py
3. Implement Step 1-8 (reorganization with puzzle.py split)
4. Start stl2numpy development in new structure
