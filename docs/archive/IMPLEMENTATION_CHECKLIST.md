# Reorganization Implementation Checklist

**Status**: Ready to implement
**Estimated Time**: 3.5 hours
**Prerequisites**: Fresh token budget (50k+ recommended)

---

## Pre-Implementation

- [ ] Review REORGANIZATION_PROPOSAL.md in full
- [ ] Verify all decisions are confirmed:
  - [x] Option A (Functional Hierarchy)
  - [x] Conservative deprecation (1-2 versions)
  - [x] stl2numpy inside numpy2stl
  - [x] puzzle.py split confirmed
- [ ] Start new session with fresh tokens
- [ ] Run existing tests to establish baseline

---

## Phase 1: Folder Structure (15 min)

### Step 1.1: Create Directories
```bash
cd "d:\OneDrive\Documents\Projects\3D Maps\Code\numpy2stl"
mkdir core
mkdir io
mkdir processing
mkdir stl2numpy
mkdir utils
mkdir applications
```

- [ ] Directories created
- [ ] Verified with `dir` command

### Step 1.2: Create Empty __init__.py Files
```bash
type nul > core\__init__.py
type nul > io\__init__.py
type nul > processing\__init__.py
type nul > stl2numpy\__init__.py
type nul > utils\__init__.py
type nul > applications\__init__.py
```

- [ ] All `__init__.py` files created
- [ ] Files are accessible (use `view` to check)

---

## Phase 2: Copy Core Files (15 min)

### Step 2.1: Core Module Files
```bash
copy generate.py core\generate.py
copy solid.py core\solid.py
copy polygon.py core\polygon.py
```

- [ ] `core/generate.py` copied
- [ ] `core/solid.py` copied
- [ ] `core/polygon.py` copied
- [ ] Verified files exist in new location

### Step 2.2: Update Internal Imports in Core
Files to check: `core/generate.py`, `core/solid.py`, `core/polygon.py`

Change pattern:
- `from . import X` → `from ..core import X` OR `from . import X` (if within core)
- Check all relative imports

- [ ] `core/generate.py` imports updated
- [ ] `core/solid.py` imports updated
- [ ] `core/polygon.py` imports updated

### Step 2.3: Write Core __init__.py
Export main functions:
```python
from .generate import array_to_mesh, array2faces
from .solid import Solid, vertices_to_index, triangles_to_facets, get_open_edges
from .polygon import get_ordered_perimeter, triangulate_polygon, polygon_to_complex

__all__ = [
    "array_to_mesh",
    "array2faces",
    "Solid",
    "vertices_to_index",
    "triangles_to_facets",
    "get_open_edges",
    "get_ordered_perimeter",
    "triangulate_polygon",
    "polygon_to_complex",
]
```

- [ ] `core/__init__.py` written
- [ ] Test: `from numpy2stl.core import array_to_mesh`

---

## Phase 3: IO Module (15 min)

### Step 3.1: Copy and Rename
```bash
copy save.py io\writers.py
```

- [ ] `io/writers.py` created

### Step 3.2: Update Internal Imports
Check `io/writers.py` for relative imports

- [ ] Imports updated if needed

### Step 3.3: Write IO __init__.py
```python
from .writers import writeSTL, writeOBJ, write3MF

__all__ = ["writeSTL", "writeOBJ", "write3MF"]
```

- [ ] `io/__init__.py` written
- [ ] Test: `from numpy2stl.io import writeSTL`

### Step 3.4: Create Placeholder readers.py
```python
"""
STL/OBJ file readers for stl2numpy module.
To be implemented in stl2numpy development phase.
"""

__all__ = []
```

- [ ] `io/readers.py` created

---

## Phase 4: Processing Module (20 min)

### Step 4.1: Copy Files
```bash
copy simplify.py processing\simplify.py
copy boolean.py processing\boolean.py
copy verify.py processing\verify.py
```

- [ ] `processing/simplify.py` copied
- [ ] `processing/boolean.py` copied
- [ ] `processing/verify.py` copied

### Step 4.2: Update Internal Imports
Check all three files for relative imports:
- Change `from .solid import X` → `from ..core.solid import X`
- Change `from .simplify import X` → `from .simplify import X`

- [ ] `processing/simplify.py` imports updated
- [ ] `processing/boolean.py` imports updated
- [ ] `processing/verify.py` imports updated

### Step 4.3: Write Processing __init__.py
```python
from .simplify import simplify_mesh_surfaces, get_open_edges
from .boolean import union_pymesh, clean_mesh
from .verify import check_model_status, find_mesh_issues

__all__ = [
    "simplify_mesh_surfaces",
    "get_open_edges",
    "union_pymesh",
    "clean_mesh",
    "check_model_status",
    "find_mesh_issues",
]
```

- [ ] `processing/__init__.py` written
- [ ] Test: `from numpy2stl.processing import simplify_mesh_surfaces`

---

## Phase 5: Split puzzle.py (45 min) ⚠️ COMPLEX

### Step 5.1: Create processing/extrusion.py
Extract these functions from puzzle.py:
- `make_hollow_prism_solid()`
- `make_hollow_cap()`
- `extrude_solid_polygon()`
- `make_prism_solid()`
- `prism_wall_vertices()`
- `prism_wall_vertices_optimized()`
- `prism_wall_vertices_old()` (if keeping)
- `robust_triangulate()`

Update imports:
- `from . import solid` → `from ..core import solid`
- `from .simplify import get_open_edges` → `from .simplify import get_open_edges`
- `from .solid import vertices_to_index` → `from ..core.solid import vertices_to_index`

- [ ] `processing/extrusion.py` created with 6-8 functions
- [ ] Imports updated to use `..core`
- [ ] All necessary imports included (numpy, trimesh, shapely)

### Step 5.2: Create applications/puzzle.py
Keep these functions:
- `make_puzzle_pts()`
- `make_puzzle_piece()`
- `make_puzzle_model()`
- `make_base_border()`

Add imports:
```python
from ..processing.extrusion import (
    make_hollow_prism_solid,
    robust_triangulate,
    # ... other needed functions
)
from ..core import solid
from ..core.solid import vertices_to_index
```

- [ ] `applications/puzzle.py` created with 4 functions
- [ ] Imports from `processing.extrusion` added
- [ ] Imports from `core` updated

### Step 5.3: Update Processing __init__.py
Add extrusion exports:
```python
from .extrusion import (
    make_hollow_prism_solid,
    make_hollow_cap,
    extrude_solid_polygon,
    make_prism_solid,
    robust_triangulate,
)
```

- [ ] Extrusion exports added to `processing/__init__.py`

### Step 5.4: Test Split
```python
# Should work:
from numpy2stl.processing import make_hollow_prism_solid
from numpy2stl.applications import make_puzzle_model
```

- [ ] Processing extrusion functions import correctly
- [ ] Applications puzzle functions import correctly

---

## Phase 6: Utils Module (15 min)

### Step 6.1: Copy and Rename
```bash
copy tools.py utils\image.py
copy view.py utils\visualization.py
```

- [ ] `utils/image.py` created
- [ ] `utils/visualization.py` created

### Step 6.2: Update Internal Imports
- `utils/image.py`: Check for imports from polygon, view
- `utils/visualization.py`: Should be mostly self-contained

- [ ] `utils/image.py` imports updated
- [ ] `utils/visualization.py` imports updated

### Step 6.3: Write Utils __init__.py
```python
from .image import resize_max, rescale
from .visualization import plot_edges_3d, plot_perimeters

__all__ = [
    "resize_max",
    "rescale",
    "plot_edges_3d",
    "plot_perimeters",
]
```

- [ ] `utils/__init__.py` written
- [ ] Test: `from numpy2stl.utils import resize_max`

---

## Phase 7: Applications Module (10 min)

### Step 7.1: Copy oceans.py
```bash
copy oceans.py applications\oceans.py
```

- [ ] `applications/oceans.py` copied

### Step 7.2: Update Imports
- Change `from .generate import array_to_mesh` → `from ..core.generate import array_to_mesh`
- Change `from .save import writeSTL` → `from ..io.writers import writeSTL`
- Check all relative imports

- [ ] `applications/oceans.py` imports updated
- [ ] `applications/puzzle.py` imports verified (from Phase 5)

### Step 7.3: Write Applications __init__.py
```python
# Domain-specific applications - import on demand
__all__ = []
```

- [ ] `applications/__init__.py` written

---

## Phase 8: Update Main __init__.py (30 min)

### Step 8.1: Backup Current __init__.py
```bash
copy __init__.py __init__.py.backup
```

- [ ] Backup created

### Step 8.2: Rewrite __init__.py
New structure:
```python
"""numpy2stl — convert NumPy arrays and geometry into STL/OBJ/3MF for 3D printing"""

__version__ = "0.9.0"  # Updated for reorganization

# Import core functionality for top-level access
from .core.generate import array_to_mesh, array2faces
from .core.solid import (
    Solid,
    vertices_to_index,
    triangles_to_facets,
    get_open_edges,
    validate_object,
)
from .core.polygon import (
    get_ordered_perimeter,
    triangulate_polygon,
    polygon_to_complex,
)

# Import IO functions
from .io.writers import writeSTL, writeOBJ, write3MF

# Expose submodules for advanced usage
from . import core
from . import io
from . import processing
from . import stl2numpy
from . import utils
from . import applications

# Maintain existing __all__ exports
__all__ = [
    # Core mesh generation
    "array_to_mesh",
    "array2faces",
    
    # Solid/mesh utilities
    "Solid",
    "vertices_to_index",
    "triangles_to_facets",
    "get_open_edges",
    "validate_object",
    
    # Polygon operations
    "get_ordered_perimeter",
    "triangulate_polygon",
    "polygon_to_complex",
    
    # File I/O
    "writeSTL",
    "writeOBJ",
    "write3MF",
    
    # Submodules
    "core",
    "io",
    "processing",
    "stl2numpy",
    "utils",
    "applications",
]
```

- [ ] New `__init__.py` written
- [ ] All previous exports maintained
- [ ] Submodules exposed

### Step 8.3: Test Top-Level Imports
```python
# These should all work:
from numpy2stl import array_to_mesh
from numpy2stl import Solid
from numpy2stl import writeSTL
from numpy2stl.core import polygon
from numpy2stl.processing import simplify_mesh_surfaces
```

- [ ] Top-level imports work
- [ ] Submodule imports work
- [ ] No import errors

---

## Phase 9: Create Deprecation Wrappers (30 min)

### Step 9.1: Create Wrapper Template
Standard wrapper for each moved file:
```python
"""
DEPRECATED: This module has been moved.
Import from numpy2stl.<new_location> instead.
This file will be removed in v2.0.0.
"""
import warnings

# Import everything from new location
from .<new_location> import *

warnings.warn(
    f"Importing from numpy2stl.{__name__.split('.')[-1]} is deprecated. "
    f"Use 'from numpy2stl.<new_location> import ...' instead. "
    "This compatibility wrapper will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2
)
```

### Step 9.2: Create Wrappers at Root
Create these files (overwrite existing):
- `generate.py` → wrapper to `core.generate`
- `solid.py` → wrapper to `core.solid`
- `polygon.py` → wrapper to `core.polygon`
- `save.py` → wrapper to `io.writers`
- `simplify.py` → wrapper to `processing.simplify`
- `boolean.py` → wrapper to `processing.boolean`
- `verify.py` → wrapper to `processing.verify`
- `tools.py` → wrapper to `utils.image`
- `view.py` → wrapper to `utils.visualization`
- `oceans.py` → wrapper to `applications.oceans`
- `puzzle.py` → wrapper to `applications.puzzle` (note: now has different functions)

⚠️ **CAREFUL**: These will overwrite existing files. Ensure new structure is working first!

- [ ] Wrapper for `generate.py`
- [ ] Wrapper for `solid.py`
- [ ] Wrapper for `polygon.py`
- [ ] Wrapper for `save.py`
- [ ] Wrapper for `simplify.py`
- [ ] Wrapper for `boolean.py`
- [ ] Wrapper for `verify.py`
- [ ] Wrapper for `tools.py`
- [ ] Wrapper for `view.py`
- [ ] Wrapper for `oceans.py`
- [ ] Wrapper for `puzzle.py`

### Step 9.3: Test Deprecation Warnings
```python
import warnings
warnings.simplefilter("always", DeprecationWarning)

# Should show warning:
from numpy2stl.generate import array_to_mesh

# Should NOT show warning:
from numpy2stl.core.generate import array_to_mesh
from numpy2stl import array_to_mesh  # This uses new __init__.py
```

- [ ] Old imports show `DeprecationWarning`
- [ ] New imports work without warnings
- [ ] Top-level imports work without warnings

---

## Phase 10: Test Migration (30 min)

### Step 10.1: Run Existing Test Suite
```bash
cd tests
pytest -v
```

- [ ] All existing tests pass (44 tests expected)
- [ ] No import errors
- [ ] Benchmarks still work

### Step 10.2: Test Old Import Patterns
Create test file to verify backward compatibility:
```python
# test_backward_compatibility.py
import warnings
warnings.simplefilter("always", DeprecationWarning)

# Old style (should work with warnings)
from numpy2stl.generate import array_to_mesh as old_a2m
from numpy2stl.solid import Solid as old_Solid
from numpy2stl.save import writeSTL as old_writeSTL

# New style (should work without warnings)
from numpy2stl.core.generate import array_to_mesh as new_a2m
from numpy2stl.core.solid import Solid as new_Solid
from numpy2stl.io.writers import writeSTL as new_writeSTL

# Verify they're the same
assert old_a2m is new_a2m
assert old_Solid is new_Solid
assert old_writeSTL is new_writeSTL
```

- [ ] Old imports work (with warnings)
- [ ] New imports work (without warnings)
- [ ] Functions are identical

### Step 10.3: Test strm2stl Integration
Navigate to parent project and test imports:
```bash
cd "d:\OneDrive\Documents\Projects\3D Maps\Code\strm2stl"
python -c "import numpy2stl; print(numpy2stl.__version__)"
python -c "from numpy2stl import array_to_mesh; print('OK')"
```

- [ ] strm2stl can still import numpy2stl
- [ ] No fatal errors
- [ ] Deprecation warnings appear (expected)

---

## Phase 11: Reorganize Tests (20 min)

### Step 11.1: Create Test Subdirectories
```bash
cd tests
mkdir test_core
mkdir test_io
mkdir test_processing
mkdir test_utils
mkdir test_applications
```

- [ ] Test subdirectories created

### Step 11.2: Move/Copy Test Files
Map existing tests to new structure:
- `test_generate.py` → `test_core/test_generate.py`
- `test_solid.py` → `test_core/test_solid.py`
- `test_polygon.py` → `test_core/test_polygon.py`
- (Create new structure based on existing tests)

- [ ] Core tests organized
- [ ] Other tests organized or marked for future work

### Step 11.3: Update Test Imports
Change test imports to use new structure:
```python
# Old:
from numpy2stl.generate import array_to_mesh

# New (preferred):
from numpy2stl.core.generate import array_to_mesh
# OR (also works):
from numpy2stl import array_to_mesh
```

- [ ] Test imports updated
- [ ] Tests still pass

---

## Phase 12: Documentation Updates (30 min)

### Step 12.1: Update README.md
Sections to update:
- Installation (mention reorganization)
- Quick Start (use new import style)
- API Reference (show submodule structure)
- Add migration note for v0.9.0

- [ ] README.md updated

### Step 12.2: Update FUNCTIONALITY.md
Update import examples throughout:
```python
# Old examples
from numpy2stl.generate import array_to_mesh

# New examples  
from numpy2stl import array_to_mesh  # Top-level (recommended)
from numpy2stl.core import array_to_mesh  # Explicit
```

- [ ] FUNCTIONALITY.md updated

### Step 12.3: Create MIGRATION_GUIDE.md
Document the reorganization:
- What changed
- Old vs new import patterns
- Deprecation timeline
- Breaking changes (none for v0.9.0)

- [ ] MIGRATION_GUIDE.md created

### Step 12.4: Update setup.py Version
Change version to `0.9.0` to signal reorganization:
```python
setup(
    name="numpy2stl",
    version="0.9.0",  # Updated for reorganization
    ...
)
```

- [ ] Version updated in setup.py
- [ ] Version updated in __init__.py

---

## Phase 13: Final Validation (15 min)

### Step 13.1: Complete Test Run
```bash
cd "d:\OneDrive\Documents\Projects\3D Maps\Code\numpy2stl"
pytest tests/ -v --tb=short
```

- [ ] All tests pass
- [ ] No unexpected errors

### Step 13.2: Import Validation
Test comprehensive import matrix:
```python
# Top-level (new style, no warnings)
from numpy2stl import array_to_mesh, Solid, writeSTL

# Submodules (new style, no warnings)
from numpy2stl.core import generate
from numpy2stl.io import writers
from numpy2stl.processing import simplify, boolean, verify, extrusion
from numpy2stl.utils import image, visualization
from numpy2stl.applications import oceans, puzzle

# Old style (with warnings)
from numpy2stl.generate import array_to_mesh
from numpy2stl.save import writeSTL
```

- [ ] All import patterns work as expected
- [ ] Warnings appear only for deprecated paths

### Step 13.3: Quick Integration Test
Run a real example:
```python
import numpy as np
from numpy2stl import array_to_mesh, Solid

# Create test array
arr = np.random.rand(50, 50) * 10

# Generate mesh
vertices, faces = array_to_mesh(arr, solid=True)
print(f"Generated {len(vertices)} vertices, {len(faces)} faces")

# Save to file
solid = Solid((vertices, faces))
solid.save_stl("test_output.stl")
print("Saved successfully")
```

- [ ] Example runs without errors
- [ ] STL file created
- [ ] Output is valid

---

## Post-Implementation

### Documentation Deliverables
- [ ] MIGRATION_GUIDE.md exists and is comprehensive
- [ ] README.md reflects new structure
- [ ] FUNCTIONALITY.md uses new import style
- [ ] REORGANIZATION_PROPOSAL.md marked as implemented

### Code Deliverables
- [ ] New folder structure in place (7 folders)
- [ ] All files copied to new locations
- [ ] All `__init__.py` files written
- [ ] puzzle.py successfully split
- [ ] Deprecation wrappers at root
- [ ] Main `__init__.py` updated

### Testing Deliverables
- [ ] All 44 tests passing
- [ ] Backward compatibility verified
- [ ] strm2stl integration verified
- [ ] Test organization matches new structure

### Version Control
- [ ] Version bumped to 0.9.0
- [ ] All changes committed
- [ ] Git tag created: `v0.9.0-reorganization`

---

## Success Criteria

✅ **Implementation Successful When**:
1. All imports work (both old and new style)
2. All 44 tests pass
3. Deprecation warnings appear for old imports
4. New imports work without warnings
5. strm2stl can still import numpy2stl
6. Documentation reflects new structure
7. Ready to build stl2numpy in clean structure

---

## Rollback Plan

**If something goes wrong**:

1. **Restore from backup**:
   ```bash
   copy __init__.py.backup __init__.py
   ```

2. **Remove new folders**:
   ```bash
   rmdir /s core io processing stl2numpy utils applications
   ```

3. **Verify tests pass with original structure**

4. **Review what went wrong before trying again**

---

## Time Tracking

| Phase | Estimated | Actual | Notes |
|-------|-----------|--------|-------|
| 1. Folder structure | 15 min | | |
| 2. Core files | 15 min | | |
| 3. IO module | 15 min | | |
| 4. Processing module | 20 min | | |
| 5. Split puzzle.py | 45 min | | ⚠️ Most complex |
| 6. Utils module | 15 min | | |
| 7. Applications | 10 min | | |
| 8. Main __init__ | 30 min | | |
| 9. Deprecation wrappers | 30 min | | |
| 10. Test migration | 30 min | | |
| 11. Reorganize tests | 20 min | | |
| 12. Documentation | 30 min | | |
| 13. Final validation | 15 min | | |
| **Total** | **3h 30min** | | |

---

## Notes

- Implement in next session with >50k tokens
- Test frequently (after each phase)
- Keep original files until validation complete
- puzzle.py split is the most complex step - take time
- Deprecation wrappers are last - ensure new structure works first
