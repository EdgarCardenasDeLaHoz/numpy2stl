"""numpy2stl — convert NumPy arrays and geometry into STL/OBJ/3MF meshes.

Core functions are available directly::

    from numpy2stl import numpy2stl, perimeter_to_walls, Solid, writeSTL

Optional submodules with heavier dependencies are imported explicitly::

    import numpy2stl.simplify as simp   # needs scipy
    import numpy2stl.oceans   as oceans  # needs geo2stl, rasterio, h5py
    import numpy2stl.puzzle   as puzzle  # needs trimesh, shapely
    import numpy2stl.boolean  as boolean # needs pymeshlab
    import numpy2stl.view     as view    # needs matplotlib
"""
from .generate import *
from .tools import *
from .save import *
from .polygon import *
from .solid import *

# Backward-compatible alias — prefer array_to_mesh in new code
from .generate import array_to_mesh as numpy2stl  # noqa: F401
