"""numpy2stl — convert NumPy arrays and geometry into STL/OBJ/3MF meshes.

This is the canonical flat package.  Import directly::

    from numpy2stl import numpy2stl, perimeter_to_walls, Solid
    import numpy2stl.simplify as simp
    import numpy2stl.puzzle   as puzzle
"""
from .generate import *
from .view import *
from .tools import *
from .save import *
from .polygon import *
from .solid import *
from .oceans import *
