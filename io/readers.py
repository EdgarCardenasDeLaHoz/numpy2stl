"""STL/OBJ/3MF mesh loading utilities — used by the stl2numpy module."""

try:
    import trimesh

    HAS_TRIMESH = True
except ImportError:
    trimesh = None
    HAS_TRIMESH = False

import numpy as np

__all__ = ["load_mesh"]


def _load_trimesh_mesh(file_path: str):
    """Return a trimesh.Trimesh object (internal use by stl2numpy)."""
    if not HAS_TRIMESH:
        raise ImportError(
            "trimesh is required. Install with: pip install trimesh"
        )
    # Pre-warm format-specific loaders so trimesh's lazy-import cache
    # doesn't surface a stale ImportError from a prior failed attempt.
    # NB: use importlib so we don't rebind the module-level `trimesh` name
    # into a function local (which would shadow it below).
    ext = str(file_path).rsplit(".", 1)[-1].lower()
    if ext == "3mf":
        import importlib
        try:
            importlib.import_module("trimesh.exchange.threemf")
        except ImportError:
            pass
    mesh = trimesh.load(file_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(
            f"Could not load {file_path!r} as a single mesh. "
            "The file may contain multiple objects; export as a merged STL."
        )
    return mesh


def load_mesh(file_path: str):
    """
    Load a mesh file and return (vertices, faces) arrays.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces    : ndarray of int, shape (M, 3)
    """
    mesh = _load_trimesh_mesh(file_path)
    return np.array(mesh.vertices), np.array(mesh.faces)
