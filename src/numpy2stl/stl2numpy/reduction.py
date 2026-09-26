"""decimate_mesh: reduce triangle count while preserving shape."""

from __future__ import annotations

import logging

import numpy as np

from ..io.readers import _load_trimesh_mesh

logger = logging.getLogger(__name__)


def decimate_trimesh(mesh, target_faces: int, preserve: bool = False):
    """Quadric-decimate an **in-memory** trimesh to `target_faces`, returning a
    new ``trimesh.Trimesh``.

    Shared primitive for both `decimate_mesh` (file path) and the
    tolerance-bounded simplifier (processing/building_simplify.py).

    ``preserve=True`` runs pymeshlab's quadric edge collapse with
    feature-preservation flags (boundary / normal / topology / planar quadrics,
    max quality threshold).  This is essential on complex multi-body *city*
    meshes: plain decimation collapses tall thin buildings (tens of units of
    surface error), whereas the preserving variant keeps building height and
    footprint (orders of magnitude lower Hausdorff).  ``preserve=False`` keeps
    the legacy fast path (trimesh → pymeshlab) used by `decimate_mesh`.
    """
    import trimesh

    target_faces = max(4, int(target_faces))
    if target_faces >= len(mesh.faces):
        return mesh.copy()

    def _pymeshlab(**kw):
        import pymeshlab as ml
        ms = ml.MeshSet()
        ms.add_mesh(ml.Mesh(
            vertex_matrix=np.array(mesh.vertices, dtype=np.float64),
            face_matrix=np.array(mesh.faces, dtype=np.int32),
        ))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces), **kw)
        result = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=result.vertex_matrix().astype(np.float64),
            faces=result.face_matrix().astype(np.int64),
            process=False,
        )

    if preserve:
        return _pymeshlab(qualitythr=1.0, preserveboundary=True, preservenormal=True,
                          preservetopology=True, planarquadric=True, autoclean=True)

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if len(simplified.faces) > 0:
            return simplified
        raise RuntimeError("empty decimation result")
    except Exception:
        return _pymeshlab()


def decimate_mesh(
    file_path: str,
    target_faces: int | None = None,
    target_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce the triangle count of a mesh via quadric error decimation.

    Parameters
    ----------
    file_path : str
        Path to an STL, OBJ, or 3MF file.
    target_faces : int or None
        Absolute number of output faces.  When None, *target_ratio* is used.
    target_ratio : float
        Fraction of original faces to keep (default 0.5 = 50 %).
        Ignored if *target_faces* is given.

    Returns
    -------
    vertices : ndarray, shape (N, 3)
    faces    : ndarray of int, shape (M, 3)
    """
    mesh = _load_trimesh_mesh(file_path)
    original_faces = len(mesh.faces)

    if target_faces is None:
        target_faces = max(4, int(original_faces * target_ratio))

    if target_faces >= original_faces:
        logger.info("target_faces >= original faces (%d); returning unchanged.", original_faces)
        return np.array(mesh.vertices), np.array(mesh.faces)

    # --- Primary: trimesh quadric decimation (needs fast_simplification) ---
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        verts = np.array(simplified.vertices, dtype=np.float64)
        faces = np.array(simplified.faces, dtype=np.int64)
    except Exception:
        # --- Fallback: pymeshlab ---
        try:
            import pymeshlab as ml

            ms = ml.MeshSet()
            ms.add_mesh(ml.Mesh(
                vertex_matrix=np.array(mesh.vertices, dtype=np.float32),
                face_matrix=np.array(mesh.faces, dtype=np.int32),
            ))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces))
            result = ms.current_mesh()
            verts = result.vertex_matrix().astype(np.float64)
            faces = result.face_matrix().astype(np.int64)
        except ImportError:
            raise ImportError(
                "decimate_mesh requires either 'fast_simplification' or 'pymeshlab'. "
                "Install with: pip install fast-simplification  OR  pip install pymeshlab"
            )
        except Exception as e:
            raise RuntimeError(f"Decimation failed: {e}") from e

    logger.info(
        "Decimated %d → %d faces (%.1f %%)",
        original_faces,
        len(faces),
        100.0 * len(faces) / original_faces,
    )

    return verts, faces
