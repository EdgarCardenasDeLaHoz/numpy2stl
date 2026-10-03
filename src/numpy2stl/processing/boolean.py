import logging

import numpy as np

logger = logging.getLogger(__name__)


def clean_mesh(ms):
    """Utility to fix common geometric issues before booleans."""
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    return ms


def mesh_volume(vertices, faces):
    """Signed volume of a closed triangle mesh (positive when faces point outward)."""
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def cut_jigsaw(vertices, faces, cutters, engine="manifold", max_loss=0.01):
    """Split a watertight mesh into jigsaw pieces by intersecting it with each cutter.

    Parameters
    ----------
    vertices, faces : ndarray
        The model; must be a closed, consistently wound solid.
    cutters : dict[str, (vertices, faces)]
        Cutter prisms, e.g. from ``applications.puzzle.make_jigsaw_cutters``.
    engine : {"manifold", "pymeshlab"}
        Boolean backend. manifold3d is exact and fast; pymeshlab is the fallback.
    max_loss : float or None
        Largest tolerated fraction of the model's volume that is missing from
        the pieces *besides* the clearance slivers between cutters (e.g. a
        strip of the model no cutter reaches, or cutters that overlap).
        ``None`` skips the check.

    Returns
    -------
    dict[str, (vertices, faces)]
        One entry per non-empty piece, same keys as ``cutters``.

    Raises
    ------
    ValueError
        If the input is not a valid solid for the engine, or the pieces'
        volume differs from the model's minus the slivers by more than
        ``max_loss``.
    """
    if engine == "manifold":
        pieces = _intersect_manifold(vertices, faces, cutters)
    elif engine == "pymeshlab":
        pieces = _intersect_pymeshlab(vertices, faces, cutters)
    else:
        raise ValueError(f"unknown engine {engine!r}; use 'manifold' or 'pymeshlab'")

    if max_loss is not None:
        _check_volume(vertices, faces, cutters, pieces, max_loss)
    return pieces


def _check_volume(vertices, faces, cutters, pieces, max_loss):
    """Pieces must add up to the model minus the gaps between the cutters.

    The gaps are the part of the model inside the cutters' convex hull but in
    no cutter; model outside the hull was never covered and counts as lost.
    """
    import manifold3d as mfd

    base = _to_manifold(vertices, faces, "model")
    union = mfd.Manifold.batch_boolean(
        [_to_manifold(v, f, f"cutter {k}") for k, (v, f) in cutters.items()], mfd.OpType.Add
    )
    v_in = base.volume()
    slivers = (base ^ union.hull()).volume() - (base ^ union).volume()
    v_out = sum(abs(mesh_volume(v, f)) for v, f in pieces.values())
    missing = (v_in - slivers - v_out) / v_in if v_in > 0 else 0.0
    logger.info(
        f"cut_jigsaw: {len(pieces)} pieces, clearance gaps {slivers / max(v_in, 1e-300):.3%}, "
        f"unaccounted {missing:.3%} of the volume"
    )
    if abs(missing) > max_loss:
        raise ValueError(
            f"jigsaw pieces miss {missing:.2%} of the model volume beyond the clearance gaps "
            f"(limit {max_loss:.2%}); do the cutters cover the whole model without overlapping?"
        )


def to_manifold(vertices, faces, what="mesh", strict=True):
    """A ``manifold3d.Manifold`` from an indexed mesh (float64).

    ``strict`` raises ``ValueError`` if the mesh is not a closed manifold solid;
    otherwise the (invalid) Manifold is returned for the caller to check.
    """
    import manifold3d as mfd  # optional extra: numpy2stl[boolean]

    mesh = mfd.Mesh64(
        # Copies: manifold3d rejects read-only arrays (e.g. another Mesh64's).
        vert_properties=np.array(vertices, dtype=np.float64, order="C"),
        tri_verts=np.array(faces, dtype=np.uint64, order="C"),
    )
    solid = mfd.Manifold(mesh)
    if strict and solid.status() != mfd.Error.NoError:
        raise ValueError(f"{what} is not a closed manifold solid: {solid.status()}")
    return solid


def from_manifold(solid):
    """``(vertices float64 (N, 3), faces int64 (M, 3))`` of a Manifold."""
    out = solid.to_mesh64()
    return (np.array(out.vert_properties[:, :3], dtype=np.float64),
            np.array(out.tri_verts, dtype=np.int64))


def union(meshes):
    """Union of the valid closed meshes in ``meshes`` [(vertices, faces), ...].

    Returns ``(Manifold or None, number rejected as not closed/manifold)``.
    """
    import manifold3d as mfd

    ms = [to_manifold(v, f, strict=False) for v, f in meshes]
    good = [m for m in ms if m.status() == mfd.Error.NoError and not m.is_empty()]
    if not good:
        return None, len(ms)
    u = mfd.Manifold.batch_boolean(good, mfd.OpType.Add) if len(good) > 1 else good[0]
    return u, len(ms) - len(good)


_to_manifold = to_manifold


def _intersect_manifold(vertices, faces, cutters):
    base = _to_manifold(vertices, faces, "model")
    pieces = {}
    for key, (v_c, f_c) in cutters.items():
        piece = base ^ _to_manifold(v_c, f_c, f"cutter {key}")  # ``^`` is intersection
        if piece.is_empty():
            logger.warning(f"Piece {key}: empty intersection")
            continue
        pieces[key] = from_manifold(piece)
    return pieces


def _intersect_pymeshlab(vertices, faces, cutters):
    import pymeshlab as ml   # fallback engine; imported only when used

    base_ms = ml.MeshSet()
    base_ms.add_mesh(
        ml.Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    )
    clean_mesh(base_ms)
    cleaned_base = base_ms.current_mesh()

    pieces = {}
    for key, (v_c, f_c) in cutters.items():
        ms = ml.MeshSet()
        ms.add_mesh(cleaned_base, "base")
        ms.add_mesh(
            ml.Mesh(np.asarray(v_c, dtype=np.float64), np.asarray(f_c, dtype=np.int32)), "cutter"
        )
        clean_mesh(ms)
        ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
        res = ms.current_mesh()
        if res.face_number() == 0:
            logger.warning(f"Piece {key}: empty intersection")
            continue
        pieces[key] = (res.vertex_matrix(), res.face_matrix().astype(np.int64))
    return pieces


def cut_puzzle_pieces(model, puzzle):
    """``cut_jigsaw`` with pymeshlab, without the volume check (older API)."""
    vertices, faces = model
    return cut_jigsaw(vertices, faces, puzzle, engine="pymeshlab", max_loss=None)


def cut_puzzle_pieces_manifold(model, puzzle):
    """``cut_jigsaw`` with manifold3d, without the volume check (older API)."""
    vertices, faces = model
    return cut_jigsaw(vertices, faces, puzzle, engine="manifold", max_loss=None)
