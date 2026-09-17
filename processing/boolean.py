import logging

import numpy as np
import pymeshlab as ml

logger = logging.getLogger(__name__)


def union_pymesh(models):

    ms = ml.MeshSet()

    for key in models:
        vx, fs = models[key]
        vx = vx.astype(np.float32)
        fs = fs.astype(np.int32)
        mesh = ml.Mesh(vertex_matrix=vx, face_matrix=fs)
        ms.add_mesh(mesh, key)

    ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)

    result = ms.current_mesh()
    vx, fs = result.vertex_matrix(), result.face_matrix()

    return vx, fs


def clean_mesh(ms):
    """Utility to fix common geometric issues before booleans."""
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    return ms


def cut_puzzle_pieces(model, puzzle):
    pieces_out = {}
    vx_base, fs_base = model

    base_ms = ml.MeshSet()
    base_ms.add_mesh(ml.Mesh(vx_base.astype(np.float32), fs_base.astype(np.int32)))
    clean_mesh(base_ms)

    cleaned_base = base_ms.current_mesh()

    for key, (vx_p, fs_p) in puzzle.items():
        ms = ml.MeshSet()

        ms.add_mesh(cleaned_base, "base")

        ms.add_mesh(ml.Mesh(vx_p.astype(np.float32), fs_p.astype(np.int32)), "cutter")
        clean_mesh(ms)

        try:
            ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)

            res = ms.current_mesh()
            if res.face_number() > 0:
                pieces_out[key] = (res.vertex_matrix(), res.face_matrix())
                logger.info(f"Piece {key}: Success ({res.face_number()} faces)")
            else:
                logger.warning(f"Piece {key}: Empty intersection")

        except Exception as e:
            logger.error(f"Piece {key}: Boolean failed - {e}")

    return pieces_out


import manifold3d as mfd


def cut_puzzle_pieces_manifold(model, puzzle):
    pieces_out = {}

    vx_m, fs_m = model
    vx_m = np.ascontiguousarray(vx_m, dtype=np.float32)
    fs_m = np.ascontiguousarray(fs_m, dtype=np.uint32)

    base_manifold = mfd.Manifold(mfd.Mesh(vert_properties=vx_m, tri_verts=fs_m))

    print("Processing pieces with Manifold engine...")

    for key, (vx_p, fs_p) in puzzle.items():
        try:
            vx_p = np.ascontiguousarray(vx_p, dtype=np.float32)
            fs_p = np.ascontiguousarray(fs_p, dtype=np.uint32)

            cutter_manifold = mfd.Manifold(mfd.Mesh(vert_properties=vx_p, tri_verts=fs_p))

            result_manifold = base_manifold ^ cutter_manifold

            res_mesh = result_manifold.to_mesh()

            if len(res_mesh.tri_verts) > 0:
                verts = res_mesh.vert_properties.reshape(-1, 3)
                pieces_out[key] = (verts, res_mesh.tri_verts)
                logger.info(f"Piece {key}: Success")
            else:
                logger.warning(f"Piece {key}: No intersection")

        except Exception as e:
            try:
                result_manifold = base_manifold.intersect(cutter_manifold)
                res_mesh = result_manifold.to_mesh()
                verts = res_mesh.vert_properties.reshape(-1, 3)
                pieces_out[key] = (verts, res_mesh.tri_verts)
                logger.info(f"Piece {key}: Success (via .intersect)")
            except Exception:
                logger.error(f"Piece {key}: Failed - {e}")

    return pieces_out
