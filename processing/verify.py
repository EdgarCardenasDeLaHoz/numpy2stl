import logging

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


def check_model_status(models, use_trimesh=True, repair=True):
    """
    Analyzes and optionally repairs 3D mesh models.
    """
    repaired_models = {}

    for key, (vertices, faces) in models.items():
        logger.info(f"{'='*30}")
        logger.info(f"Model ID: {key}")

        centroid = vertices.mean(axis=0)
        dims = vertices.ptp(axis=0)
        logger.info(f"Centroid (XYZ):  {np.round(centroid, 2)}")
        logger.info(f"Dimensions:      {np.round(dims, 2)}")

        naked, non_manifold = find_mesh_issues(faces)
        conflicts = check_orientation(faces)

        logger.info(f"Holes (Naked Edges): {len(naked)}")
        logger.info(f"Non-Manifold Edges:  {len(non_manifold)}")
        logger.info(f"Normal Conflicts:    {len(conflicts)} (Winding issues)")

        if use_trimesh:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

            diagnose_mesh(mesh)

            if repair:
                mesh.remove_duplicate_faces()
                mesh.remove_infinite_values()
                mesh.remove_unreferenced_vertices()
                mesh.merge_vertices(merge_tex=True, merge_norm=True)
                mesh.fix_normals()
                mesh.fill_holes()

            repaired_models[key] = (mesh.vertices, mesh.faces)

    print(f"{'='*30}\n")
    return repaired_models


def find_mesh_issues(faces):
    """Vectorized check for holes and over-shared edges."""
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edge_view = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * 2)))
    _, counts = np.unique(edge_view, return_counts=True)

    naked = np.sum(counts == 1)
    non_manifold = np.sum(counts > 2)
    return ["edge"] * naked, ["edge"] * non_manifold


def check_orientation(faces):
    """Vectorized check for inconsistent winding (Normal conflicts)."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    max_v = faces.max() + 1
    edge_hashes = edges[:, 0] * max_v + edges[:, 1]

    unique_hashes, counts = np.unique(edge_hashes, return_counts=True)
    conflicts = unique_hashes[counts > 1]
    return conflicts


def diagnose_mesh(mesh):
    """Detailed Trimesh-based manifold health check."""
    status = []
    if mesh.is_watertight:
        status.append("Watertight")
    else:
        status.append("Holes Detected")

    if mesh.is_volume:
        status.append("Valid Volume")
    else:
        status.append("Zero/Invalid Volume")

    if not mesh.is_winding_consistent:
        status.append("Inconsistent Winding")

    logger.info(f"Status:          {' | '.join(status)}")
    logger.info(f"Volume:          {mesh.volume:.4f}")

    degenerate_count = np.sum(mesh.area_faces < 1e-7)
    if degenerate_count > 0:
        logger.warning(f"Degenerate:      {degenerate_count} zero-area faces found")

    if np.any(mesh.extents < 1e-4):
        logger.warning("Warning:         Mesh is effectively 2D/Flat.")
