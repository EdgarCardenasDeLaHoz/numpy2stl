import trimesh
import numpy as np

import numpy as np
from collections import Counter

import trimesh
import numpy as np
from collections import Counter

def check_model_status(models, use_trimesh=True, repair=True):
    """
    Analyzes and optionally repairs 3D mesh models.
    """
    repaired_models = {}

    for key, (vertices, faces) in models.items():
        print(f"\n{'='*30}")
        print(f"ID: {key}")
        
        # 1. Basic Stats
        centroid = vertices.mean(axis=0)
        dims = vertices.ptp(axis=0)
        print(f"Centroid (XYZ):  {np.round(centroid, 2)}")
        print(f"Dimensions:      {np.round(dims, 2)}")

        # 2. Geometry Analysis (Vectorized)
        naked, non_manifold = find_mesh_issues(faces)
        conflicts = check_orientation(faces)

        print(f"Holes (Naked Edges): {len(naked)}")
        print(f"Non-Manifold Edges:  {len(non_manifold)}")
        print(f"Normal Conflicts:    {len(conflicts)} (Winding issues)")

        if use_trimesh:
            # Initialize Mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            
            # Diagnostic Report
            diagnose_mesh(mesh)

            if repair:
                # The 'Healing' Pipeline
                mesh.remove_duplicate_faces()
                mesh.remove_infinite_values()
                mesh.remove_unreferenced_vertices()
                mesh.merge_vertices(merge_tex=True, merge_norm=True)
                mesh.fix_normals()  # Fixes the 'Inside-Out' issue
                mesh.fill_holes()   # Attempts to make it watertight
            
            # Store repaired data
            repaired_models[key] = (mesh.vertices, mesh.faces)
            
    print(f"{'='*30}\n")
    return repaired_models

def find_mesh_issues(faces):
    """Vectorized check for holes and over-shared edges."""
    edges = np.sort(np.vstack([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
    # View as structured array for unique counting
    edge_view = np.ascontiguousarray(edges).view(np.dtype((np.void, edges.dtype.itemsize * 2)))
    _, counts = np.unique(edge_view, return_counts=True)
    
    naked = np.sum(counts == 1)
    non_manifold = np.sum(counts > 2)
    return ["edge"] * naked, ["edge"] * non_manifold

def check_orientation(faces):
    """Vectorized check for inconsistent winding (Normal conflicts)."""
    edges = np.vstack([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]])
    # Map directed edges to a single hashable value
    max_v = faces.max() + 1
    edge_hashes = edges[:, 0] * max_v + edges[:, 1]
    
    unique_hashes, counts = np.unique(edge_hashes, return_counts=True)
    # Conflicts: edges that appear more than once in the SAME direction
    conflicts = unique_hashes[counts > 1]
    return conflicts

def diagnose_mesh(mesh):
    """Detailed Trimesh-based manifold health check."""
    status = []
    if mesh.is_watertight: status.append("✅ Watertight")
    else: status.append("❌ Holes Detected")
    
    if mesh.is_volume: status.append("✅ Valid Volume")
    else: status.append("❌ Zero/Invalid Volume")
    
    if not mesh.is_winding_consistent: status.append("⚠️ Inconsistent Winding")
    
    print(f"Status:          {' | '.join(status)}")
    print(f"Volume:          {mesh.volume:.4f}")
    
    # Check for degenerate (zero-area) faces
    degenerate_count = np.sum(mesh.area_faces < 1e-7)
    if degenerate_count > 0:
        print(f"Degenerate:      {degenerate_count} zero-area faces found")
    
    # Dimensionality check
    if np.any(mesh.extents < 1e-4):
        print("Warning:         Mesh is effectively 2D/Flat.")
