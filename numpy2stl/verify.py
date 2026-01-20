import trimesh
import numpy as np
from .simplify import  get_open_edges

def check_model_status(models):

    # Assuming get_open_edges is defined elsewhere in your script
    for key, (vertices, faces) in models.items():
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        # --- Repair Pipeline ---
        mesh.fix_normals()
        mesh.fill_holes()            # Attempts to close gaps
        mesh.merge_vertices()        # Merges vertices within a small epsilon
        mesh.remove_duplicate_faces()

        # --- Data Collection ---
        centroid = mesh.vertices.mean(axis=0)
        dims = mesh.vertices.ptp(axis=0)
        open_edges = get_open_edges(mesh.faces)
        # Find faces where vertex indices are repeated (degenerate)
        degenerate_faces = [i for i, face in enumerate(mesh.faces) if len(set(face)) < len(face)]

        # --- Improved Print Statements ---
        print(f"{'='*30}")
        print(f"ID: {key}")

        print(f"Centroid (XYZ):  {np.round(centroid, 3)}")
        print(f"Dimensions:      {np.round(dims, 3)}")
        print(f"Watertight:      {'✅ Yes' if mesh.is_watertight else '❌ No'}")
        print(f"Valid Volume:    {'✅ Yes' if mesh.is_volume else '❌ No'}")
        
        if not mesh.is_watertight:
          print(f"Open Edges:      {len(open_edges)} (Holes detected)")
        
        if degenerate_faces:
          print(f"Degenerate:      {len(degenerate_faces)} faces found")
          
        print(f"{'-'*30}")


def diagnose_mesh(mesh):
    print(f"Watertight: {mesh.is_watertight}")
    print(f"Volume:     {mesh.volume:.6f}")
    
    # 1. Check for Zero-Area Triangles (Degenerate Faces)
    degenerate = mesh.area_faces < 1e-8
    if np.any(degenerate):
        print(f"❌ Found {np.sum(degenerate)} degenerate (zero-area) triangles.")

    # 2. Check Face Orientations (Normals)
    # If normals point in conflicting directions, volume can sum to zero.
    if not mesh.is_winding_consistent:
        print("❌ Winding order is inconsistent (some faces are inside-out).")
    
    # 3. Check for Self-Intersections
    # A "watertight" mesh that intersects itself often has zero or negative volume.
    if mesh.is_self_intersecting:
        print("❌ Mesh is self-intersecting.")

    # 4. Dimensionality Check (Is it flat?)
    extents = mesh.extents
    print(f"Bounding Box Dimensions: {extents}")
    if np.any(extents < 1e-5):
        print("❌ Mesh is effectively 2D (one dimension is near zero).")

    # 5. Visual Highlight (Optional)
    # This colorizes faces that might be causing issues
    if np.any(degenerate):
        mesh.visual.face_colors[degenerate] = [255, 0, 0, 255] # Red for degenerate
        print("👉 Degenerate faces have been colored Red.")