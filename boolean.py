import numpy as np
import pymeshlab as ml

def union_pymesh(models):

	ms = ml.MeshSet()
	
	for key in models:
		vx,fs = models[key]
		vx = vx.astype(np.float32)
		fs = fs.astype(np.int32)
		mesh = ml.Mesh(vertex_matrix=vx, face_matrix=fs)
		ms.add_mesh(mesh, key)

	ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)

	result = ms.current_mesh()
	vx, fs = result.vertex_matrix(), result.face_matrix()

	return vx, fs


import pymeshlab as ml
import numpy as np

def clean_mesh(ms):
    """Utility to fix common geometric issues before booleans."""
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_faces()
    return ms

def cut_puzzle_pieces(model, puzzle):
    pieces_out = {}
    vx_base, fs_base = model
    
    # Setup base mesh set for cleaning
    base_ms = ml.MeshSet()
    base_ms.add_mesh(ml.Mesh(vx_base.astype(np.float32), fs_base.astype(np.int32)))
    clean_mesh(base_ms)
    
    # Store the cleaned base mesh to reuse
    cleaned_base = base_ms.current_mesh()

    for key, (vx_p, fs_p) in puzzle.items():
        ms = ml.MeshSet()
        
        # Add cleaned base
        ms.add_mesh(cleaned_base, "base")
        
        # Add and clean the puzzle piece 'cutter'
        ms.add_mesh(ml.Mesh(vx_p.astype(np.float32), fs_p.astype(np.int32)), "cutter")
        clean_mesh(ms) 

        try:
            # Re-index because cleaning might have shifted mesh IDs
            # Mesh 0 is base, Mesh 1 is cutter
            ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
            
            res = ms.current_mesh()
            if res.face_number() > 0:
                pieces_out[key] = (res.vertex_matrix(), res.face_matrix())
                print(f"Piece {key}: Success ({res.face_number()} faces)")
            else:
                print(f"Piece {key}: Empty intersection")
                
        except Exception as e:
            print(f"Piece {key}: Boolean failed - {e}")

    return pieces_out

import manifold3d as mfd
import numpy as np
## faster method using manifold

def cut_puzzle_pieces_manifold(model, puzzle):
    pieces_out = {}
    
    # 1. Unpack and cast the base model
    vx_m, fs_m = model
    vx_m = np.ascontiguousarray(vx_m, dtype=np.float32)
    fs_m = np.ascontiguousarray(fs_m, dtype=np.uint32)
    
    # Initialize Manifold from Mesh
    base_manifold = mfd.Manifold(mfd.Mesh(vert_properties=vx_m, tri_verts=fs_m))
    
    print("Processing pieces with Manifold engine...")

    for key, (vx_p, fs_p) in puzzle.items():
        try:
            # 2. Prep the puzzle piece 'cutter'
            vx_p = np.ascontiguousarray(vx_p, dtype=np.float32)
            fs_p = np.ascontiguousarray(fs_p, dtype=np.uint32)
            
            cutter_manifold = mfd.Manifold(mfd.Mesh(vert_properties=vx_p, tri_verts=fs_p))

            # 3. Intersection Operation
            # In some Manifold versions:
            # ^ is Intersection
            # + is Union
            # - is Difference
            result_manifold = base_manifold ^ cutter_manifold 

            # 4. Convert back
            res_mesh = result_manifold.to_mesh()
            
            if len(res_mesh.tri_verts) > 0:
                # Rebuild the vertex matrix to (N, 3)
                verts = res_mesh.vert_properties.reshape(-1, 3)
                pieces_out[key] = (verts, res_mesh.tri_verts)
                print(f"Piece {key}: Success")
            else:
                print(f"Piece {key}: No intersection")

        except Exception as e:
            # If ^ fails, let's try the intersect method directly
            try:
                result_manifold = base_manifold.intersect(cutter_manifold)
                res_mesh = result_manifold.to_mesh()
                # ... repeat conversion ...
                verts = res_mesh.vert_properties.reshape(-1, 3)
                pieces_out[key] = (verts, res_mesh.tri_verts)
                print(f"Piece {key}: Success (via .intersect)")
            except:
                print(f"Piece {key}: Failed - {e}")

    return pieces_out