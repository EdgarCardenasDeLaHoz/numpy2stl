

import numpy as np
import scipy
import triangle as tr
import matplotlib.pyplot as plt

from numpy2stl.numpy2stl import get_open_edges, get_ordered_perimeter, simplify_surface, get_surfaces

import numpy as np
import shapely
from shapely import Polygon, orient_polygons
from shapely.geometry import Polygon
from shapely import constrained_delaunay_triangles



def simplify_mesh_surfaces(vertices, faces, min_faces=10):

		surfaces = extract_surfaces(vertices, faces, min_faces)

		surfaces = filter_collinear_perimeters(surfaces, vertices)

		faces_out = []
		for s in surfaces:
				if s["type"] == "complex":
						faces_out.append(remesh_surface(s))
				else:
						faces_out.append(s["faces"])

		return np.vstack(faces_out)
		

from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon

# -----------------------------
# SURFACE EXTRACTION
# -----------------------------
def extract_surfaces(vertices, faces, min_faces=10):
    surfaces_idx, _ = get_surfaces(vertices, faces, normals=None)
    surfaces = []
    sid = 0

    for sub_faces in surfaces_idx:
        face_list = faces[sub_faces]

        if len(sub_faces) < min_faces:
            surfaces.append({
                "id": sid,
                "faces": face_list,
                "type": "minor"
            })
            sid += 1
            continue

        pr, pts_idx, local_faces = project_vertices(face_list, vertices)
        perimeters_local, removed_faces_local = get_surface_perimeter(local_faces, pr)

        # 🔑 MAP LOCAL → GLOBAL
        # 1. Map perimeters: local array indices -> global vertex IDs
        perimeters_global = [pts_idx[p] for p in perimeters_local]
        
        # 2. Map removed faces: local face indices -> global vertex IDs
        removed_faces_global = pts_idx[removed_faces_local] if len(removed_faces_local) > 0 else []

        # --- Main surface
        surfaces.append({
            "id": sid,
            "faces": face_list,
            "local_faces": local_faces,
            "pts_idx": pts_idx, # Kept for remeshing lookup
            "pr": pr,           # Kept for 2D geometric checks
            "perimeters": perimeters_global, 
            "type": "complex"
        })
        sid += 1

        # --- Removed faces as their own surface
        if len(removed_faces_global) > 0:
            surfaces.append({
                "id": sid,
                "faces": removed_faces_global, 
                "type": "removed"
            })
            sid += 1

    return surfaces

import math

def filter_collinear_perimeters(surfaces, vertices, tolerance=1e-12):
	must_keep = set()
	v_usage = defaultdict(list)

	# PASS 1: Identify "Protected" Vertices (Global IDs)
	for s_idx, s in enumerate(surfaces):
		if s["type"] in ["minor", "removed"]:
			# Protect every vertex used by non-complex surfaces
			must_keep.update(s["faces"].ravel())
		else:
			# Track which complex surfaces use which boundary vertices
			for p_idx, p_nodes in enumerate(s["perimeters"]):
				for i, v_id in enumerate(p_nodes):
					v_usage[v_id].append((s_idx, p_idx, i))

	surface_mappings = {}
	for s_idx, s in enumerate(surfaces):
		if s["type"] == "complex":
			surface_mappings[s_idx] = {gid: idx for idx, gid in enumerate(s["pts_idx"])}

	# PASS 2: Identify Sharp Corners (Geometry check in 2D)
	for v_id, usages in v_usage.items():
		if v_id in must_keep: continue
		
		for s_idx, p_idx, i in usages:
			s = surfaces[s_idx]
			p_nodes = s["perimeters"][p_idx]
			n = len(p_nodes)
			
			# Get Local Indices for the triplet (prev, curr, next)
			global_to_local = surface_mappings[s_idx]
			try:
				ids = [global_to_local[p_nodes[(i-1)%n]], 
					global_to_local[v_id], 
					global_to_local[p_nodes[(i+1)%n]]]
			except KeyError:
				must_keep.add(v_id); break

			# Collinearity check using 2D projected coordinates
			pts = s["pr"][ids]
			v1 = pts[1] - pts[0]
			v2 = pts[2] - pts[1]
			
			cross = v1[0] * v2[1] - v1[1] * v2[0]
			#norm = np.linalg.norm(v1) * np.linalg.norm(v2)
			norm = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
			if (abs(cross) / norm) > tolerance:
				must_keep.add(v_id)
				break

	# PASS 3: Iterative Removal
	for s in surfaces:
		if s["type"] != "complex": continue
		
		global_to_local = {gid: idx for idx, gid in enumerate(s["pts_idx"])}
		updated_perimeters = []
		
		for p_nodes in s["perimeters"]:
			p_list = list(p_nodes)
			i = 0
			while i < len(p_list) and len(p_list) > 3:
				v_id = p_list[i]
				if v_id in must_keep:
					i += 1
					continue
				
				# Test if removing this point breaks the polygon
				test_p = p_list[:i] + p_list[i+1:]
				p_list = test_p
			updated_perimeters.append(np.array(p_list))
		
		s["perimeters"] = updated_perimeters
		
	return surfaces

def remesh_surface(surface):
		
		def fast_keys(arr):
				return np.round(arr[:, 0], 6) * 1e7 + np.round(arr[:, 1], 6)
			
		def rematch_face(all_coords, pr, pts_idx):
				flat_tri = all_coords.reshape(-1, 4, 2)[:, :3, :].reshape(-1, 2)

				# Create 1D keys for matching (Rounding to 6 decimals to handle jitter)
				source_keys = fast_keys(pr)
				target_keys = fast_keys(flat_tri)

				# Match target_keys to source_keys using searchsorted
				sort_idx = np.argsort(source_keys)
				matched_indices = np.searchsorted(source_keys, target_keys, sorter=sort_idx)
				local_indices = sort_idx[matched_indices]

				# Map Local -> Global index
				final_faces = pts_idx[local_indices].reshape(-1, 3)
				return final_faces

		pr = surface["pr"]
		pts_idx = surface["pts_idx"]
		perimeters_global = surface["perimeters"]

		# 1. Map Global IDs -> Local indices for 2D math
		global_to_local = {gid: i for i, gid in enumerate(pts_idx)}
		
		shell_coords = pr[[global_to_local[gid] for gid in perimeters_global[0]]]
		holes_coords = [pr[[global_to_local[gid] for gid in p]] for p in perimeters_global[1:]]

		# 2. Create Polygon and Triangulate
		poly = Polygon(shell_coords, holes=holes_coords)
		poly = orient_polygons(poly)

		tri_collection = constrained_delaunay_triangles(poly)

		all_coords = shapely.get_coordinates(tri_collection)

		final_faces = rematch_face(all_coords, pr, pts_idx)

		return final_faces



def project_vertices(group_faces, vertices):
		
		# 1. IDENTIFY UNIQUE VERTICES (The "haystack")
		# pts_idx maps: local_index -> global_vx_index
		pts_idx = np.unique(group_faces.ravel())
		local_vertices = vertices[pts_idx]

		# 2. OPTIMIZED PROJECTION
		max_val = group_faces.max() + 1
		lut = np.full(max_val, -1, dtype=np.int32)
		lut[pts_idx] = np.arange(len(pts_idx))
		local_faces = lut[group_faces]

		# Project to 2D
		projected, _ = project_to_plane_with_face(local_vertices, local_faces[0])

		return projected, pts_idx, local_faces


def get_surface_perimeter(faces, projected):

		#projected = project_vertices(group_faces, vertices)
		removed_faces = []

		open_edges = get_open_edges(faces)

		# 2. Find junctions (vertices with > 2 open edges)
		counts = np.bincount(open_edges.ravel())
		junctions = np.where(counts > 2)[0]

		if junctions.size > 0:
				# 3. Identify and remove faces touching junctions
				bad_map = np.zeros(faces.max() + 1, dtype=bool)
				bad_map[junctions] = True
				mask = bad_map[faces].any(axis=1)
				
				removed_faces = faces[mask]
				faces = faces[~mask]
				
				# Re-calculate open edges for the "cleaned" mesh
				open_edges = get_open_edges(faces)

		# 4. Chain them (the O(N) successor map method)
		#perimeters = chain_open_edges(open_edges)

		perimeters = get_ordered_perimeter(projected, open_edges)
	
		return perimeters, removed_faces






def vertices_to_index(triangles):
		"""
		"""
		vertices_list = np.reshape(triangles, (triangles.shape[0]*3,3))
		uni_vertices, vertices_idx = np.unique(vertices_list, axis=0 , return_inverse=True)
		vertices_idx = np.reshape(vertices_idx, (triangles.shape[0],3))
		
		return uni_vertices, vertices_idx

def triangulate_polygon(vertices_2D, perimeters):

		grouped_edges = perimeters_to_edges(perimeters)

		holes = []
		if len(grouped_edges)>1:
				for i in range(1,len(grouped_edges)):

					verts, faces = triangulate_edges(vertices_2D, grouped_edges[i])
					tris = verts[faces]
					cents = tris.mean(axis=1)
					holes.append(cents[0])
		else:
				holes = None

		all_edges = np.concatenate(grouped_edges)
		vertices, faces = triangulate_edges(vertices_2D, all_edges, holes=holes)

		return vertices, faces

def triangulate_edges(vertices_2D, edges, holes=None):

		if vertices_2D.shape[1]==3:
			vertices_2D = vertices_2D[:,:2]

			vertices_2D


		if holes is None:
			shape = {"vertices": vertices_2D, "segments": edges}   
		else:
			shape = {"vertices": vertices_2D, "segments": edges, "holes": holes}
		t = tr.triangulate(shape,'p')  
		vertices = t["vertices"]
		faces =  t['triangles'] 

		sub_faces = np.unique(faces.reshape(1,-1),return_inverse=True)[1].reshape(-1,3)

		vertices = vertices[edges[:,0]]
		faces = np.argsort(edges[:,0])[sub_faces]

		return vertices,faces

def project_to_plane_with_face(vertices, ref_face):
		"""
		Project 3D vertices to 2D plane using the plane defined by a single triangle (ref_face)
		"""
		# Reference triangle vertices
		v0, v1, v2 = vertices[ref_face]

		# Compute normal of the triangle
		normal = np.cross(v1 - v0, v2 - v0)
		normal /= np.linalg.norm(normal)

		# Define plane axes
		axis_x = v1 - v0
		axis_x /= np.linalg.norm(axis_x)
		axis_y = np.cross(normal, axis_x)

		# Project all vertices onto the plane axes
		projected = np.zeros((vertices.shape[0], 2))
		projected[:, 0] = np.dot(vertices - v0, axis_x)
		projected[:, 1] = np.dot(vertices - v0, axis_y)

		return projected, normal

def perimeters_to_edges(perimeters):
		edges = [np.stack([ p, np.roll(p,1,axis=0) ],axis=1) for p in  perimeters ]
		return edges 

def plot_surface(projected, group_faces, open_edges, pts_idx, perimeters, simp_faces ):

		if len(group_faces)<1000:

			plt.scatter(projected[pts_idx, 0], projected[pts_idx, 1], s=5) #
			for f in group_faces: 
				f = np.append(f, f[0]) 
				plt.plot(projected[f, 0], projected[f, 1], 'k-', alpha=0.3) 


		if len(group_faces)<100:
			for n,(idx) in enumerate(pts_idx):
				plt.text(projected[idx, 0], projected[idx, 1], n) #


		for e in open_edges: 
			plt.plot(projected[e, 0], projected[e, 1], 'r-', lw=5) 

		for p in perimeters: 
			plt.plot(projected[p, 0], projected[p, 1], 'g-', lw=4) 

		for f in simp_faces: 
			f = np.append(f, f[0])
			plt.plot(projected[f, 0], projected[f, 1], 'b-', lw=2) 
			
		plt.axis("equal")
		plt.title("Projected surface + simplified mesh") 
		return

def simplify_surface(vertices, perimeters, normal=None):
		"""
		"""    
		## Flatten boundry to 2D
		if normal is None:  normal = np.array([0,0,1])

		sub_verts = vertices[np.concatenate(perimeters)]
		sub_peri = []
		end = 0 
		for p in perimeters:
			sub_peri.append(np.arange(len(p))+end)
			end += len(p)

		_,sub_faces = triangulate_polygon( sub_verts , sub_peri)
		faces = np.concatenate(perimeters)[sub_faces]

		return sub_verts, faces

def triangle_area_3d(p1, p2, p3):
		"""
		Calculates the area of a triangle in 3D space given three points.

		Args:
			p1, p2, p3: numpy arrays or lists representing the 3D coordinates 
						of the three vertices (e.g., [x, y, z]).

		Returns:
			The area of the triangle.
		"""
		# Create two vectors that form two sides of the triangle, e.g., p1p2 and p1p3
		vector1 = np.array(p2) - np.array(p1)
		vector2 = np.array(p3) - np.array(p1)
		
		# Calculate the cross product of the two vectors
		cross_product = np.cross(vector1, vector2)
		
		# The magnitude of the cross product is the area of the parallelogram formed 
		# by the vectors. The triangle's area is half of this magnitude.
		area = 0.5 * np.linalg.norm(cross_product)
		
		return area

def calculate_areas_of_triangles_list(triangles_list):
			
		areas = np.sum([triangle_area_3d(p1, p2, p3) for p1, p2, p3 in triangles_list])
		return areas

