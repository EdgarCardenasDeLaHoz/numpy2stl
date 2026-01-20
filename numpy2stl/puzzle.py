from city2stl.create import triangulate_prism
from numpy2stl.numpy2stl import vertices_to_index
from numpy2stl.numpy2stl.generate import perimeter_to_walls
from numpy2stl.numpy2stl.simplify import  get_open_edges


import numpy as np
from shapely.geometry import Polygon, MultiLineString
from shapely.ops import triangulate, unary_union, polygonize

import trimesh

########################################
def make_puzzle_model(width, b,m, base_n, a=0, z=100):

	pts_list = make_puzzle_pts(width, b,m,base_n,a=a)

	models = {}
	for n,pts in enumerate(pts_list):
		models[str(n)] = make_prism_solid(pts.copy(), z0=-base_n,z1=z)

	###
	for key in models:
		vertices, faces = models[key]	
		mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
		mesh.fix_normals()
		models[key]	= mesh.vertices, mesh.faces

	return models

def make_prism_solid(pts, z0=0,z1=5):

	vert = pts.copy()
	zdim = np.zeros((len(vert),1)) + z1
	vert = np.concatenate([vert, zdim],axis=1)        

	perimeters = [np.arange(len(vert))]
	edges = np.stack([ perimeters, np.roll(perimeters,1,axis=0) ],axis=1) 	
	vert2, faces = robust_triangulate(vert, edges[0], holes=None)

	#########
	top_triangles = vert2[faces]

	#########
	bottom_vertices = vert2.copy()
	bottom_vertices[:,2] = z0
	bottom_triangles = bottom_vertices[faces[:,[1,0,2]]]

	#########
	wall_triangles = prism_wall_vertices(vert2, faces, floor_val=z0)
	
	#########
	all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])

	vx, fs = vertices_to_index(all_triangles)
	fs = np.array([f for f in fs if len(set(f)) == len(f)])
	vx[:,[1,0]] = vx[:,[0,1]]

	return vx,fs

def prism_wall_vertices(vertices, faces, floor_val=0):

	open_edges = get_open_edges(faces)

	wall_vertices = []

	for edge in open_edges:
		
		a = vertices[edge[0]]
		b = vertices[edge[1]]

		top_left = np.concatenate([  a[:2], [floor_val] ])
		top_right = np.concatenate([  b[:2], [floor_val] ])

		bottom_left = np.array(  a  )
		bottom_right = np.array(  b )

		vert = [top_right, top_left, bottom_right]
		wall_vertices.append(vert)

		vert = [bottom_right, top_left, bottom_left]
		wall_vertices.append(vert)
	
	return wall_vertices


def robust_triangulate(vertices, edges, holes=None):
    # 1. Geometry Cleaning
    lines = [vertices[edge] for edge in edges]
    merged_lines = unary_union(MultiLineString(lines))
    
    # Form the boundary polygon(s)
    polygons = list(polygonize(merged_lines))
    if not polygons:
        raise ValueError("Could not form a closed polygon from edges.")
    boundary_poly = unary_union(polygons)
    
    # 2. Prep Holes (if any)
    holes = [Polygon(h) for h in (holes or [])]
    
    # 3. Generate Delaunay and FILTER
    raw_triangles = triangulate(boundary_poly)
    keep_coords = []
    
    for tri in raw_triangles:
        centroid = tri.centroid
        
        # KEY ADDITION: Check if triangle is valid
        # Must be inside the perimeter AND NOT inside any hole
        is_inside = boundary_poly.contains(centroid)
        is_in_hole = any(h.contains(centroid) for h in holes)
        
        if is_inside and not is_in_hole:
            keep_coords.append(np.array(tri.exterior.coords)[:3])
    
    if not keep_coords:
        return np.array([]), np.array([])

    # 4. Re-indexing
    flat_coords = np.vstack(keep_coords)
    unique_verts, inverse_indices = np.unique(
        np.round(flat_coords, 5), 
        axis=0, 
        return_inverse=True
    )
    faces = inverse_indices.reshape(-1, 3)
    
    return unique_verts, faces

def make_puzzle_pts(width, b, m, base_n, a=0):
    # Handle width as scalar or list

    width_x, width_y = width if isinstance(width, (list, tuple)) else (width, width)

    # Create the grid ranges
    grid_range_x = list(np.arange(0, width_x, b)) 
    grid_range_y = list(np.arange(0, width_y, b)) 

    bx = min(b, width_x/len(grid_range_x))
    by = min(b, width_y/len(grid_range_y))
    b = (bx,by)

    pts_list = []
    for ni, i in enumerate(grid_range_x):
      for nj, j in enumerate(grid_range_y):
        # Pass the actual lengths of the grid to check for boundaries
        temp = make_puzzle_piece(b, m, ni, nj, base_n, len(grid_range_x), len(grid_range_y))
        
        # Apply the global offset
        temp[:, 0] += (ni * bx)
        temp[:, 1] += (nj * by)
        pts_list.append(temp)

    for n,pts in enumerate(pts_list):
      pts[pts==0] = -base_n
      pts[pts[:,0]>(width_x-base_n),0] = width_x + base_n
      pts[pts[:,1]>(width_y-base_n),1] = width_y + base_n
      pts_list[n] = pts	

    return pts_list

def make_puzzle_piece(b, m, ni, nj, base_n, len_x, len_y, a=0):

    def get_notched_square(b, m, n_left, n_bottom, n_right, n_top, a=0):

      # Handle width as scalar or list
      b_x, b_y = b if isinstance(b, (list, tuple)) else (b, b)

      x = np.array([
          # Left Side
          [a, b_y], 
          [a, b_y-m], [a+n_left, b_y-m], [a+n_left, m], [a, m], 
          # Bottom Side
          [a, a], 
          [m, a], [m, a+n_bottom], [b_x-m, a+n_bottom], [b_x-m, a], 
          # Right Side
          [b_x, a], 
          [b_x, m], [b_x-n_right, m], [b_x-n_right, b_y-m], [b_x, b_y-m], 
          # Top Side
          [b_x, b_y], 
          [b_x-m, b_y], [b_x-m, b_y-n_top], [m, b_y-n_top], [m, b_y], 
          #Return to Start
          [a, b_y] 
        ])
      return x
    
    # The opposite side gets the inverse to match the next neighbor    
    n_l = (1 if (nj) % 2 == 0 else -1) * base_n
    n_b = (1 if (ni) % 2 == 0 else -1) * base_n
    n_r = -n_l
    n_t = -n_b

    # Border cleanup: If it's an exterior edge, set notch depth to 0 (flat)
    if ni == 0: n_l = 0
    if ni == len_x - 1: n_r = 0
    if nj == 0: n_b = 0
    if nj == len_y - 1: n_t = 0

    return get_notched_square(b, m, n_l, n_b, n_r, n_t, a)
