import numpy as np
import trimesh
from shapely import constrained_delaunay_triangles

from shapely.ops import triangulate, unary_union, polygonize, polygonize, orient
from shapely.geometry import MultiLineString, Polygon, MultiPolygon

from city2stl.create import triangulate_prism
from .solid import vertices_to_index
from .generate import perimeter_to_walls
from .simplify import get_open_edges
from . import solid


def make_base_border(width, b, m, base_n, height=1, offset_dist=5):

	pts_list = make_puzzle_pts(width, b, m, base_n, a=0, border_buffer=0)

	base_border = {}
	for n, pts in enumerate(pts_list):
		vs, fs = make_hollow_prism_solid(pts, offset_dist=offset_dist, z1=height , z0=0)

		base_border["Base"+ str(n)] = vs,fs

	return base_border

def make_hollow_prism_solid(pts, offset_dist=1, z0=0, z1=5):
	# 1. Get 2D Geometry
	outer_pts, edges, holes = make_hollow_cap(pts, offset_dist=offset_dist)
	
	# 2. Triangulate (The source of truth for all vertices)
	vx_2d, faces = robust_triangulate(outer_pts, edges, holes=holes)
	num_v = len(vx_2d)

	# 3. Create 3D Vertices (Stacked: Bottom then Top)
	v_bottom = np.column_stack([vx_2d, np.full(num_v, z0)])
	v_top = np.column_stack([vx_2d, np.full(num_v, z1)])

	top_triangles = v_top[faces]
	bottom_triangles = v_bottom[faces[:,[1,0,2]]]
	wall_triangles = prism_wall_vertices(v_top, faces, floor_val=z0)
	all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])

	vx, fs = solid.vertices_to_index(all_triangles)
	fs = np.array([f for f in fs if len(set(f)) == len(f)])
	vx[:,[1,0]] = vx[:,[0,1]]

	if 0:

		mesh = trimesh.Trimesh(vertices=vx, faces=fs, process=True)
		# Final cleanup for watertight volume
		mesh.merge_vertices()
		mesh.fix_normals()
		# Coordinate swap for your specific system
		vx, fs = mesh.vertices.copy(), mesh.faces.copy()
	
	return vx, fs

def make_hollow_cap(pts, offset_dist=2):
	# 1. Create the geometries
	outer_poly = Polygon(pts)
	# The internal offset (negative buffer)
	inner_poly = outer_poly.buffer(-offset_dist, join_style=2)
	
	# 2. Extract vertices and edges for the outer boundary
	outer_coords = np.array(outer_poly.exterior.coords)[:-1]
	# Create the edge list [ [0,1], [1,2]... ]
	outer_indices = np.arange(len(outer_coords))
	edges = np.stack([outer_indices, np.roll(outer_indices, -1)], axis=1)
	
	# 3. Extract hole coordinates (as a list of arrays)
	# Handle case where offset might split polygon into MultiPolygon
	if inner_poly.is_empty:
		holes = None
	elif isinstance(inner_poly, MultiPolygon):
		holes = [np.array(p.exterior.coords) for p in inner_poly.geoms]
	else:
		holes = [np.array(inner_poly.exterior.coords)]
		
	return outer_coords, edges, holes

def prism_wall_vertices_optimized(pts, z0, z1, is_internal=False):
	walls = []
	n = len(pts)
	for i in range(n):
		p1 = pts[i]
		p2 = pts[(i + 1) % n]
		
		# 4 corners of the wall segment
		b1 = [p1[0], p1[1], z0]
		b2 = [p2[0], p2[1], z0]
		t1 = [p1[0], p1[1], z1]
		t2 = [p2[0], p2[1], z1]
		
		# Two triangles per rectangular segment
		# If it's an internal hole, we flip the winding so normals face "in"
		if is_internal:
			walls.append([t1, b1, b2])
			walls.append([t1, b2, t2])
		else:
			walls.append([t1, b2, b1])
			walls.append([t2, b2, t1])
			
	return np.array(walls)



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

def extrude_solid_polygon(pts, z0=0, z1=5):
    """
    Creates a watertight 3D manifold from 2D points using Trimesh's 
    built-in extrusion which is more robust than manual wall building.
    """
    # 1. Create a 2D profile using Shapely
    poly = Polygon(pts)
    
    # 2. Extrude the polygon 
    # This automatically handles triangulation and side-wall creation
    height = z1 - z0
    mesh = trimesh.creation.extrude(poly, height)
    
    # 3. Translate to the correct z-offset
    mesh.apply_translation([0, 0, z0])
    
    # 4. Clean up (Optional but recommended)
    mesh.merge_vertices()
    mesh.fix_normals()
    
    return mesh.vertices, mesh.faces

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

def prism_wall_vertices_old(vertices, faces, floor_val=0):

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

def prism_wall_vertices(vertices_top, faces, floor_val=0):
    open_edges = get_open_edges(faces)
    wall_vertices = []

    for edge in open_edges:
        # i1 and i2 are indices from the Top Cap (Z=Z1)
        i1, i2 = edge 
        
        # 3D points at the Top (Z=Z1)
        v_top_1 = vertices_top[i1]
        v_top_2 = vertices_top[i2]

        # 3D points at the Bottom (Z=Z0)
        # We assume the Bottom cap vertices have the same X,Y as Top
        v_bot_1 = np.array([v_top_1[0], v_top_1[1], floor_val])
        v_bot_2 = np.array([v_top_2[0], v_top_2[1], floor_val])

        # Create two triangles for the quad. 
        # Note the order: to face OUTWARD, we must reverse the top edge's direction.
        
        # Triangle 1: Top1 -> Bottom1 -> Bottom2
        wall_vertices.append([v_top_1, v_bot_1, v_bot_2])
        
        # Triangle 2: Top1 -> Bottom2 -> Top2
        wall_vertices.append([v_top_1, v_bot_2, v_top_2])
    
    return wall_vertices

def robust_triangulate(vertices, edges, holes=None):
    # 1. Geometry Cleaning & Shell Formation
    lines = [vertices[edge] for edge in edges]
    merged_lines = unary_union(MultiLineString(lines))
    polygons = list(polygonize(merged_lines))
    
    if not polygons:
        raise ValueError("Could not form a closed polygon from edges.")
    
    # Use the largest polygon formed as our boundary shell
    boundary_poly = unary_union(polygons)
    
    # 2. Prep Holes
    # Ensure holes are coordinate sequences for the Polygon constructor
    holes_coords = []
    if holes:
        for h in holes:
            # Strip redundant last point if it's identical to the first
            h_arr = np.array(h)
            if np.allclose(h_arr[0], h_arr[-1]):
                h_arr = h_arr[:-1]
            holes_coords.append(h_arr)

    # 3. Create the Formal Polygon with Holes
    # orient() ensures Shell is CCW and Holes are CW for proper CDT behavior
    poly = orient(Polygon(boundary_poly.exterior.coords, holes=holes_coords))

    # 4. Generate CDT and Handle GeometryCollection
    # We call the CDT tool (e.g., from your 'puzzle' or 'shapely.ops' extension)
    tri_output = constrained_delaunay_triangles(poly)
    
    # FIX: Check if output is a GeometryCollection and extract geometries
    if hasattr(tri_output, 'geoms'):
        tri_list = tri_output.geoms
    else:
        tri_list = tri_output # Assume it's already a list

    keep_coords = []
    for tri in tri_list:
        # 5. Final Filter
        # Even with CDT, it might fill the convex hull; keep only triangles inside the poly
        if poly.contains(tri.centroid):
            keep_coords.append(np.array(tri.exterior.coords)[:3])
    
    if not keep_coords:
        return np.array([]), np.array([])

    # 6. Re-indexing into Vertices and Faces
    flat_coords = np.vstack(keep_coords)
    unique_verts, inverse_indices = np.unique(
        np.round(flat_coords, 6), 
        axis=0, 
        return_inverse=True
    )
    faces = inverse_indices.reshape(-1, 3)
    
    return unique_verts, faces

def make_puzzle_pts(width, b, m, base_n, a=0, border_buffer=None, tol=0.4):
    # Handle width as scalar or list

    print(tol)

    if border_buffer is None:
        border_buffer = base_n

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
        temp = make_puzzle_piece(b, m, ni, nj, base_n, len(grid_range_x), len(grid_range_y), a=a, tol=tol)
        
        # Apply the global offset
        temp[:, 0] += (ni * bx)
        temp[:, 1] += (nj * by)
        pts_list.append(temp)

    for n,pts in enumerate(pts_list):
      pts[pts==0] = -border_buffer
      pts[pts[:,0]>(width_x-base_n),0] = width_x + border_buffer
      pts[pts[:,1]>(width_y-base_n),1] = width_y + border_buffer
      pts_list[n] = pts	

    return pts_list

def make_puzzle_piece(b, m, ni, nj, base_n, len_x, len_y, a=0, tol=1):

    def get_notched_square(b, m, n_left, n_bottom, n_right, n_top, a=0, tol=1):
        
        b_x, b_y = b if isinstance(b, (list, tuple)) else (b, b)

        # Logic: If n is negative (a tab), we increase the margin 'm' 
        # by the tolerance to make the tab narrower.
        m_l = m + tol if n_left < 0 else m
        m_b = m + tol if n_bottom < 0 else m
        m_r = m + tol if n_right < 0 else m
        m_t = m + tol if n_top < 0 else m

        x = np.array([
            # Left Side
            [a, b_y], 
            [a, b_y-m_l], [a+n_left, b_y-m_l], [a+n_left, m_l], [a, m_l], 
            # Bottom Side
            [a, a], 
            [m_b, a], [m_b, a+n_bottom], [b_x-m_b, a+n_bottom], [b_x-m_b, a], 
            # Right Side
            [b_x, a], 
            [b_x, m_r], [b_x-n_right, m_r], [b_x-n_right, b_y-m_r], [b_x, b_y-m_r], 
            # Top Side
            [b_x, b_y], 
            [b_x-m_t, b_y], [b_x-m_t, b_y-n_top], [m_t, b_y-n_top], [m_t, b_y], 
            # Return to Start
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

    return get_notched_square(b, m, n_l, n_b, n_r, n_t, a, tol)

