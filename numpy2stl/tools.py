
import numpy as np
import shapely.ops as ops

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing, LineString, orient

from .view import *

####################### 3D Object Processing tools #######################################

def calculate_normals(vertices):
    """
    """
    normals = np.cross(vertices[:,1] - vertices[:,0] , vertices[:,2] - vertices[:,0])
    normals = normals / np.array([np.linalg.norm(normals, axis=1)]).T
    
    return normals

def vertices_to_index(vertices):
    
    vertices_list = np.reshape(vertices, (vertices.shape[0]*3,3))
    uni_vertices, vertices_idx = np.unique(vertices_list, axis=0 , return_inverse=True)
    vertices_idx = np.reshape(vertices_idx, (vertices.shape[0],3))
    
    return uni_vertices, vertices_idx


def get_surfaces(vertices, normals=None):
    """ 
        vertices is a M x 3 x 3 array of M triangles in 3D, 
        normals are [x,y,z] float normal vertices 
    """
    ## Todo: use shared edges not vertices to make surface 

    if normals is None:
        normals = calculate_normals(vertices)
    
    _, normals_idx = np.unique(normals.round(decimals=9), axis=0 , return_inverse=True)
    _, tri_vertices_idx = vertices_to_index(vertices)

    surfaces = []

    for n_idx in range(  np.max(normals_idx)+1  ):

        same_normal = normals_idx == n_idx   
        idx_list = np.nonzero(normals_idx == n_idx)[0]
        idx = idx_list[0]
        if len(idx_list)==1:
            surfaces.append([idx])
            continue

        surf, to_do = [],[]
        surf.append(idx) 
        
        sub_vert = tri_vertices_idx[same_normal]  
        tri = tri_vertices_idx[idx]
        b_checked = np.zeros(sub_vert.shape[0],dtype=bool)  
        b_checked[0] = True
        
        while (b_checked==False).any():      
                    
            same_edges = (sub_vert==tri[0])|(sub_vert==tri[1])|(sub_vert==tri[2])
            shared_edges= np.sum(same_edges,axis=1)==2
            shared_edges = shared_edges & (b_checked==False)

            # shared_edges = b_checked==False
            # same_edges = (sub_vert[shared_edges] == tri[:,None,None]).any(axis=0)
            # any_edge = same_edges.any(axis=1)
            # any_edge[any_edge] = np.sum(same_edges[any_edge],axis=1)==2
            # shared_edges[shared_edges] = any_edge
            
            #Continue treversing touching vertices or in queue
            if (shared_edges).any() or (len(to_do)>0):

                sub_idxs = np.where(shared_edges)[0]         
                to_do.extend(sub_idxs)
                b_checked[sub_idxs] = True 
                
                surf.extend(idx_list[sub_idxs])                
                
                next_idx = to_do.pop()
                tri = sub_vert[next_idx]
                
            #If queue is empty start a new Surface on same normal plane   
            else:

                surfaces.append(np.unique(surf))
                surf = []

                sub_idx = np.where(b_checked==False)[0][0]  
                surf.append( idx_list[sub_idx] )
                tri  = sub_vert[ sub_idx  ]
                b_checked[sub_idx] = True 

        surfaces.append(np.unique(surf))
                
    return surfaces

def simplify_object_3D(vertices):
    """
    """
    surfaces = get_surfaces(vertices)
    normals = calculate_normals(vertices)

    required_vertices = get_required_vertices(vertices,surfaces,normals)

    vertices_out = []
    n = 0
    for surf in surfaces:

        print(n)
        n = n+1
        norm = normals[surf][0]
        surface_vertices = vertices[surf]   

        if len(surface_vertices)>4:
            surface_vertices = simplify_surface(surface_vertices,norm, keep_vertices=required_vertices) 

        vertices_out.append( surface_vertices )  

    vertices_out = np.concatenate(vertices_out)
    vertices_out = validate_object(vertices_out)    

    print( len(vertices), " vertices reduced to " , len(vertices_out))
            
    return vertices_out

def get_required_vertices(vertices,surfaces=None,normals=None):

    if surfaces is None:
        surfaces = get_surfaces(vertices)

    if normals is None:
        normals = calculate_normals(vertices)

    simplified_vertices = []
    for surf in surfaces:
        norm = normals[surf][0]
        surface_vertices = vertices[surf]   
        if len(surface_vertices)>4:
            edges = get_open_edges(surface_vertices)
            perimeters = get_ordered_perimeter(edges)
            simplified_perimeters = simplify_perimeters( perimeters, norm)
            if len(simplified_perimeters)==0:
                continue
        else:
            simplified_perimeters = surface_vertices

        simplified_vertices.append(  np.concatenate( simplified_perimeters )  )

    required_vertices = np.concatenate(simplified_vertices)  
    return required_vertices

def simplify_perimeters( perimeters, normal):

    simplified_perimeters = []
    for peri in perimeters:
        if len(peri)>4:
            perimeter_2d = rotate_3D(peri, normal, [0,0,1] )
            angles = get_perimeter_angles( perimeter_2d[:,0:2]) 
            sim_peri = np.array(peri[angles != 180])
        else:
            sim_peri = peri
        simplified_perimeters.append(sim_peri)

    return simplified_perimeters


def perimeter_to_2D( perimeters, normal, simplify_lines=False):
    
    perimeter_2d = [rotate_3D(peri, normal, [0,0,1] ) for peri in perimeters ]

    if simplify_lines:
        angles = [get_perimeter_angles(peri[:,0:2]) for peri in perimeter_2d]
        perimeter_2d = [np.array(peri[angles[n] != 180]) for n, peri in enumerate(perimeter_2d)]

    return perimeter_2d
    
def simplify_surface(vertices,normal, keep_vertices=None):
    """
    """
    ## Get ordered perimeter points 
    edges = get_open_edges(vertices)
    perimeters = get_ordered_perimeter(edges)

    if keep_vertices is not None:
        
        perimeters = [peri[(peri[:,None] == keep_vertices).all(axis=2).any(axis=1)] for peri in perimeters]

    ## Flatten boundry to 2D
    perimeter_2d = perimeter_to_2D( perimeters, normal, simplify_lines=False)
    points_2d_flat = np.concatenate(perimeter_2d)
    
    ## Simplify Triangulation
    if len(perimeter_2d)>1:

        peri = partition_holes(perimeter_2d)
        peri_out = [partition_concave([p]) for p in peri ]
        peri_out = [p for peri in peri_out for p in peri]
        print("x")
    else:   
        if  len(perimeter_2d[0])>3:
            peri_out = partition_concave(perimeter_2d)
            print("y")
        else:
            peri_out = perimeter_2d
            print("z")

    print(peri_out)
    vert_2D_tri = np.concatenate([triangulate_polygon([p]) if len(p)>3 else [p] for p in peri_out ])
    uni_vert_2D, idx_vert_2D = vertices_to_index(vert_2D_tri)

    ## Reconstruct polygon from new triangles 
    idx_to_boundry = np.array( [ np.nonzero(np.isclose(v, points_2d_flat).all(axis=1))[0][0] for v in uni_vert_2D ]   )
    idx_to_boundry = idx_to_boundry[idx_vert_2D]
    idx_to_boundry = np.reshape(idx_to_boundry, vert_2D_tri.shape[0:2])

    all_points = np.concatenate(perimeters)
    simplifed_vert = all_points[idx_to_boundry]

    return simplifed_vert

def validate_object(vertices):
    """
    """
    ## Check for invalid triangles 
    normals = np.cross(vertices[:,1] - vertices[:,0] , vertices[:,2] - vertices[:,0])
    invalid = (normals==0).all(axis=1)
    vertices = vertices[invalid==False]

    ## Check for invalid edges 
    open_edges = get_open_edges(vertices)
    if len(open_edges) > 0:
        print("Open edges exist in object!!")

    return vertices

def rotate_3D(pts, src_vertex, des_vertex):
    """
    """
    mat = rotation_matrix_from_vertices(src_vertex, des_vertex)
    rotated_points = mat.dot(pts.T).T
    
    return rotated_points

def rotation_matrix_from_vertices(vec1, vec2):
    """ 
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vertex
    :param vec2: A 3d "destination" vertex
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a = (vec1 / np.linalg.norm(vec1)).reshape(3)
    b = (vec2 / np.linalg.norm(vec2)).reshape(3)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],  [v[2], 0, -v[0]],  [-v[1], v[0], 0]])

    if s == 0 and c==1:
        rotation_matrix =np.eye(3)
    elif s == 0 and c==-1:
        rotation_matrix = np.eye(3) * [1,-1,-1]
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

def triangulate_polygon(points):
    """
    
    """
    LR_ext = LinearRing(points[0])
    polygon = Polygon(LR_ext)   
    tri_Poly = ops.triangulate(polygon)
    tri_Poly_within = [tri for tri in tri_Poly if tri.within(polygon)]
    vert =  np.array([np.array(tri.exterior.xy)[:,0:3].T for tri in tri_Poly_within]) 
    all_points = np.concatenate(points)
    zval = np.full((*vert.shape[0:2],1), all_points[0,2] )  
    
    vertices_tri = np.concatenate((vert, zval ), axis =2)
    return vertices_tri


def get_open_edges(vertices):
    
    ## Make vertex list
    uni_vertices, vert_idx = vertices_to_index(vertices)
    ## Make Edges list
    edges = np.array([[[v[0],v[1]],[v[1],v[2]],[v[2],v[0]]] for v in vert_idx ])
    edge_list = np.reshape(edges, (edges.shape[0]*3, 2))
    ## Count edge occurance 
    edge_sorted = np.sort(edge_list,axis=1)
    _, edge_idx, edge_counts = np.unique(edge_sorted, axis=0, return_counts=True, return_inverse=True)
    edge_counts_exp = edge_counts[edge_idx]
    ## Select edges that only occur once and thus open
    open_edge_id = edge_list[edge_counts_exp==1]
    ## Reconstruct Edge positions from locations
    open_edges = np.array(uni_vertices[open_edge_id])
    
    return open_edges


def get_ordered_perimeter(edges,validate=False):
    """
    """
    edges_list = np.reshape(edges, (-1,3))
    uni_edges, edges_idx = np.unique(edges_list, axis=0 , return_inverse=True)
    edges_idx = np.reshape(edges_idx, (edges.shape[0],-1))
    
    edges_left = list(edges_idx)
    this_idx = edges_left[0][0]
    traces_all,trace = [],[]
    
    for _ in range(len(edges_idx)):
        trace.append(this_idx)  
        next_list_idx = np.nonzero((edges_left == this_idx)[:,0])[0]

        if len(next_list_idx)>1 and len(trace)>1:

            next_idx_options = np.array(edges_left)[next_list_idx]
            smallest_idx = find_smaller_angle(uni_edges, trace[-2], trace[-1], next_idx_options[:,1] )
            next_list_idx = next_list_idx[smallest_idx]
        else:
            next_list_idx = next_list_idx[0] 

        next_edge = edges_left.pop(next_list_idx)  

        this_idx = next_edge[1]
            
        if this_idx in trace:
            trace.append(this_idx)
            traces_all.append(trace)
            trace = []    
            if len(edges_left)>0:
                this_idx = edges_left[0][0]
        
    #Validate the traces to be closed perimeters
    traces_out = []
    for trace in traces_all:
        if (trace[0]!=trace[-1]): 
            continue
        trace = trace[0:-1]   
        if len(trace)<3:
            continue 
        traces_out.append(trace)

    perimeters = [uni_edges[idx] for idx in traces_out]

    return perimeters

def find_smaller_angle(points, last_idx, this_idx, next_idx_options ):

    bc = [points[last_idx] - points[this_idx]]
    ba = points[next_idx_options] - points[this_idx]
    ## Return dot product 
    ba = ba / np.linalg.norm(ba,axis=1)[:,None]
    bc = bc / np.linalg.norm(bc,axis=1)[:,None]
    dot_prod = np.sum(ba*bc,axis=1)

    ## Self angles should be last considered (if considered at all)
    dot_prod[np.isclose(dot_prod,1)] = -1
    idx = np.argsort(dot_prod)
    return idx[-1]

def get_perimeter_angles(peri):

    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = peri_wrapped[0:-2] - peri_wrapped[1:-1]
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]
    
    dot_prod = np.sum(ba*bc, axis=1) 
    cross_prod = np.cross(bc,ba)
    angle = np.arctan2(cross_prod,dot_prod)

    angle = np.degrees(angle)
    angle[angle<0] += 360
    
    return angle 

def partition_holes(perimeter):

    edges_list = perimeters_to_edges(perimeter)
    all_edges = np.concatenate(edges_list)

    new_edges = find_closest_exterior(perimeter)   
    first_edge = np.concatenate(perimeter)[new_edges]
    last_edge =  np.concatenate(perimeter)[new_edges[:,::-1]]
    
    edges = np.concatenate([first_edge,all_edges,last_edge ])
    
    peri_new = get_ordered_perimeter(edges)
    peri_new = set_orientation(peri_new, orientation=1)

    return peri_new

def find_closest_exterior(perimeter):

    angle_list = [get_perimeter_angles(p[:,0:2]) for p in perimeter]
    convex_idx_list = [np.nonzero( (a>180))[0] for a in angle_list]
    
    p_exterior = perimeter[0]
    exterior_idx = np.array(range(len(p_exterior)))
    p_idx = 1
    
    edges = [] 

    flat_idx_start = len(p_exterior)

    for p_interior in perimeter[1::]:

        convex_idx = convex_idx_list[p_idx]
        
        bisect = get_bisect(p_interior)
        bisect = bisect / np.linalg.norm(bisect)

        for idx in convex_idx:

            interior_point = p_interior [ idx ]
            point_dist = p_exterior - interior_point
            point_dist = point_dist / (np.linalg.norm(point_dist, axis=1))[:,None] 
            edge_dist = np.dot( point_dist, bisect[idx] )
            match_idx = exterior_idx [ np.nonzero(edge_dist == np.max(edge_dist))[0][0] ]

            self_idx = idx + flat_idx_start
            new_edge = [self_idx, match_idx ]

            edges.append( new_edge )

        p_idx += 1
        flat_idx_start += len(p_interior)
    
    edges = np.array(edges)

    return edges 

def find_closest_opposite(perimeter, idx):

    perimeter_flat = np.concatenate(perimeter)
    convex_point = perimeter_flat[idx]
    bisect = get_bisect(perimeter_flat)[idx]
    bisect = bisect / np.linalg.norm(bisect)
    
    other_idx = np.array([i for i in range(len(perimeter_flat)) if i not in [idx-1,idx,idx+1]])

    point_dist = perimeter_flat[other_idx] - convex_point
    point_dist = point_dist / (np.linalg.norm(point_dist, axis=1))[:,None] 
    opposite_dist = np.dot(point_dist,bisect)

    opposite_idx = other_idx[np.nonzero(opposite_dist == np.max(opposite_dist))[0][0]]
    return opposite_idx 

def partition_concave(perimeter):

    angles = np.concatenate([get_perimeter_angles(p[:,0:2]) for p in perimeter])
    convex_angles = angles[angles > 180]

    if len(convex_angles)==0:

        return perimeter

    convex_idx = np.nonzero( angles == np.max(convex_angles) )[0]
    convex_idx = convex_idx[0]   
    opposite_idx = find_closest_opposite(perimeter, convex_idx )

    idx1, idx2 = convex_idx, opposite_idx 
    if  convex_idx < opposite_idx:
        idx1, idx2 = idx2, idx1

    edges = perimeters_to_edges(perimeter)
    first_edge = np.concatenate(perimeter)[[idx1,idx2]]
    last_edge =  np.concatenate(perimeter)[[idx2,idx1]]
    edges = np.concatenate([[first_edge],edges[0],[last_edge] ])
    peri_new = get_ordered_perimeter(edges)
    peri_new = set_orientation(peri_new, orientation=1)

    peri_out = []
    for p in peri_new:
        peri_out.extend( partition_concave([p]))

    return peri_out

def get_bisect(peri):

    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = peri_wrapped[0:-2] - peri_wrapped[1:-1] 
    ba = ba / np.array(np.linalg.norm(ba, axis=1))[:,None]
    
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]
    bc = bc / np.array(np.linalg.norm(bc, axis=1))[:,None]
   
    bisect = -(ba + bc)
    bisect = bisect / np.array(np.linalg.norm(bisect, axis=1))[:,None]
    return bisect
    
def perimeters_to_edges(perimeters):

    edges = [np.stack([ p, np.roll(p,1,axis=0) ],axis=1) for p in  perimeters ]

    return edges 

def test_orientation(perimeter):
    edges = perimeters_to_edges([perimeter])
    orientation = [np.sum( np.diff(e[:,:,0],n=1,axis=1) * np.sum(e[:,:,1],axis=1)[:,None] ) for e in edges]
    return orientation

def set_orientation(perimeter, orientation=1):

    perimeter_out = []
    for p in perimeter:
        result = test_orientation(p)[0]
        if (orientation < 0 and result > 0) or (orientation > 0 and result < 0):
            p = p[::-1]
        result = test_orientation(p)[0]
        perimeter_out.append(p)

    return perimeter_out



