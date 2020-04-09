
import numpy as np

import triangle as tr

import shapely.ops as ops
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing, LineString, orient

from .view import *

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
        elif len(next_list_idx)>0:
            next_list_idx = next_list_idx[0] 
        else:
            print(next_list_idx)
            print(uni_edges[trace])

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

def simplify_perimeters( perimeters, normal):

    simplified_perimeters = []
    for line in perimeters:
        if len(line)>4:
            line_2d = rotate_3D(line, normal, [0,0,1] )
            angles = get_perimeter_angles( line_2d) 
            simpified_line = np.array(line[angles != 180])
        else:
            simpified_line = line
        simplified_perimeters.append(simpified_line)

    return simplified_perimeters

def perimeter_to_2D( perimeters, normal, simplify_lines=False):
    
    perimeter_2d = [rotate_3D(peri, normal, [0,0,1] ) for peri in perimeters ]
    if simplify_lines:
        angles = [get_perimeter_angles(line) for line in perimeter_2d]
        perimeter_2d = [np.array(line[angles[n] != 180]) for n, line in enumerate(perimeter_2d)]

    return perimeter_2d
    
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


def get_angle_vectors(ba, bc ):

    ba = ba / np.array(np.linalg.norm(ba, axis=1))[:,None]
    bc = bc / np.array(np.linalg.norm(bc, axis=1))[:,None]
    dot_prod = np.sum(ba*bc, axis=1) 
    cross_prod = np.cross(ba,bc)

    angle = np.arctan2(cross_prod , dot_prod)
    angle = np.degrees(angle)
    angle[angle<0] += 360

    return angle

def get_perimeter_angles(peri):

    if peri.shape[1]==3:
        peri = peri[:,0:2]
        
    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = peri_wrapped[0:-2] - peri_wrapped[1:-1]
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]
    
    angle = get_angle_vectors(bc, ba )
   
    return angle 

def perimeters_to_edges(perimeters):

    edges = [np.stack([ p, np.roll(p,1,axis=0) ],axis=1) for p in  perimeters ]

    return edges 

def get_perimeter_normal(perimeter):

    n = 0
    normal = np.cross(perimeter[n+1] - perimeter[n] , perimeter[n-1] - perimeter[n])
    while (np.linalg.norm(normal) == 0) and (n < (len(perimeter))-1):
        normal = np.cross(perimeter[n+1] - perimeter[n] , perimeter[n-1] - perimeter[n])
        n = n+1

    if (np.linalg.norm(normal)==0):
        print(perimeter)

    normal = normal / np.linalg.norm(normal)
    
    return normal

def get_area(perimeter):
    
    normal = get_perimeter_normal(perimeter)
    perimeter = rotate_3D(perimeter, normal, [0,0,1] )

    x = perimeter[:,0]
    y = perimeter[:,1]

    area = 0.5*np.abs(np.dot(x,np.roll(y,1))   -  np.dot(y,np.roll(x,1)))
    
    orient = get_orientation(perimeter)
    if orient[0]<0:
        area = -area

    return area

def get_orientation(perimeter):

    edges = perimeters_to_edges([perimeter])
    orientation = [np.sum( np.diff(e[:,:,0],n=1,axis=1) * np.sum(e[:,:,1],axis=1)[:,None] ) for e in edges]
    return orientation

def set_orientation(perimeter, orientation=1):

    perimeter_out = []
    for p in perimeter:
        result = get_orientation(p)[0]
        if (orientation < 0 and result > 0) or (orientation > 0 and result < 0):
            p = p[::-1]
        perimeter_out.append(p)

    return perimeter_out

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


def triangulate_polygon(perimeters):

    grouped_edges = perimeters_to_edges(perimeters)

    holes = []
    if len(grouped_edges)>1:
        for i in range(1,len(grouped_edges)):

            verts, faces = triangulate_edges(grouped_edges[i])
            tris = verts[faces]
            cents = tris.mean(axis=1)
            holes.append(cents[0])
    else:
        holes = None

    all_edges = np.concatenate(grouped_edges)
    vertices, faces = triangulate_edges(all_edges, holes=holes)

    return vertices, faces


def triangulate_edges(edges, holes=None):

    uni_edges, edge_idx = np.unique(np.concatenate(edges), axis=0, return_inverse=True)
    edge_idx = edge_idx.reshape((-1,2))

    if holes is None:
        shape = {"vertices": uni_edges[:,[0,1]], "segments": edge_idx}   
    else:
        shape = {"vertices": uni_edges[:,[0,1]], "segments": edge_idx, "holes": holes}
    t = tr.triangulate(shape,'p')  

    vertices = t["vertices"]
    faces =  t['triangles'] 

    vertices = vertices[edge_idx[:,0]]
    faces = np.argsort(edge_idx[:,0])[faces]

    return vertices,faces


### Partition Polygon

def triangulate_polygon__(points):
    """
    
    """
    if len(points[0])==3:
        return points

    if len(points[0])<3:
        return []

    LR_ext = LinearRing(points[0])
    polygon = Polygon(LR_ext)   
    tri_Poly = ops.triangulate(polygon)
    tri_Poly_within = [tri for tri in tri_Poly if tri.within(polygon)]
    vert =  np.array([np.array(tri.exterior.xy)[:,0:3].T for tri in tri_Poly_within]) 

    all_points = np.concatenate(points)
    zval = np.full((*vert.shape[0:2],1), all_points[0,2] )  
    vertices_tri = np.concatenate((vert, zval ), axis =2)

    return vertices_tri

def partition_to_convex(perimeter_2d):

    ## Triangles are already convex
    if len(perimeter_2d[0] ) == 3:
        return  perimeter_2d

    ## More that one perimeter implies polygon with holes
    if len(perimeter_2d)>1:

        peri = partition_holes(perimeter_2d)
        peri_out = [partition_concave([p]) for p in peri]
        peri_out = [p for peri in peri_out for p in peri]
    else:   
        peri_out = partition_concave(perimeter_2d)

    return peri_out

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

    angle_list = [get_perimeter_angles(p) for p in perimeter]
    convex_idx_list = [np.nonzero( (a>180))[0] for a in angle_list]
    
    p_exterior = perimeter[0]
    exterior_idx = np.array(range(len(p_exterior)))
    p_idx = 1
    
    edges = [] 

    flat_idx_start = len(p_exterior)

    for p_interior in perimeter[1::]:

        convex_idx = convex_idx_list[p_idx]
        bisect = get_bisect_vector(p_interior)
        
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

    perimeter = perimeter[0]
    if perimeter.shape[1]==3:
        perimeter = perimeter[:,0:2]
        
    adjcent = np.array([idx-1,idx,idx+1])
    adjcent = np.mod(adjcent,len(perimeter)) 
    other_idx = np.array([i for i in range(len(perimeter)) if i not in adjcent])

    ba = np.array([perimeter[idx-1] - perimeter[idx]])
    bc = perimeter[other_idx] - perimeter[idx]
    
    cross_angle = get_angle_vectors(bc, ba )    
    bc = np.array([perimeter[adjcent[2]] - perimeter[idx]])
    max_angle = get_angle_vectors(bc, ba )    

    other_idx = other_idx[(cross_angle < max_angle)  & (cross_angle > 0)]

    convex_point = perimeter[idx]

    dist_vect = perimeter[other_idx] - convex_point
    dist_vect_n = dist_vect / (np.linalg.norm(dist_vect, axis=1))[:,None]    

    bisect = get_bisect_vector(perimeter)[idx] 

    cos_dist = np.dot(dist_vect_n ,bisect)
    sin_dist = np.cross(dist_vect_n ,bisect)
    
    opposite_dist = (1-cos_dist) * (np.linalg.norm(dist_vect, axis=1)) * abs(sin_dist)

    opposite_idx = other_idx[np.argmin(opposite_dist)]
    return opposite_idx 

def partition_concave(perimeter):

    angles = np.concatenate([get_perimeter_angles(p) for p in perimeter])
    
    convex_angles = angles[angles > 180]

    if len(convex_angles)==0:
        return perimeter

    convex_idx = np.nonzero( angles == np.max(convex_angles) )[0][0] 
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

def get_bisect_vector(peri):

    Z = None
    if peri.shape[1]==3:
        Z = peri[:,2]
        peri = peri[:,0:2]

    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = np.array([[1,0]])
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]

    base_angle = get_angle_vectors(ba, bc )    

    angle = get_perimeter_angles(peri)/2 + base_angle
    angle = np.deg2rad(angle)
    bisect = np.stack([np.cos(angle), np.sin(angle)],axis=-1)
    bisect = bisect / np.array(np.linalg.norm(bisect, axis=1))[:,None]

    if Z is not None:
        bisect = np.concatenate([bisect,Z[:,None]],axis=1)

    return bisect


def triangulate_convex(points):
    """
    
    """
    if len(points[0])==3:
        return points

    if len(points[0])<3:
        return []

    LR_ext = LinearRing(points[0])
    polygon = Polygon(LR_ext)   
    tri_Poly = ops.triangulate(polygon)
    tri_Poly_within = [tri for tri in tri_Poly if tri.within(polygon)]
    vert =  np.array([np.array(tri.exterior.xy)[:,0:3].T for tri in tri_Poly_within]) 

    all_points = np.concatenate(points)
    zval = np.full((*vert.shape[0:2],1), all_points[0,2] )  
    vertices_tri = np.concatenate((vert, zval ), axis =2)

    return vertices_tri
