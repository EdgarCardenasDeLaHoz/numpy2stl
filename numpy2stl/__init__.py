from itertools import product
import struct
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3

import shapely.ops as ops

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing, LineString, orient


############################# convert array to facet list ##########################################

def numpy2stl(A, mask_val=None, solid=False):
    """
    Reads a numpy array, and list of facets

    Inputs:
     A (ndarray) - an 'm' by 'n' 2D numpy array

    Optional input:

     mask_val (float) - any element of the inputted array that is less than this value will not be included in the mesh.
                    default renders all vertices (x > -inf for all float x)

     solid (bool): sets whether to create a solid geometry (with sides and a bottom) or not.
                                        
    Returns: facets
    """
    if mask_val is None:
        mask_val = A.min() - 1.
        
    min_val = 0        
    
    print("Creating top...")
    top_facets = top_facet_from_array(A, mask_val = mask_val)

    if solid:

        #Calculate Floor z value
        vertices = np.array([[facet[3:6],facet[6:9],facet[9:12]] for facet in top_facets])

        print("Creating Edges...")
        edges = get_open_edges(vertices)
        edge_facets = facets_from_edges(edges, floor_val=min_val)

        ##Bottom 
        
        print("Creating bottom...")

        bottom_facets = []
        for i, facet in enumerate(top_facets):

            this_bottom = np.concatenate( [facet[:3], facet[6:8], [min_val], facet[3:5], [min_val], facet[9:11], [min_val]])
            bottom_facets.append(this_bottom)

        facets = np.concatenate([top_facets, edge_facets, bottom_facets])
    
    else:
        facets = top_facets
    
    return facets


def top_facet_from_array(A, mask_val=0, ):
    
    m, n = A.shape
    top_facets = []
    mask = np.zeros((m, n))

    for i, k in product(range(m - 1), range(n - 1)):

        this_pt = np.array([i, k, A[i, k]])
        top_right = np.array([i, k + 1, A[i, k + 1]])
        bottom_left = np.array([i + 1, k , A[i + 1, k]])
        bottom_right = np.array([i + 1, k + 1, A[i + 1, k + 1]])

        n1, n2 = np.zeros(3), np.zeros(3)

        if ((this_pt[-1] > mask_val) and (top_right[-1] > mask_val) and 
                (bottom_right[-1] > mask_val) and (bottom_left[-1] > mask_val)):

            facet = np.concatenate([n1, top_right, this_pt, bottom_right])
            mask[i, k] = 1
            mask[i, k + 1] = 1
            mask[i + 1, k] = 1
            top_facets.append(facet)

            facet = np.concatenate([n2, bottom_right, this_pt, bottom_left])
            top_facets.append(facet)
            mask[i, k] = 1
            mask[i + 1, k + 1] = 1
            mask[i + 1, k] = 1

    top_facets = np.array(top_facets)
    return top_facets


def limit_facet_size(facets, max_width=1000., max_depth=1000., max_height=1000.):
    """
    max_width, max_depth, max_height (floats) - maximum size of the stl object (in mm). 
                    Match this to the dimensions of a 3D printer platform.
    """
    xsize = facets[:, 3::3].ptp()
    if xsize > max_width:
        facets = facets * float(max_width) / xsize

    ysize = facets[:, 4::3].ptp()
    if ysize > max_depth:
        facets = facets * float(max_depth) / ysize

    zsize = facets[:, 5::3].ptp()
    if zsize > max_height:
        facets = facets * float(max_height) / zsize

    return facets


def facets_from_edges(edges, floor_val=0):
    """
    
    """ 

    edge_facets = []
    n1,n2 = np.zeros(3),np.zeros(3)

    for i, e in enumerate(edges): 

        top_left = np.concatenate([e[0,0:2], [floor_val]])
        top_right = np.concatenate([e[1,0:2], [floor_val]])
        bottom_left = np.array(e[0])
        bottom_right = np.array(e[1])

        facet = np.concatenate([n1, top_right, top_left, bottom_right])
        edge_facets.append(facet)

        facet = np.concatenate([n2, bottom_right, top_left, bottom_left])
        edge_facets.append(facet)

    edge_facets = np.array(edge_facets)
    return edge_facets

def roll2d(image, shifts):
    return np.roll(np.roll(image, shifts[0], axis=0), shifts[1], axis=1)



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

    ## Todo use shared edges not vertices to make surface 

    if normals is None:
        normals = calculate_normals(vertices)
    
    _, vertices_idx = vertices_to_index(vertices)

    surfaces, surf = [],[0]

    b_checked = np.zeros(vertices_idx.shape[0],dtype=bool)    
    vert, norm  = vertices_idx[0], normals[0] 
    b_checked[0] = True
    to_do = []    
    
    same_normal = (norm==normals).all(axis=1)
    sub_vert = vertices_idx[same_normal]
    
    while (b_checked==False).any():      
                
        same_surface = same_normal.copy()
        same_edges = ((sub_vert==vert[0])|(sub_vert==vert[1])|(sub_vert==vert[2])).any(axis=1)
        same_surface[same_normal] = same_edges
        same_surface = same_surface & (b_checked==False)
        
        #Continue treversing touching vertices or in queue
        if (same_surface).any() or (len(to_do)>0):
            idxs = np.where(same_surface)[0]         
            surf.extend(idxs)
            to_do.extend(idxs)
            b_checked[idxs] = True 
            next_idx = to_do.pop()
            vert = vertices_idx[next_idx]
            
        #If queue is empty start a new Surface     
        else:
            surfaces.append(list(set(surf)))
            surf = []
            idx = np.where(b_checked==False)[0][0]  
            
            surf.append(idx)
            vert, norm  = vertices_idx[idx], normals[idx]
            b_checked[idx] = True 
            
            same_normal = np.array((norm==normals).all(axis=1))
            sub_vert = vertices_idx[same_normal]
            
    surfaces.append(list(set(surf)))
    return surfaces

def simplify_object_3D(vertices):
    """
    """
    surfaces = get_surfaces(vertices)
    normals = calculate_normals(vertices)
    

    simplified_vertices = []
    for surf in surfaces:
        norm = normals[surf][0]
        surface_vertices = vertices[surf]   

        edges = get_open_edges(surface_vertices)
        perimeters = get_ordered_perimeter(edges)
        perimeter_2d = [rotate_3D(peri, norm, [0,0,1] ) for peri in perimeters ]
        angles = [get_perimeter_angles(peri[:,0:2]) for peri in perimeter_2d]
        points = [peri[angles[n] != 180] for n, peri in enumerate(perimeters)]
        simplified_vertices.append(points)

    keep_vertices = np.concatenate([np.concatenate(vert) for vert in simplified_vertices])   

    vertices_out = []
    for surf in surfaces:
        norm = normals[surf][0]
        surface_vertices = vertices[surf]   
        
        if len(surface_vertices) > 1:
            surface_vertices = simplify_surface(surface_vertices,norm, keep_vertices=keep_vertices) 
            
        vertices_out.append( surface_vertices )  

    vertices_out = np.concatenate(vertices_out)
    vertices_out = validate_object(vertices_out)    
            
    return vertices_out

def simplify_surface(vertices,normal, keep_vertices=None):
    """
    """

    ## Get ordered perimeter points 
    edges = get_open_edges(vertices)
    perimeters = get_ordered_perimeter(edges)

    if keep_vertices is not None:
        perimeters = [peri[(peri[:,None] == keep_vertices).all(axis=2).any(axis=1)] for peri in perimeters]

    ## Flatten boundry to 2D
    perimeter_2d = [rotate_3D(peri, normal, [0,0,1] ) for peri in perimeters ]
    points_2d_flat = np.concatenate(perimeter_2d)
    
    ## Simplify Triangulation 
    vert_2D_tri = triangulate_polygon(perimeter_2d)
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
        print("Open edges exist in object")

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
    polygon = Polygon(LR_ext, points[1:])   
    ##TODO: Make Triangulation Edge Constrained 
    tri_Poly = ops.triangulate(polygon)

    tri_Poly_within = [tri for tri in tri_Poly if tri.within(polygon)]

    ## Check if Edge Case Convexity has occurred -- if the area out is not the area in
    intersect_area =  (ops.unary_union(tri_Poly_within)).area / polygon.area
    if np.isclose(intersect_area,1) == False:
        tri_Poly_outside = [tri for tri in tri_Poly if not tri.within(polygon)]    
        if  len(tri_Poly_outside)>0:      
            ## Union the remaining polygons and split them by the edge again
            Poly_outside = ops.unary_union(tri_Poly_outside)
            Poly_outside = ops.split(Poly_outside,LineString(points[0]))
            Poly_outside = [ops.triangulate(tri) for tri in Poly_outside]
            Poly_outside = [A for B in Poly_outside for A in B]
            tri_Poly_within.extend([orient(tri, sign=1) for tri in Poly_outside if tri.within(polygon)])

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
    traces_all,trace = [],[]
    this_idx = edges_left[0][0]
    #while len(edges_left)>0:
    #
    for _ in range(len(edges_idx)):
        trace.append(this_idx)  
        list_idx = np.nonzero((edges_left == this_idx)[:,0])[0][0]

        next_edge = edges_left.pop(list_idx)  
        this_idx = next_edge[1]
            
        if np.any(this_idx==trace):
            trace.append(this_idx)
            traces_all.append(trace)
            trace = []    
            if len(edges_left)>0:
                this_idx = edges_left[0][0]
        
    #Validate the traces to be closed perimeters
    traces_out = []
    for trace in traces_all:
        if (trace[0]==trace[-1]): 
            trace = trace[0:-1]    
        traces_out.append(trace)

    perimeters = [uni_edges[idx] for idx in traces_out]

    return perimeters

def get_perimeter_angles(peri):

    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = peri_wrapped[0:-2] - peri_wrapped[1:-1]
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]
    hypo = np.array(np.linalg.norm(ba, axis=1))* np.array(np.linalg.norm(bc, axis=1))

    dot_prod = np.sum(ba*bc, axis=1) 
    cross_prod = np.cross(bc,ba)
    angle = np.arctan2(cross_prod,dot_prod)

    angle = np.degrees(angle)
    angle[angle<0] += 360
    return angle 

def partition_concave(perimeter):
    angles = np.concatenate([get_perimeter_angles(p[:,0:2]) for p in perimeter])
    convex_idx = np.nonzero( (angles == np.max(angles)) & (angles > 180)  )[0]

    if len(convex_idx)==0:
        return [perimeter]
    
    convex_idx = convex_idx[0]   
    opposite_idx = find_closest_opposite(perimeter, convex_idx )
     
    if  convex_idx > opposite_idx:
        idx1, idx2 = convex_idx, opposite_idx
    else:
        idx1, idx2 = opposite_idx, convex_idx

    first_edge = np.concatenate(perimeter)[[idx1,idx2]]
    last_edge =  np.concatenate(perimeter)[[idx2,idx1]]

    edges = perimeters_to_edges(perimeter)
    edges = np.concatenate([[first_edge],edges,[last_edge] ])
    
    peri_new = get_ordered_perimeter(edges)
    peri_out = []
    for p in peri_new:
        peri_out.extend( partition_concave(peri_new))
    return peri_out


def find_closest_opposite(perimeter, idx):

    perimeter_flat = np.concatenate(perimeter)
    convex_point = perimeter_flat[idx]
    bisect = get_bisect(perimeter_flat)[idx]
    bisect = bisect / np.linalg.norm(bisect)
    
    other_idx = np.array([i for i in range(len(perimeter_flat)) if i != idx])

    point_dist = perimeter_flat[other_idx] - convex_point
    point_dist = point_dist / (np.linalg.norm(point_dist, axis=1))[:,None] 
    opposite_dist = np.dot(point_dist,bisect)
    opposite_idx = other_idx[np.nonzero(opposite_dist == np.max(opposite_dist))[0][0]]
    return opposite_idx 

def get_bisect(peri):

    peri_wrapped = np.concatenate([[peri[-1]],peri,[peri[0]]])

    ba = peri_wrapped[0:-2] - peri_wrapped[1:-1] 
    ba = ba / np.array(np.linalg.norm(ba, axis=1))[:,None]
    
    bc = peri_wrapped[2::] - peri_wrapped[1:-1]
    bc = bc / np.array(np.linalg.norm(bc, axis=1))[:,None]
   
    bisect = -(ba + bc)
    #bisect = bisect / np.array(np.linalg.norm(bisect, axis=1))[:,None]
    return bisect
    
def perimeters_to_edges(perimeters):

    edges = np.concatenate([ np.array(list(zip(p, np.roll(p,1,axis=0) )))  for p in  perimeters ])

    return edges 


    


######################### Functions for Plotting in 3D ################################

def plot_edges_3d(edges,ax=None):

    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    for e in edges:
        color = np.random.rand(3)
        ax.plot3D([e[0,0],e[1,0]], [e[0,1],e[1,1]], [e[0,2],e[1,2]], color=color)
        ax.plot3D([e[1,0]], [e[1,1]], [e[1,2]], 'o', color=color)
        
    x = edges[:,:,0].ravel()
    y = edges[:,:,1].ravel()
    z = edges[:,:,2].ravel()
    
    set_limits_3D(ax,x,y,z)
    
    return ax
    
def plot_perimeters_3d(perimeter,ax=None):

    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    for p in perimeter:
        color = np.random.rand(3)
        ax.plot3D(p[:,0],p[:,1],p[:,2], color=color)

    x = perimeter[0][:,0].ravel()
    y = perimeter[0][:,1].ravel()
    z = perimeter[0][:,2].ravel()

    set_limits_3D(ax,x,y,z)
    
    return ax

def draw_3D_vertices(vertices, surfaces=None, surf_color=None, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = plt3.Axes3D(fig)

    if surfaces is None:
        surfaces = [range(len(vertices))]
         
    for i, surf in enumerate(surfaces):
        tri = plt3.art3d.Poly3DCollection(vertices[surf])
        
        if surf_color is None:
            face_color = np.random.rand(3)
        else:
            face_color = surf_color[i]
            
        tri.set_facecolor(face_color)
        tri.set_edgecolor('k')
        ax.add_collection3d(tri)

    x = vertices[:,0,0]
    y = vertices[:,0,1]
    z = vertices[:,0,2]

    set_limits_3D(ax,x,y,z)
    
    plt.show()
    return ax

def set_limits_3D(ax,x,y,z):
    
    xlim = np.array((np.amin(x)-1, np.amax(x)+1))
    ylim = np.array((np.amin(y)-1, np.amax(y)+1))
    zlim = np.array((np.amin(z)-1, np.amax(z)+1))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)  

################### Write to file ##########################

def _build_binary_stl(facets):
    """returns a string of binary binary data for the stl file"""

    BINARY_HEADER = "80sI"
    BINARY_FACET = "12fH"

    lines = [struct.pack(BINARY_HEADER, b'Binary STL Writer', len(facets)), ]
    for facet in facets:
        facet = list(facet)
        facet.append(0)  # need to pad the end with a unsigned short byte
        lines.append(struct.pack(BINARY_FACET, *facet))
    return lines


def _build_ascii_stl(facets):
    """returns a list of ascii lines for the stl file """

    ASCII_FACET = """  facet normal  {face[0]:e}  {face[1]:e}  {face[2]:e}
        outer loop
        vertex    {face[3]:e}  {face[4]:e}  {face[5]:e}
        vertex    {face[6]:e}  {face[7]:e}  {face[8]:e}
        vertex    {face[9]:e}  {face[10]:e}  {face[11]:e}
        endloop
    endfacet"""

    lines = ['solid ffd_geom', ]
    for facet in facets:
        lines.append(ASCII_FACET.format(face=facet))
    lines.append('endsolid ffd_geom')
    return lines

def writeSTL(facets, file_name, ascii=False):
    """writes an ASCII or binary STL file"""

    f = open(file_name, 'wb')
    if ascii:
        lines = _build_ascii_stl(facets)
        lines_ = "\n".join(lines).encode("UTF-8")
        f.write(lines_)
    else:
        data = _build_binary_stl(facets)
        data = b"".join(data)
        f.write(data)

    f.close()
