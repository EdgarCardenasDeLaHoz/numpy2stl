from itertools import product
import struct
import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3

import shapely.ops as ops

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing, LineString, orient


def numpy2stl(A, scale=0.1, mask_val=None, solid=False, min_thickness_percent=0.1 ):
    """
    Reads a numpy array, and outputs an STL file

    Inputs:
     A (ndarray) - an 'm' by 'n' 2D numpy array

    Optional input:
     scale (float)  - scales the height (surface) of the
                    resulting STL mesh. Tune to match needs

     mask_val (float) - any element of the inputted array that is less than this value will not be included in the mesh.
                    default renders all vertices (x > -inf for all float x)

     solid (bool): sets whether to create a solid geometry (with sides and a bottom) or not.
                    
     min_thickness_percent (float) : when creating the solid bottom face, this
                    multiplier sets the minimum thickness in the final geometry (shallowest interior
                    point to bottom face), as a percentage of the thickness of the model computed up to
                    that point.
    Returns: (None)
    """

    m, n = A.shape
    #A = scale * (A - A.min())

    if mask_val is None:
        mask_val = A.min() - 1.

    min_val = 0
    
    top_facets = []
    mask = np.zeros((m, n))
    print("Creating top mesh...")
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

    if solid:

        print("Computing edges...")
        #Calculate Floor z value
        zvals = top_facets[:, 5::3]
       
        vertices = np.array([[facet[3:6],facet[6:9],facet[9:12]] for facet in top_facets])

        edges = get_open_edges(vertices)
        edge_facets = facets_from_edges(edges, floor_val=min_val)

        ##Bottom 
        bottom_facets = []
        print("Creating bottom...")
        for i, facet in enumerate(top_facets):

            this_bottom = np.concatenate( [facet[:3], facet[6:8], [min_val], facet[3:5], [min_val], facet[9:11], [min_val]])
            bottom_facets.append(this_bottom)

        facets = np.concatenate([top_facets, edge_facets, bottom_facets])
    
    else:
        facets = top_facets
    
    return facets

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
    print("Creating Edges...")
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

####################### 3D Object analysis tools #######################################

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
    if normals is None:
        normals = calculate_normals(vertices)
    
    _, vertices_idx = vertices_to_index(vertices)

    surfaces, surf = [],[0]

    b_checked = np.zeros(vertices_idx.shape[0],dtype=bool)    
    vert = vertices_idx[0]
    norm = normals[0] 
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
            vert = vertices_idx[idx]
            norm = normals[idx]
            b_checked[idx] = True 
            
            same_normal = np.array((norm==normals).all(axis=1))
            sub_vert = vertices_idx[same_normal]
            
    surfaces.append(list(set(surf)))
    return surfaces

def simplify_surface(vertices,normal):
    """
    """
    ## Get ordered perimeter points 
    edges = get_open_edges(vertices)
    perimeters = treverse_lines(edges)
    
    ## Flatten boundry to 2D
    peri_points = [edges[peri,0] for peri in perimeters]
    points2d = [ rotate_3D(points, normal, [0,0,1] ) for points in peri_points ]
    
    ## Simplify Triangulation 
    vertices_tri = triangulate_polygon(points2d)
    vertices_flat = vertices_tri.reshape(-1,vertices_tri.shape[-1])
    
    ## Reconstruct polygon from new triangles 
    all_points_2d = np.concatenate(points2d)
    idx_to_boundry = np.array([ np.nonzero(np.isclose(v, all_points_2d).all(axis=1))[0][0] for v in vertices_flat ])
    idx_to_boundry = np.reshape(idx_to_boundry, vertices_tri.shape[0:2])
    all_points = np.concatenate(peri_points)
    
    simplifed_vert = all_points[idx_to_boundry]
    
    return simplifed_vert

def simplify_object_3D(vertices):
    """
    """
    surfaces = get_surfaces(vertices)
    normals = calculate_normals(vertices)
    
    vertices_out = []
    for surf in surfaces:
        norm = normals[surf][0]
        surface_vertices = vertices[surf]   
        if len(surface_vertices) > 1:
            surface_vertices = simplify_surface(surface_vertices,norm) 
            surface_vertices = validate_object(surface_vertices)
        vertices_out.append( surface_vertices )  
            
    return np.concatenate(vertices_out)

def treverse_lines(lines_in):
    """
    """
    starts_left = list(lines_in[:,0])

    traces_all,trace = [],[]
    
    new_start = starts_left[0]
    this_idx = np.where( (new_start==lines_in[:,0]).all(axis=1) )[0][0]
    trace.append(this_idx)
    
    while len(starts_left)>0:
        end_point = lines_in[this_idx,1]
        next_idx_left = np.where((end_point==starts_left).all(axis=1))[0]

        if (len(next_idx_left) == 0):
            traces_all.append(np.array(trace))
            trace = []    
            new_start = starts_left[0]
            this_idx = np.where( (new_start==lines_in[:,0]).all(axis=1) )[0][0]
        else:
            starts_left.pop(next_idx_left[0])
            next_idx = np.where( (end_point==lines_in[:,0]).all(axis=1) )[0][0]
            this_idx = next_idx
        trace.append(this_idx)
            
    traces_all.append(np.array(trace))
    
    #Validate the traces to be closed perimeters
    traces_out = []
    for trace in traces_all:
        if (trace[0]==trace[-1]): 
            traces_out.append(trace[0:-1])
    return traces_out

def calculate_normals(vertices):
    """
    """
    normals = np.cross(vertices[:,1] - vertices[:,0] , vertices[:,2] - vertices[:,0])
    normals = normals / np.array([np.linalg.norm(normals, axis=1)]).T
    
    return normals

def validate_object(vertices):
    """
    """
    normals = np.cross(vertices[:,1] - vertices[:,0] , vertices[:,2] - vertices[:,0])
    invalid = (normals==0).all(axis=1)
    vertices = vertices[invalid==False]
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
    if s==0: s=1  
        
    kmat = np.array([[0, -v[2], v[1]],  [v[2], 0, -v[0]],  [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def triangulate_polygon(points):
    """
    
    """
    LR_ext = LinearRing(points[0])
    polygon = Polygon(LR_ext, points[1:])   
    ## TODO: Make Triangulation Edge Constrained 
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
            tri_Poly_within.extend([orient(tri, sign=1) for tri in Poly_outside if tri.within(polygon)])
            
    vert =  np.array([np.array(tri.exterior.xy)[:,0:3].T for tri in tri_Poly_within])
    
    all_points = np.concatenate(points)
    zval = np.full((*vert.shape[0:2],1), all_points[0,2] )  
    vertices_tri = np.concatenate((vert, zval ), axis =2)
    
    return vertices_tri

def get_open_edges(vertices):
    
    ## Make vertex list
    vertices_list = np.reshape(vertices, (vertices.shape[0]*3,3))
    uni_vertices, vert_idx = np.unique(vertices_list, axis=0, return_inverse=True)
    vert_idx = np.reshape(vert_idx, (vertices.shape[0],3))
        
    ## Make Edges list
    edges = np.array([[[v[0],v[1]],[v[1],v[2]],[v[2],v[0]]] for v in vert_idx ])
    edge_list = np.reshape(edges, (edges.shape[0]*3, 2))
    
    edge_sorted = np.sort(edge_list,axis=1)
    _, edge_idx, edge_counts = np.unique(edge_sorted, axis=0, return_counts=True, return_inverse=True)
    
    edge_counts_exp = edge_counts[edge_idx]
    open_edge_id = edge_list[edge_counts_exp==1]
    open_edges = np.array(uni_vertices[open_edge_id])
    
    return open_edges 

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
