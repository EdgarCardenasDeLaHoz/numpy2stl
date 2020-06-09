
import numpy as np
from itertools import product
from .tools import *
from .solid import *


############################# convert array to facet list ##########################################

def numpy2stl(A, mask_val=None, solid=False):
    """
    Reads a numpy array, and list of facets

    Inputs:
     A (ndarray) - an 'm' by 'n' 2D numpy array

    Optional input:

     mask_val (float) - any element of the inputted array that is less than this value will not be included in the mesh.
     solid (bool): sets whether to create a solid geometry (with sides and a bottom) or not.
                                        
    Returns: vertices 
    """

    if mask_val is None:
        mask_val = A.min() - 1.

    min_val = 0        
    
    print("Creating top...")
    top_vertices, top_faces = triangles_from_array(A, mask_val=mask_val)
    top_triangles = top_vertices[top_faces]

    if solid:

        ## Walls
        print("Creating walls...")
        edges = get_open_edges(top_faces)
        
        perimeters = get_ordered_perimeter(top_vertices, edges )
        wall_triangles = perimeter_to_wall_vertices(top_vertices, perimeters, floor_val=min_val)
        
        ##Bottom 
        print("Creating bottom...")

        bottom_vertices = top_vertices
        bottom_vertices[:,2] = min_val

        _, bottom_faces = simplify_surface(bottom_vertices, perimeters)
        bottom_triangles = bottom_vertices[bottom_faces]
        
        all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])
    
    else:
        all_triangles = top_triangles

    facets = triangles_to_facets(all_triangles)   

    return facets

def array2faces(A, mask_val=0):
    
    m, n = A.shape
    xv,yv = np.meshgrid(range(m),range(n))
    vertices = np.stack([xv.ravel(),yv.ravel(),A.ravel()]).T

    idxs = np.array(range(m*n)).reshape(m,n)

    faces = []
    for i, k in product(range(m - 1), range(n - 1)):

        if ((A[i, k] > mask_val) and (A[i, k+1] > mask_val) and 
            (A[i+1, k] > mask_val) and (A[i+1, k+1]  > mask_val)):
            
            this_point = idxs[i, k]
            top_right =  idxs[i, k+1]
            bot_left  =  idxs[i+1, k]
            bot_right =  idxs[i+1, k+1]    

            faces.append( [top_right, this_point, bot_right] )
            faces.append( [bot_right, this_point, bot_left] )

    faces = np.array(faces)
    return vertices, faces


def triangles_from_array(A, mask_val=0, ):
    
    m, n = A.shape
    facets = []
    
    for i, k in product(range(m - 1), range(n - 1)):

        this_point = [i, k, A[i, k]]
        top_right =  [i, k + 1, A[i, k + 1]]
        bot_left  =  [i + 1, k , A[i + 1, k]]
        bot_right =  [i + 1, k + 1, A[i + 1, k + 1]]

        if ((this_point[-1] > mask_val) and (top_right[-1] > mask_val) and 
            (bot_right[-1] > mask_val) and (bot_left[-1] > mask_val)):

            facets.append( [top_right, this_point, bot_right] )
            facets.append( [bot_right, this_point, bot_left] )

    vertices = np.array(facets)
    vertices, faces = vertices_to_index(vertices)
    return vertices,faces


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

def perimeter_to_wall_vertices(vertices, perimeters, floor_val=0):
    """
    
    """ 
    wall_vertices = []

    for peri in perimeters: 

        peri = vertices[peri]

        peri_roll = np.roll(peri,1,axis=0)
        for n, point in enumerate(peri):

            top_left = np.concatenate([  point[0:2], [floor_val]  ])
            top_right = np.concatenate([  peri_roll[n,0:2], [floor_val] ])
            bottom_left = np.array(  point  )
            bottom_right = np.array(  peri_roll[n] )

            vert = [top_right, top_left, bottom_right]
            wall_vertices.append(vert)

            vert = [bottom_right, top_left, bottom_left]
            wall_vertices.append(vert)

    wall_vertices = np.array(wall_vertices)
    return wall_vertices


def roll2d(image, shifts):

    return np.roll(np.roll(image, shifts[0], axis=0), shifts[1], axis=1)

def triangles_to_facets(triangles):

    normals = calculate_normals(triangles)
    facets = np.array([np.concatenate([normals[n],v[0],v[1],v[2]]) for n,v in enumerate(triangles)])

    return facets