
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
    top_facets = top_facet_from_array(A, mask_val = mask_val)

    if solid:

        vertices = np.array([[facet[3:6],facet[6:9],facet[9:12]] for facet in top_facets])

        ## Walls
        print("Creating walls...")
        edges = get_open_edges(vertices)
        perimeter = get_ordered_perimeter(edges)

        wall_verts = perimeter_to_wall_vertices(perimeter, floor_val=min_val)
        wall_facets = vertices_to_facets(wall_verts)

        ##Bottom 
        print("Creating bottom...")

        bottom_facets = []
        for facet in top_facets:
            this_bottom = np.concatenate( [facet[:3], facet[6:8], [min_val], facet[3:5], [min_val], facet[9:11], [min_val]])
            bottom_facets.append(this_bottom)

            
        facets = np.concatenate([top_facets, wall_facets, bottom_facets])
    
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

def perimeter_to_wall_vertices(perimeters, floor_val=0):
    """
    
    """ 
    wall_vertices = []

    for peri in perimeters: 

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

def vertices_to_facets(vertices):

    normals = calculate_normals(vertices)
    facets = np.array([np.concatenate([normals[n],v[0],v[1],v[2]]) for n,v in enumerate(vertices)])

    return facets