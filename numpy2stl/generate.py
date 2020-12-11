
import numpy as np
from itertools import product
from .tools import *
from .solid import *


############################# convert array to facet list ##########################################

def numpy2stl(A, mask_val=0, solid=True):
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
    min_val = mask_val
    
    print("Creating top...")
    top_vertices, top_faces = array2faces(A, mask_val=mask_val)
    top_triangles = top_vertices[top_faces]

    if solid:
        ## Walls
        print("Creating walls...")
        edges = get_open_edges(top_faces)
        perimeters = get_ordered_perimeter(top_vertices, edges )
        wall_triangles = perimeter_to_walls(top_vertices, perimeters, floor_val=min_val)
        
        ##Bottom 
        print("Creating bottom...")
        bottom_vertices = top_vertices.copy()
        bottom_vertices[:,2] = min_val

        _, bottom_faces = simplify_surface(bottom_vertices, perimeters)
        bottom_faces = bottom_faces[:,[1,0,2]]
        bottom_triangles = bottom_vertices[bottom_faces]
        
        all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])
    
    else:
        all_triangles = top_triangles

    return all_triangles

def array2faces__(A, mask_val=0):
    
    m, n = A.shape
    xv,yv = np.meshgrid(range(n),range(m))
    vertices = np.stack([xv.ravel(),yv.ravel(),A.ravel()]).T

    idxs = np.array(range(m*n)).reshape(m,n)

    faces = []

    masked = A > mask_val
    for i, k in product(range(m - 1), range(n - 1)):

        if ((masked[i, k]) and (masked[i, k+1]) and 
            (masked[i+1, k]) and (masked[i+1, k+1])):
            
            faces.append( [idxs[i, k], idxs[i, k+1], idxs[i+1, k+1] ] )
            faces.append( [idxs[i, k], idxs[i+1, k+1], idxs[i+1, k] ] )

    faces = np.array(faces)

    
    return vertices, faces

import scipy.ndimage as ndi

def array2faces(A, mask_val=0):
    
    m, n = A.shape
    xv,yv = np.meshgrid(range(n),range(m))
    vertices = np.stack([xv.ravel(),yv.ravel(),A.ravel()]).T

    idxs = np.array(range(m*n)).reshape(m,n)

    masked = A > mask_val

    tl = idxs[:-1,:-1].ravel()
    tr = idxs[:-1, 1:].ravel()
    bl = idxs[ 1:,:-1].ravel()
    br = idxs[ 1:, 1:].ravel()

    all_faces = np.vstack([tl,tr,bl,br])

    structure=np.array([[0,0,0],[0,1,1],[0,1,1]])
    masked = ndi.binary_dilation(masked, structure=structure)
    masked = masked[:-1,:-1]
            
    faces = all_faces[:,masked.ravel()]
    faces = faces[[0,1,3,0,3,2],:].T
    faces = faces.reshape(-1,3)

    return vertices, faces

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


def polygon_to_prism(vertices, perimeters=None, base_val=0):

    if perimeters is None:
        perimeters = [ np.arange(len(vertices)) ]

    wall_triangles = perimeter_to_walls(vertices, perimeters, floor_val=base_val)
    
    _, faces = simplify_surface(vertices, perimeters)

    bottom_vertices = vertices.copy()
    bottom_vertices[:,2] = base_val

    top_triangles = vertices[faces]
    bottom_triangles = bottom_vertices[faces]
    
    all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])

    return all_triangles


def perimeter_to_walls(vertices, perimeters, floor_val=0):
    """
    """ 
    wall_vertices = []

    for peri in perimeters: 
        peri = vertices[peri]
        peri_roll = np.roll(peri,1,axis=0)

        for n,_ in enumerate(peri):

            top_left = np.concatenate([  peri[n,0:2], [floor_val] ])
            top_right = np.concatenate([  peri_roll[n,0:2], [floor_val] ])

            bottom_left = np.array(  peri[n]  )
            bottom_right = np.array(  peri_roll[n] )

            vert = [top_right, top_left, bottom_right]
            wall_vertices.append(vert)

            vert = [bottom_right, top_left, bottom_left]
            wall_vertices.append(vert)

    wall_vertices = np.array(wall_vertices)
    return wall_vertices


def roll2d(image, shifts):

    return np.roll(np.roll(image, shifts[0], axis=0), shifts[1], axis=1)

