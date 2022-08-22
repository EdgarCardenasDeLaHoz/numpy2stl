
from .tools import *
from .polygon import *
from .save import *

from collections import defaultdict 


class Solid:

    def __init__(self, triangles):

        vertices, faces = vertices_to_index(triangles)
        surfaces, normals = get_surfaces(triangles)

        self.vertices = vertices
        self.faces = faces
        self.surfaces = surfaces
        self.normals = normals 

    def validate_object(self):
        validate_object(self)

    def simplify(self):
        solid = simplify_object_3D(self)
        validate_object(solid)  
        return solid

    def save_stl(self, filename, ascii=False):

        triangles = self.vertices[self.faces]
        facets = triangles_to_facets(triangles)
        writeSTL(facets, filename, ascii=ascii)


        
def calculate_normals(triangles):
    """
    """
    normals = np.cross(triangles[:,1] - triangles[:,0] , triangles[:,2] - triangles[:,0])
    normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    
    return normals

def vertices_to_index(triangles):
    """
    """
    vertices_list = np.reshape(triangles, (triangles.shape[0]*3,3))
    uni_vertices, vertices_idx = np.unique(vertices_list, axis=0 , return_inverse=True)
    vertices_idx = np.reshape(vertices_idx, (triangles.shape[0],3))
    
    return uni_vertices, vertices_idx

def get_face_area(triangles):
    """
    """
    normals = np.cross(triangles[1] - triangles[0] , triangles[2] - triangles[0])
    area =  np.linalg.norm(normals) / 2
        
    return area

def get_surfaces(triangles, normals=None):
    """ 
        vertices is a M x 3 x 3 array of M triangles in 3D, 
        normals are [x,y,z] float normal vertices 
    """
    ## Todo: use shared edges not vertices to make surface 

    if normals is None:
        normals = calculate_normals(triangles)

    norm_Dict = normal_to_dict(normals)
    _, edge_idx = index_edges(triangles)    

    surfaces = []
    for n_id in norm_Dict:

        face_id = norm_Dict[n_id]

        if len(face_id)==1:
            surfaces.append(face_id)
            continue
        
        sub_edges = edge_idx[face_id] 
        surf = contiguous_edges(sub_edges,face_id)

        surfaces.extend(surf)

    surface_normals = np.array([normals[surf[0]] for surf in surfaces])
                
    return surfaces, surface_normals

def normal_to_dict(normals):

    normals = normals.round(decimals=9)
    uni_normals, normals_idx = np.unique(normals, axis=0 , return_inverse=True)
    norm_Dict = defaultdict(list) 
    for face_id in range(len(normals_idx)): 
        n_id = normals_idx[face_id] 
        norm_Dict[n_id].append(face_id) 

    return norm_Dict

def edges_to_dict(edge_idx):

    edge_dict = defaultdict(list)
    for f_idx, tri in enumerate(edge_idx):
        for e in tri:
            edge_dict[e].append( f_idx )

    return edge_dict

def contiguous_edges(edge_idx, idx_list):

    edge_dict = edges_to_dict(edge_idx)

    b_unchecked = np.ones(edge_idx.shape[0],dtype=bool)  
    surfaces, surf, queue = [],[],[]

    sub_idx = 0
    b_unchecked[sub_idx] = False
    idx = idx_list[sub_idx]
    surf.append(idx) 
    for _ in range(edge_idx.shape[0]-1):

        edges  = edge_idx[ sub_idx ]
        for e in edges:
            for idx in edge_dict[e]:
                if idx != sub_idx and b_unchecked[idx]:
                    
                    queue.append(idx)
                    surf.append(idx_list[idx])   
                    b_unchecked[idx] = False

        #Continue treversing touching vertices or in queue
        if len(queue)>0:
            sub_idx = queue.pop()
            
        #If queue is empty start a new Surface on same normal plane   
        else:
            surfaces.append(np.array(surf))
            surf = []

            sub_idx = np.where(b_unchecked)[0][0]

            surf.append( idx_list[sub_idx] )
            b_unchecked[sub_idx] = False
                 
    surfaces.append(np.array(surf))
    return surfaces

def simplify_object_3D(solid):
    """
    """
    vertices = solid.vertices
    normals = solid.normals
    #surfaces = solid.surfaces
    #faces = solid.faces

    perimeter_list = get_perimeter_list(solid)
    perimeter_list = get_min_required_vertices( vertices, perimeter_list, normals)
        
    faces_out = []

    for n, peri in enumerate(perimeter_list):
        
        if len(peri) == 1 and len(peri[0])==3:
            new_faces = [peri[0]]

        #elif len(peri) == 1 and len(peri[0])==4:
        #    new_faces = faces[surfaces[n]]
           
        else:
            norm = normals[n]
            peri = [p for p in peri]
            _, new_faces = simplify_surface( vertices, peri, norm, ) 

        faces_out.append( new_faces )  
    facets_out = vertices[np.concatenate(faces_out)]

    solid = Solid(facets_out)
    

    return solid

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

    if sub_verts.shape[1]==2:
        sub_verts_2D = sub_verts
    elif  sub_verts.shape[1]==3:
        sub_verts_2D = rotate_3D(sub_verts, normal)

    _,sub_faces = triangulate_polygon( sub_verts_2D , sub_peri)
    faces = np.concatenate(perimeters)[sub_faces]

    return sub_verts, faces


def get_perimeter_list(solid):

    vertices = solid.vertices
    faces = solid.faces
    surfaces = solid.surfaces
  
    perimeter_list = []
    for surf in surfaces:

        surf_faces = faces[surf]
        if len(surf)<2:

            perimeter_list.append([surf_faces[0]]) 
            continue

        edges = get_open_edges(surf_faces)
        perimeters = get_ordered_perimeter(vertices, edges)
        
        perimeter_list.append(perimeters)

    return perimeter_list

def get_min_required_vertices( vertices, perimeter_list, normal_list):

    simplified_vertices = []
    
    for n, perimeters in enumerate(perimeter_list):

        norm = normal_list[n]
        simplified_perimeters = simplify_perimeters( vertices, perimeters, norm)

        if len(simplified_perimeters)==0:
            continue
            
        simplified_vertices.append(  np.concatenate( simplified_perimeters )  )

    req_v_idx = np.concatenate(simplified_vertices)  
    req_v_idx = set(np.unique(req_v_idx, axis=0))

    req_vert_idx = [[
    [p for p in peri if p in req_v_idx] 
    for peri in poly]  
    for poly in perimeter_list]

    return req_vert_idx
    
def validate_object(solid):
    """
    """
    vertices = solid.vertices
    faces = solid.faces

    triangles = vertices[faces]

    is_valid = True

    ## Check for invalid triangles 
    normals = np.cross(triangles[:,1] - triangles[:,0] , triangles[:,2] - triangles[:,0])
    invalid = (normals==0).all(axis=1)
    triangles = triangles[invalid==False]

    if np.sum(invalid)>0:
        print("invalid faces exist in object!!")

    ## Check for invalid edges 
    open_edges = get_open_edges(faces)
    if len(open_edges) > 0:

        is_valid = False
        print(list(open_edges))
        print(list(vertices[open_edges]))
        print("Open edges exist in object!!")

    if (is_valid)==False:
        print("Solid is not valid")

def index_edges(vertices):

    ## Make vertex list
    _, vert_idx = vertices_to_index(vertices)
    ## Make Edges list
    edges = vert_idx[:,[[0,1],[1,2],[2,0]]]
    edge_list = np.reshape(edges, (edges.shape[0]*3, 2))
    ## Count edge occurance 
    edge_sorted = np.sort(edge_list,axis=1)
    uni_edges, edge_idx = np.unique(edge_sorted, axis=0, return_inverse=True)

    edge_idx = np.reshape(edge_idx, (vertices.shape[0],3))

    return uni_edges, edge_idx

def get_open_edges(faces):
        
    ## Make Edges list
    edges = faces[:,[[0,1],[1,2],[2,0]]]
    edge_list = np.reshape(edges, (edges.shape[0]*3, 2))
    ## Count edge occurance 
    edge_sorted = np.sort(edge_list,axis=1)
    _, edge_idx, edge_counts = np.unique(edge_sorted, axis=0, return_counts=True, return_inverse=True)

    edge_counts_exp = edge_counts[edge_idx]
    ## Select edges that only occur once and thus open
    open_edges = edge_list[edge_counts_exp==1]
    ## Reconstruct Edge positions from locations
    
    return open_edges

def triangles_to_facets(triangles):

    normals = calculate_normals(triangles)
    facets = np.array([np.concatenate([normals[n],v[0],v[1],v[2]]) for n,v in enumerate(triangles)])

    return facets

    
