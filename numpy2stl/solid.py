
from .tools import *

class Solid:

    def __init__(self, facets):

        vertices, faces = vertices_to_index(facets)
        surfaces, normals = get_surfaces(facets)

        self.vertices = vertices
        self.faces = faces
        self.surfaces = surfaces
        self.normals = normals 

        
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
    
    uni_normals , normals_idx = np.unique(normals.round(decimals=9), axis=0 , return_inverse=True)
    _, edge_idx = index_edges(triangles)

    surfaces = []
    surf_normals = []
    # for each set of coplanear facets 
    for n_idx in range(len(uni_normals)):

        same_normal = normals_idx == n_idx   
        idx_list = np.nonzero(normals_idx == n_idx)[0]

        if len(idx_list)==1:
            surfaces.append(idx_list)
            continue
        
        sub_edges = edge_idx[same_normal] 
        surf = contiguous_edges(sub_edges,idx_list)

        surfaces.extend(surf)

    surface_normals = np.array([normals[surf[0]] for surf in surfaces])
                
    return surfaces, surface_normals

def contiguous_edges(edge_idx, idx_list):

    b_checked = np.zeros(edge_idx.shape[0],dtype=bool)  
    surfaces, surf, queue = [],[],[]

    sub_idx = 0
    b_checked[sub_idx] = True
    idx = idx_list[sub_idx]
    surf.append(idx) 

    edge_dict = defaultdict(list)
    for f_idx, tri in enumerate(edge_idx):
        for e in tri:
            edge_dict[e].append( f_idx )

    for _ in range(edge_idx.shape[0]-1):

        edges  = edge_idx[ sub_idx ]
        for e in edges:
            for idx in edge_dict[e]:
                if idx != sub_idx and b_checked[idx]==False:
                    
                    queue.append(idx)
                    surf.append(idx_list[idx])   
                    b_checked[idx] = True

        #Continue treversing touching vertices or in queue
        if len(queue)>0:
            sub_idx = queue.pop()
            
        #If queue is empty start a new Surface on same normal plane   
        else:
            surfaces.append(np.array(surf))
            surf = []
            sub_idx = np.where(b_checked==False)[0][0]  
            surf.append( idx_list[sub_idx] )
            b_checked[sub_idx] = True
                 
    surfaces.append(np.array(surf))

    return surfaces

def simplify_object_3D(solid):
    """
    """
    normals = solid.normals
    perimeter_list = get_perimeter_list(solid)

    min_vertices = get_min_required_vertices( perimeter_list, normals)

    vertices_out = []
    for n, peri in enumerate(perimeter_list):
        
        if len(peri) == 1 and len(peri[0])==3:
            surface_vertices = peri[0][None,:]
        else:
            norm = normals[n]

            peri = [p[(p[:,None] == [min_vertices]).all(axis=2).any(axis=1)] for p in peri]

            surface_vertices = simplify_surface( peri, norm, ) 

        vertices_out.append( surface_vertices )  
    vertices_out = np.concatenate(vertices_out)

    solid = Solid(vertices_out)

    #solid = validate_object(solid)    
    return solid


def simplify_surface(perimeters, normal ):
    """
    """    
    ## Flatten boundry to 2D
    perimeter_2d = perimeter_to_2D( perimeters, normal, simplify_lines=False)
    
    _,faces = triangulate_polygon( perimeter_2d )

    simplifed_vert = np.concatenate(perimeters)[faces]

    #sum1 = np.sum(np.array([get_area(p) if n==0 else -get_area(p) for n,p in enumerate(perimeters)]))
    #sum2 = np.sum(np.array([get_area(v) for v in simplifed_vert]))

    #if np.isclose(sum1, sum2) == False:
        
    #    print(  '1', sum1 )
    #    print(  '2', sum2 )
        
    return simplifed_vert

def get_perimeter_list(solid):

    vertices = solid.vertices
    faces = solid.faces
    surfaces = solid.surfaces
  
    perimeter_list = []
    for surf in surfaces:

        surface_vertices = vertices[faces[surf]]
        
        if len(surface_vertices)==1 and len(surface_vertices[0]):
            perimeter_list.append([surface_vertices[0]])  
            continue

        edges = get_open_edges(surface_vertices)
        perimeters = get_ordered_perimeter(edges)
        
        perimeter_list.append(perimeters)   

    return perimeter_list


def get_min_required_vertices( perimeter_list, normal_list):

    simplified_vertices = []  

    for n, perimeters in enumerate(perimeter_list):

        norm = normal_list[n]
        simplified_perimeters = simplify_perimeters( perimeters, norm)
        if len(simplified_perimeters)==0:          continue
            
        simplified_vertices.append(  np.concatenate( simplified_perimeters )  )

    required_vertices = np.concatenate(simplified_vertices)  
    required_vertices = np.unique(required_vertices, axis=0)

    return required_vertices

    
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
    open_edges = get_open_edges(triangles)
    if len(open_edges) > 0:

        is_valid = False
        print(open_edges)
        print("Open edges exist in object!!")

    return is_valid

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

def get_open_edges(vertices):
    
    ## Make vertex list
    uni_vertices, vert_idx = vertices_to_index(vertices)
    ## Make Edges list
    edges = vert_idx[:,[[0,1],[1,2],[2,0]]]
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


    