from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .tools import *
from .polygon import *
from .save import *

from collections import defaultdict


class Solid:

    def __init__(self, triangles):

        # Accept either a tuple (vertices, faces) or raw triangle facets
        if triangles is None:
            vertices = np.zeros((0, 3))
            faces = np.zeros((0, 3), dtype=int)
        elif isinstance(triangles, (list, tuple)) and len(triangles) == 2:
            vertices, faces = triangles
        else:
            vertices, faces = vertices_to_index(triangles)

        self.vertices = vertices
        self.faces = faces
        # self.surfaces = surfaces
        # self.normals = normals

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
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    return normals


def vertices_to_index(triangles):
    """
    Optimized vertex indexing using the void-view trick.
    Reduces 100% bottleneck in np.unique(axis=0).
    """
    # 1. Flatten and Round
    v_flat = triangles.reshape(-1, 3)
    v_rounded = np.round(v_flat, 7)

    # 2. View as a 1D array of 24-byte blocks
    # This makes np.unique much faster than axis=0
    v_view = v_rounded.view(np.dtype((np.void, v_rounded.dtype.itemsize * 3)))

    # 3. Perform the unique operation
    _, unique_indices, inverse_indices = np.unique(
        v_view,
        return_index=True,
        return_inverse=True
    )

    # 4. Reconstruct the indexed mesh
    uni_vertices = v_rounded[unique_indices]
    vertices_idx = inverse_indices.reshape(-1, 3)

    return uni_vertices, vertices_idx


def get_face_area(triangles):
    """
    """
    normals = np.cross(triangles[1] - triangles[0],
                       triangles[2] - triangles[0])
    area = np.linalg.norm(normals) / 2

    return area


def get_surfaces(vertices, faces, normals=None):
    """ 
        vertices is a M x 3 x 3 array of M triangles in 3D, 
        normals are [x,y,z] float normal vertices 
    """
    # Todo: use shared edges not vertices to make surface

    triangles = vertices[faces]
    if normals is None:
        normals = calculate_normals(triangles)

    norm_Dict = normal_to_dict(normals)
    _, edge_idx = index_edges(faces)

    surfaces = []
    for n_id in norm_Dict:

        face_id = norm_Dict[n_id]

        if len(face_id) == 1:
            surfaces.append(face_id)
            continue

        sub_edges = edge_idx[face_id]
        surf = contiguous_edges(sub_edges, face_id)
        surfaces.extend(surf)

    surface_normals = np.array([normals[surf[0]] for surf in surfaces])
    # print([len(s) for s in surfaces], surface_normals)

    return surfaces, surface_normals


def normal_to_dict(normals, decimals=3):
    """
    Groups face indices by their normal vectors.
    decimals: number of decimal places to round to (handles float jitter).
    """
    # 1. Round to handle precision errors (e.g., 0.99999999 vs 1.0)
    rounded_normals = np.round(normals, decimals=decimals)

    # 2. Find unique rows and the mapping (inverse)
    # axis=0 treats each [x, y, z] as a single unit
    unique_norms, inverse, counts = np.unique(
        rounded_normals,
        axis=0,
        return_inverse=True,
        return_counts=True
    )

    # 3. Sort face indices so those sharing the same normal are adjacent
    idx_sort = np.argsort(inverse)

    # 4. Split the sorted indices into groups based on counts
    split_indices = np.cumsum(counts)[:-1]
    groups = np.split(idx_sort, split_indices)

    # Return a dictionary where keys are the actual (rounded) normal tuple
    return {tuple(norm): group.tolist() for norm, group in zip(unique_norms, groups)}


def edges_to_dict(edge_idx):

    edge_dict = defaultdict(list)
    for f_idx, tri in enumerate(edge_idx):
        for e in tri:
            edge_dict[e].append(f_idx)

    return edge_dict


def contiguous_edges(edge_idx, idx_list):
    """
    Finds contiguous surfaces within a subset of faces.
    edge_idx: (N, 3) vertex indices for the subset of faces
    idx_list: (N,) the original global face IDs
    """
    # Safety check for empty input
    if edge_idx.shape[0] == 0:
        return []
    if edge_idx.shape[0] == 1:
        return [np.array(idx_list)]

    num_faces = edge_idx.shape[0]

    # 1. Flatten the data for the matrix
    # Map which sub-face uses which vertex/edge
    flat_edges = edge_idx.ravel()
    # Create local indices [0, 0, 0, 1, 1, 1...]
    face_indices = np.repeat(np.arange(num_faces), edge_idx.shape[1])

    # 2. Build the B matrix (Edges x Faces)
    # This maps which faces share the same edge/vertex value
    B = csr_matrix((np.ones_like(flat_edges), (flat_edges, face_indices)))

    # 3. Create the Adjacency Matrix (Faces x Faces)
    # The dot product finds faces that share at least one value in B
    adj_matrix = B.T @ B

    # 4. Find connected components
    # This replaces the manual queue/while loop logic
    n_components, labels = connected_components(
        csgraph=adj_matrix, directed=False)

    # 5. Group the original idx_list by the labels found
    # We must convert idx_list to a numpy array to use advanced indexing
    idx_list = np.asarray(idx_list)
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]
    sorted_idx_list = idx_list[sort_idx]

    # Split the sorted list whenever the label changes
    diff = np.where(np.diff(sorted_labels))[0] + 1
    surfaces = np.split(sorted_idx_list, diff)

    return surfaces


def contiguous_edges_old(edge_idx, idx_list):

    # Old Slow Original Method
    edge_dict = edges_to_dict(edge_idx)

    b_unchecked = np.ones(edge_idx.shape[0], dtype=bool)
    surfaces, surf, queue = [], [], []

    sub_idx = 0
    b_unchecked[sub_idx] = False
    idx = idx_list[sub_idx]
    surf.append(idx)
    for _ in range(edge_idx.shape[0]-1):

        edges = edge_idx[sub_idx]
        for e in edges:
            for idx in edge_dict[e]:
                if idx != sub_idx and b_unchecked[idx]:

                    queue.append(idx)
                    surf.append(idx_list[idx])
                    b_unchecked[idx] = False

        # Continue treversing touching vertices or in queue
        if len(queue) > 0:
            sub_idx = queue.pop()

        # If queue is empty start a new Surface on same normal plane
        else:
            surfaces.append(np.array(surf))
            surf = []

            sub_idx = np.where(b_unchecked)[0][0]

            surf.append(idx_list[sub_idx])
            b_unchecked[sub_idx] = False

    surfaces.append(np.array(surf))
    return surfaces


def simplify_object_3D(solid):
    """
    """
    vertices = solid.vertices
    normals = solid.normals
    # surfaces = solid.surfaces
    # faces = solid.faces

    perimeter_list = get_perimeter_list(solid)
    perimeter_list = get_min_required_vertices(
        vertices, perimeter_list, normals)

    faces_out = []

    for n, peri in enumerate(perimeter_list):

        if len(peri) == 1 and len(peri[0]) == 3:  # Perimeter is a triangle
            new_faces = [peri[0]]

        elif len(peri) == 1 and len(peri[0]) == 4:  # Perimeter is a quad
            new_faces = [peri[0][[0, 1, 2]], peri[0][[0, 2, 3]]]

        else:
            norm = normals[n]
            peri = [p for p in peri]
            _, new_faces = simplify_surface(vertices, peri, norm, )

        faces_out.append(new_faces)
    facets_out = vertices[np.concatenate(faces_out)]

    solid = Solid(facets_out)

    return solid


def simplify_surface(vertices, perimeters, normal=None):
    """
    """
    # Flatten boundry to 2D
    if normal is None:
        normal = np.array([0, 0, 1])

    sub_verts = vertices[np.concatenate(perimeters)]
    sub_peri = []
    end = 0
    for p in perimeters:
        sub_peri.append(np.arange(len(p))+end)
        end += len(p)

    if sub_verts.shape[1] == 2:
        sub_verts_2D = sub_verts
    elif sub_verts.shape[1] == 3:
        sub_verts_2D = rotate_3D(sub_verts, normal)

    _, sub_faces = triangulate_polygon(sub_verts_2D, sub_peri)
    faces = np.concatenate(perimeters)[sub_faces]

    return sub_verts, faces


def get_perimeter_list(solid):

    vertices = solid.vertices
    faces = solid.faces
    surfaces = solid.surfaces

    perimeter_list = []
    for surf in surfaces:

        surf_faces = faces[surf]
        if len(surf) < 2:

            perimeter_list.append([surf_faces[0]])
            continue

        edges = get_open_edges(surf_faces)
        perimeters = get_ordered_perimeter(vertices, edges)

        perimeter_list.append(perimeters)

    return perimeter_list


def get_min_required_vertices(vertices, perimeter_list, normal_list):

    simplified_vertices = []

    for n, perimeters in enumerate(perimeter_list):

        norm = normal_list[n]
        simplified_perimeters = simplify_perimeters(vertices, perimeters, norm)

        if len(simplified_perimeters) == 0:
            continue

        simplified_vertices.append(np.concatenate(simplified_perimeters))

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

    # Check for invalid triangles
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    invalid = (normals == 0).all(axis=1)
    triangles = triangles[invalid == False]

    if np.sum(invalid) > 0:
        print("invalid faces exist in object!!")

    # Check for invalid edges
    open_edges = get_open_edges(faces)
    if len(open_edges) > 0:

        is_valid = False
        print(list(open_edges))
        print(list(vertices[open_edges]))
        print("Open edges exist in object!!")

    if (is_valid) == False:
        print("Solid is not valid")


def index_edges(faces):
    # 1. Extract edges from faces (each face has 3 edges)
    # faces is (M, 3) where each row is [v1, v2, v3]
    # We create pairs: [v1,v2], [v2,v3], [v3,v1]
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    edge_list = edges.reshape(-1, 2)

    # 2. Pack the vertex ID pairs into a single 64-bit integer
    # This assumes your vertex indices fit in 32-bit integers
    packed_edges = edge_list[:, 0].astype(np.int64) << 32 | edge_list[:, 1]

    # 3. Find unique edges using 1D logic (extremely fast)
    # return_inverse gives us the ID of each edge
    _, edge_inverse = np.unique(packed_edges, return_inverse=True)

    # 4. Reshape back to (M, 3) to map each triangle to its 3 edge IDs
    edge_idx = edge_inverse.reshape(-1, 3)

    return _, edge_idx


def index_edges_old(vertices):

    # Make vertex list
    _, vert_idx = vertices_to_index(vertices)
    # Make Edges list
    edges = vert_idx[:, [[0, 1], [1, 2], [2, 0]]]
    edge_list = np.reshape(edges, (-1, 2))
    # Count edge occurance
    edge_sorted = np.sort(edge_list, axis=1)
    uni_edges, edge_idx = np.unique(edge_sorted, axis=0, return_inverse=True)

    edge_idx = np.reshape(edge_idx, (-1, 3))

    return uni_edges, edge_idx


def get_open_edges_old(faces):

    # Make Edges list
    edges = faces[:, [[0, 1], [1, 2], [2, 0]]]
    edge_list = np.reshape(edges, (edges.shape[0]*3, 2))
    # Count edge occurance
    edge_sorted = np.sort(edge_list, axis=1).astype(np.int64)
    _, edge_idx, edge_counts = np.unique(
        edge_sorted, axis=0, return_counts=True, return_inverse=True)

    edge_counts_exp = edge_counts[edge_idx]
    # Select edges that only occur once and thus open
    open_edges = edge_list[edge_counts_exp == 1]
    # Reconstruct Edge positions from locations

    return open_edges


def get_open_edges_old(faces):
    # 1. Create all edges (N*3, 2)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    # 2. Sort edges so [A, B] and [B, A] are identical
    # We do this manually to avoid np.sort overhead
    p1 = edges.min(axis=1)
    p2 = edges.max(axis=1)

    # 3. Hash the edges into a 1D array (int64)
    # This is the secret sauce. It makes the unique search 1D.
    # Max index is used to ensure no collisions.
    max_idx = faces.max() + 1
    edge_keys = p1 * max_idx + p2

    # 4. Replace np.unique with argsort (The Speedup)
    # We find where a key is NOT equal to its neighbors in a sorted list
    idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[idx]

    # Mask elements that are different from both their left and right neighbor
    mask = np.ones(len(sorted_keys), dtype=bool)
    mask[1:] &= (sorted_keys[1:] != sorted_keys[:-1])
    mask[:-1] &= (sorted_keys[:-1] != sorted_keys[1:])

    # 5. Map back to original edges using the sorted index
    return edges[idx[mask]]


def get_open_edges(faces):
    # 1. Create all directed edges (N*3, 2)
    # This keeps the [A -> B], [B -> C], [C -> A] flow
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    # 2. Create the "Symmetric Key" for finding unique pairs
    p1 = edges.min(axis=1)
    p2 = edges.max(axis=1)
    max_idx = faces.max() + 1
    edge_keys = p1.astype(np.int64) * max_idx + p2

    # 3. Sort and Mask (Your Speedup)
    idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[idx]

    mask = np.ones(len(sorted_keys), dtype=bool)
    mask[1:] &= (sorted_keys[1:] != sorted_keys[:-1])
    mask[:-1] &= (sorted_keys[:-1] != sorted_keys[1:])
    # 4. Map back to the ORIGINAL directed edges
    # This ensures if the face was [13, 11, 19], we return [13, 11]
    # NOT a sorted [11, 13].
    return edges[idx[mask]]


def triangles_to_facets(triangles):

    normals = calculate_normals(triangles)
    facets = np.array([np.concatenate([normals[n], v[0], v[1], v[2]])
                      for n, v in enumerate(triangles)])

    return facets
