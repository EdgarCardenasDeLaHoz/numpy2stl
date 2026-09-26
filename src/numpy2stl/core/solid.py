from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..io.writers import writeSTL
from .polygon import (
    get_ordered_perimeter,
    rotate_3D,
    simplify_perimeters,
    triangulate_polygon,
)

__all__ = [
    "Solid",
    "calculate_normals",
    "vertices_to_index",
    "get_face_area",
    "get_surfaces",
    "triangles_to_facets",
    "get_open_edges",
    "validate_object",
]


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

    def validate_object(self):
        validate_object(self)

    def simplify(self):
        solid = simplify_object_3D(self)
        validate_object(solid)
        return solid

    def save_stl(self, filename, ascii=False):
        """Save the solid mesh to an STL file.

        Parameters
        ----------
        filename : str
            Output STL filename
        ascii : bool, optional
            If True, write ASCII STL format. Default is False (binary).

        Examples
        --------
        >>> import numpy as np
        >>> from numpy2stl import Solid
        >>> vertices = np.array([[0,0,0], [1,0,0], [0.5,1,0]])
        >>> faces = np.array([[0,1,2]])
        >>> solid = Solid((vertices, faces))
        >>> solid.save_stl("triangle.stl")
        """

        triangles = self.vertices[self.faces]
        facets = triangles_to_facets(triangles)
        writeSTL(facets, filename, ascii=ascii)


def calculate_normals(triangles):
    """ """
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    return normals


def vertices_to_index(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimized vertex indexing using the void-view trick.
    Reduces 100% bottleneck in np.unique(axis=0).

    Parameters
    ----------
    triangles : ndarray, shape (N, 3, 3)
        Array of triangle vertices

    Returns
    -------
    vertices : ndarray, shape (M, 3)
        Unique vertices
    faces : ndarray of int, shape (N, 3)
        Face indices into vertices array

    Examples
    --------
    >>> triangles = np.array([[[0,0,0], [1,0,0], [0,1,0]],
    ...                       [[1,0,0], [1,1,0], [0,1,0]]])
    >>> vertices, faces = vertices_to_index(triangles)
    >>> len(vertices)  # 4 unique vertices
    4
    """
    # 1. Flatten and Round
    v_flat = triangles.reshape(-1, 3)
    v_rounded = np.round(v_flat, 6)

    # 2. View as a 1D array of 24-byte blocks
    v_view = v_rounded.view(np.dtype((np.void, v_rounded.dtype.itemsize * 3)))

    # 3. Perform the unique operation
    _, unique_indices, inverse_indices = np.unique(v_view, return_index=True, return_inverse=True)

    # 4. Reconstruct the indexed mesh
    uni_vertices = v_rounded[unique_indices]
    vertices_idx = inverse_indices.reshape(-1, 3)

    return uni_vertices, vertices_idx


def get_face_area(triangles):
    """ """
    normals = np.cross(triangles[1] - triangles[0], triangles[2] - triangles[0])
    area = np.linalg.norm(normals) / 2

    return area


def get_surfaces(vertices, faces, normals=None):
    """
    vertices is a M x 3 x 3 array of M triangles in 3D,
    normals are [x,y,z] float normal vertices
    """
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

    return surfaces, surface_normals


def normal_to_dict(normals, decimals=3):
    """
    Groups face indices by their normal vectors.
    decimals: number of decimal places to round to (handles float jitter).
    """
    rounded_normals = np.round(normals, decimals=decimals)

    unique_norms, inverse, counts = np.unique(
        rounded_normals, axis=0, return_inverse=True, return_counts=True
    )

    idx_sort = np.argsort(inverse)

    split_indices = np.cumsum(counts)[:-1]
    groups = np.split(idx_sort, split_indices)

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
    if edge_idx.shape[0] == 0:
        return []
    if edge_idx.shape[0] == 1:
        return [np.array(idx_list)]

    num_faces = edge_idx.shape[0]

    flat_edges = edge_idx.ravel()
    face_indices = np.repeat(np.arange(num_faces), edge_idx.shape[1])

    B = csr_matrix((np.ones_like(flat_edges), (flat_edges, face_indices)))

    adj_matrix = B.T @ B

    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)

    idx_list = np.asarray(idx_list)
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]
    sorted_idx_list = idx_list[sort_idx]

    diff = np.where(np.diff(sorted_labels))[0] + 1
    surfaces = np.split(sorted_idx_list, diff)

    return surfaces


def simplify_object_3D(solid):
    """ """
    vertices = solid.vertices
    normals = solid.normals

    perimeter_list = get_perimeter_list(solid)
    perimeter_list = get_min_required_vertices(vertices, perimeter_list, normals)

    faces_out = []

    for n, peri in enumerate(perimeter_list):

        if len(peri) == 1 and len(peri[0]) == 3:  # Perimeter is a triangle
            new_faces = [peri[0]]

        elif len(peri) == 1 and len(peri[0]) == 4:  # Perimeter is a quad
            new_faces = [peri[0][[0, 1, 2]], peri[0][[0, 2, 3]]]

        else:
            norm = normals[n]
            peri = [p for p in peri]
            _, new_faces = simplify_surface(
                vertices,
                peri,
                norm,
            )

        faces_out.append(new_faces)
    facets_out = vertices[np.concatenate(faces_out)]

    solid = Solid(facets_out)

    return solid


def simplify_surface(vertices, perimeters, normal=None):
    """ """
    # Flatten boundry to 2D
    if normal is None:
        normal = np.array([0, 0, 1])

    sub_verts = vertices[np.concatenate(perimeters)]
    sub_peri = []
    end = 0
    for p in perimeters:
        sub_peri.append(np.arange(len(p)) + end)
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

    req_vert_idx = [
        [[p for p in peri if p in req_v_idx] for peri in poly] for poly in perimeter_list
    ]

    return req_vert_idx


def validate_object(solid):
    """ """
    vertices = solid.vertices
    faces = solid.faces

    triangles = vertices[faces]

    is_valid = True

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    invalid = (normals == 0).all(axis=1)
    triangles = triangles[invalid == False]

    if np.sum(invalid) > 0:
        print("invalid faces exist in object!!")

    open_edges = get_open_edges(faces)
    if len(open_edges) > 0:

        is_valid = False
        print(list(open_edges))
        print(list(vertices[open_edges]))
        print("Open edges exist in object!!")

    if (is_valid) == False:
        print("Solid is not valid")


def index_edges(faces):
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]], axis=2)
    edge_list = edges.reshape(-1, 2)

    packed_edges = edge_list[:, 0].astype(np.int64) << 32 | edge_list[:, 1]

    _, edge_inverse = np.unique(packed_edges, return_inverse=True)

    edge_idx = edge_inverse.reshape(-1, 3)

    return _, edge_idx


def get_open_edges(faces):
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])

    p1 = edges.min(axis=1)
    p2 = edges.max(axis=1)
    max_idx = faces.max() + 1
    edge_keys = p1.astype(np.int64) * max_idx + p2

    idx = np.argsort(edge_keys)
    sorted_keys = edge_keys[idx]

    mask = np.ones(len(sorted_keys), dtype=bool)
    mask[1:] &= sorted_keys[1:] != sorted_keys[:-1]
    mask[:-1] &= sorted_keys[:-1] != sorted_keys[1:]

    return edges[idx[mask]]


def triangles_to_facets(triangles):

    normals = calculate_normals(triangles)
    facets = np.array(
        [np.concatenate([normals[n], v[0], v[1], v[2]]) for n, v in enumerate(triangles)]
    )

    return facets
