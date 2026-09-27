from collections import defaultdict

import numpy as np
import triangle as tr


def get_ordered_perimeter(vertices, edges, validate=False):

    edge_graph = defaultdict(list)
    for e in edges:
        edge_graph[e[0]].append(e[1])

    this_idx = list(edge_graph.keys())[0]
    traces_all, trace = [], []
    for _ in range(len(edges)):

        trace.append(this_idx)
        next_idxs = np.array(edge_graph[this_idx])

        if len(next_idxs) > 1 and len(trace) > 1:
            smallest_idx = find_smaller_angle(
                vertices[trace[-2]], vertices[trace[-1]], vertices[next_idxs]
            )
            next_idx = next_idxs[smallest_idx]

        elif len(next_idxs) > 0:
            next_idx = next_idxs[0]

        if next_idx in edge_graph[this_idx]:
            edge_graph[this_idx].remove(next_idx)

        if len(edge_graph[this_idx]) == 0:
            edge_graph.pop(this_idx)

        this_idx = next_idx

        if this_idx in trace:
            trace.append(this_idx)
            traces_all.append(trace)
            trace = []

            if len(edge_graph) > 0:
                this_idx = list(edge_graph.keys())[0]

    # Validate the traces to be closed perimeters
    traces_out = []
    for trace in traces_all:
        if trace[0] != trace[-1]:
            continue
        trace = trace[0:-1]
        if len(trace) < 3:
            continue
        traces_out.append(np.array(trace))

    return traces_out


def simplify_perimeters(vertices, perimeters, normal):

    simplified_perimeters = []
    for line_idx in perimeters:

        if len(line_idx) < 5:
            simpified_line = line_idx
        else:
            line = vertices[line_idx]
            line_2d = rotate_3D(line, normal, [0, 0, 1])
            angles = get_perimeter_angles(line_2d)
            simpified_line = np.array(line_idx[angles != 180])

        simplified_perimeters.append(simpified_line)

    return simplified_perimeters


def perimeters_to_edges(perimeters):
    edges = [np.stack([p, np.roll(p, 1, axis=0)], axis=1) for p in perimeters]
    return edges


def perimeter_to_2D(perimeters, normal, simplify_lines=False):
    """Rotate 3-D perimeters so ``normal`` points along +z.

    With ``simplify_lines`` the collinear (180 degree) points are dropped from
    each rotated perimeter.
    """
    perimeter_2d = [rotate_3D(lines, normal, [0, 0, 1]) for lines in perimeters]

    if simplify_lines:
        perimeter_2d = [simplify_line(line) for line in perimeter_2d]

    return perimeter_2d


##########################################################################################
##########################################################################################


def get_perimeter_normal(perimeter):
    """Unit normal of a planar 3-D perimeter, from the first non-degenerate corner."""
    n = 0
    normal = np.cross(perimeter[n + 1] - perimeter[n], perimeter[n - 1] - perimeter[n])
    while (np.linalg.norm(normal) == 0) and (n < len(perimeter) - 1):
        normal = np.cross(perimeter[n + 1] - perimeter[n], perimeter[n - 1] - perimeter[n])
        n = n + 1

    normal = normal / np.linalg.norm(normal)

    return normal


def get_area(perimeter):
    """Area of a planar 3-D perimeter, negative when it winds clockwise.

    The perimeter is first rotated onto its own plane (see
    ``get_perimeter_normal``), then measured with the shoelace formula.
    """
    normal = get_perimeter_normal(perimeter)
    perimeter = rotate_3D(perimeter, normal, [0, 0, 1])

    x, y = perimeter[:, 0], perimeter[:, 1]

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    orient = get_orientation(perimeter)
    if orient[0] < 0:
        area = -area

    return area


def get_orientation(perimeter):
    """Twice the signed xy-area of a perimeter, in a one-element list.

    Positive for counter-clockwise (seen from +z), negative for clockwise.
    """
    edges = perimeters_to_edges([perimeter])
    orientation = [
        np.sum(np.diff(e[:, :, 0], n=1, axis=1) * np.sum(e[:, :, 1], axis=1)[:, None])
        for e in edges
    ]
    return orientation


def set_orientation(perimeter, orientation=1):
    """Reverse the perimeters in a list so each winds CCW (``orientation > 0``) or CW."""
    perimeter_out = []
    for p in perimeter:
        result = get_orientation(p)[0]
        if (orientation < 0 and result > 0) or (orientation > 0 and result < 0):
            p = p[::-1]
        perimeter_out.append(p)

    return perimeter_out


##########################################################################################
##########################################################################################


def get_perimeter_angles(line_2D):

    if line_2D.shape[1] == 3:
        line_2D = line_2D[:, 0:2]

    line_wrapped = np.concatenate([[line_2D[-1]], line_2D, [line_2D[0]]])
    ba = line_wrapped[0:-2] - line_wrapped[1:-1]
    bc = line_wrapped[2::] - line_wrapped[1:-1]
    angles = get_angle_vectors(bc, ba)

    return angles


def simplify_line(line_2D):
    """Drop the points of a closed 2-D line where it runs straight (angle 180)."""
    angles = get_perimeter_angles(line_2D)
    simpified_line = np.array(line_2D[angles != 180])

    return simpified_line


def get_angle_vectors(ba, bc):

    ba = ba / np.array(np.linalg.norm(ba, axis=1))[:, None]
    bc = bc / np.array(np.linalg.norm(bc, axis=1))[:, None]
    dot_prod = np.sum(ba * bc, axis=1)
    # z of the 2-D cross product (np.cross rejects 2-D vectors since NumPy 2.0)
    cross_prod = ba[:, 0] * bc[:, 1] - ba[:, 1] * bc[:, 0]

    angle = np.arctan2(cross_prod, dot_prod)
    angle = np.degrees(angle)
    angle[angle < 0] += 360

    return angle


def find_smaller_angle(last_pt, this_pt, next_pts):
    bc = [last_pt - this_pt]
    ba = next_pts - this_pt
    ## Return dot product
    ba = ba / np.linalg.norm(ba, axis=1)[:, None]
    bc = bc / np.linalg.norm(bc, axis=1)[:, None]
    dot_prod = np.sum(ba * bc, axis=1)

    ## Self angles should be last considered (if considered at all)
    dot_prod[np.isclose(dot_prod, 1)] = -1
    idx = np.argsort(dot_prod)
    return idx[-1]


##########################################################################################
############################################################################################


def rotate_3D(pts, src_vertex, des_vertex=[0, 0, 1]):
    """ """
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
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    if s == 0 and c == 1:
        rotation_matrix = np.eye(3)
    elif s == 0 and c == -1:
        # 180° rotation: find a perpendicular axis and rotate around it
        perp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - np.dot(perp, a) * a
        perp = perp / np.linalg.norm(perp)
        rotation_matrix = 2 * np.outer(perp, perp) - np.eye(3)
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

    return rotation_matrix


#################################################################################
#################################################################################


def triangulate_polygon(vertices_2D, perimeters):

    grouped_edges = perimeters_to_edges(perimeters)

    holes = []
    if len(grouped_edges) > 1:
        for i in range(1, len(grouped_edges)):

            verts, faces = triangulate_edges(vertices_2D, grouped_edges[i])
            tris = verts[faces]
            cents = tris.mean(axis=1)
            holes.append(cents[0])
    else:
        holes = None

    all_edges = np.concatenate(grouped_edges)
    vertices, faces = triangulate_edges(vertices_2D, all_edges, holes=holes)

    return vertices, faces


def triangulate_edges(vertices_2D, edges, holes=None):

    if holes is None:
        shape = {"vertices": vertices_2D[:, [0, 1]], "segments": edges}
    else:
        shape = {"vertices": vertices_2D[:, [0, 1]], "segments": edges, "holes": holes}
    t = tr.triangulate(shape, "p")

    vertices = t["vertices"]
    faces = t["triangles"]

    sub_faces = np.unique(faces.reshape(1, -1), return_inverse=True)[1].reshape(-1, 3)

    vertices = vertices[edges[:, 0]]
    faces = np.argsort(edges[:, 0])[sub_faces]

    return vertices, faces
