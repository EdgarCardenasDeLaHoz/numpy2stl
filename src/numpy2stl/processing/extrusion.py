import numpy as np
import trimesh
from shapely import constrained_delaunay_triangles
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import orient, polygonize, unary_union

from ..core.solid import get_open_edges, vertices_to_index


def make_hollow_prism_solid(pts, offset_dist=1, z0=0, z1=5):
    # 1. Get 2D Geometry
    outer_pts, edges, holes = make_hollow_cap(pts, offset_dist=offset_dist)

    # 2. Triangulate (The source of truth for all vertices)
    vx_2d, faces = robust_triangulate(outer_pts, edges, holes=holes)
    num_v = len(vx_2d)

    # 3. Create 3D Vertices (Stacked: Bottom then Top)
    v_bottom = np.column_stack([vx_2d, np.full(num_v, z0)])
    v_top = np.column_stack([vx_2d, np.full(num_v, z1)])

    top_triangles = v_top[faces]
    bottom_triangles = v_bottom[faces[:, [1, 0, 2]]]
    wall_triangles = prism_wall_vertices(v_top, faces, floor_val=z0)
    all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])

    vx, fs = vertices_to_index(all_triangles)
    fs = np.array([f for f in fs if len(set(f)) == len(f)])
    vx[:, [1, 0]] = vx[:, [0, 1]]

    return vx, fs


def make_hollow_cap(pts, offset_dist=2):
    # 1. Create the geometries
    outer_poly = Polygon(pts)
    # The internal offset (negative buffer)
    inner_poly = outer_poly.buffer(-offset_dist, join_style=2)

    # 2. Extract vertices and edges for the outer boundary
    outer_coords = np.array(outer_poly.exterior.coords)[:-1]
    # Create the edge list [ [0,1], [1,2]... ]
    outer_indices = np.arange(len(outer_coords))
    edges = np.stack([outer_indices, np.roll(outer_indices, -1)], axis=1)

    # 3. Extract hole coordinates (as a list of arrays)
    if inner_poly.is_empty:
        holes = None
    elif isinstance(inner_poly, MultiPolygon):
        holes = [np.array(p.exterior.coords) for p in inner_poly.geoms]
    else:
        holes = [np.array(inner_poly.exterior.coords)]

    return outer_coords, edges, holes


def prism_wall_vertices_optimized(pts, z0, z1, is_internal=False):
    walls = []
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]

        b1 = [p1[0], p1[1], z0]
        b2 = [p2[0], p2[1], z0]
        t1 = [p1[0], p1[1], z1]
        t2 = [p2[0], p2[1], z1]

        if is_internal:
            walls.append([t1, b1, b2])
            walls.append([t1, b2, t2])
        else:
            walls.append([t1, b2, b1])
            walls.append([t2, b2, t1])

    return np.array(walls)


def extrude_solid_polygon(pts, z0=0, z1=5):
    """
    Creates a watertight 3D manifold from 2D points using Trimesh's
    built-in extrusion which is more robust than manual wall building.
    """
    poly = Polygon(pts)

    height = z1 - z0
    mesh = trimesh.creation.extrude(poly, height)

    mesh.apply_translation([0, 0, z0])

    mesh.merge_vertices()
    mesh.fix_normals()

    return mesh.vertices, mesh.faces


def make_prism_solid(pts, z0=0, z1=5):

    vert = pts.copy()
    zdim = np.zeros((len(vert), 1)) + z1
    vert = np.concatenate([vert, zdim], axis=1)

    perimeters = [np.arange(len(vert))]
    edges = np.stack([perimeters, np.roll(perimeters, 1, axis=0)], axis=1)
    vert2, faces = robust_triangulate(vert, edges[0], holes=None)

    top_triangles = vert2[faces]

    bottom_vertices = vert2.copy()
    bottom_vertices[:, 2] = z0
    bottom_triangles = bottom_vertices[faces[:, [1, 0, 2]]]

    wall_triangles = prism_wall_vertices(vert2, faces, floor_val=z0)

    all_triangles = np.concatenate([top_triangles, wall_triangles, bottom_triangles])

    vx, fs = vertices_to_index(all_triangles)
    fs = np.array([f for f in fs if len(set(f)) == len(f)])
    vx[:, [1, 0]] = vx[:, [0, 1]]

    return vx, fs


def make_sloped_prism_solid(pts, z0=0, plane=None, z1=5):
    """Watertight prism whose TOP cap may be a slanted plane.

    Builds a flat prism with trimesh's robust polygon extrusion, then (for a
    sloped ``plane=(a, b, c)``) tilts the top-cap vertices to ``z = a*x+b*y+c``.
    This lets a roughly-planar but *tilted* roof be approximated by one
    sloped-top prism ("approximate surfaces as planes") rather than a staircase
    of flat layers.  ``plane=None`` ⇒ flat top at ``z1``.

    Returns ``(vertices, faces)``.  Falls back to a flat top if the plane would
    dip to/below ``z0`` or is non-finite (a bad fit).
    """
    import trimesh
    from shapely.geometry import Polygon as _ShPoly

    p2d = np.asarray(pts, dtype=np.float64)
    if p2d.shape[1] == 3:
        p2d = p2d[:, :2]
    poly = _ShPoly(p2d)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    if plane is not None:
        a, b, c = float(plane[0]), float(plane[1]), float(plane[2])
        coords = np.asarray(poly.exterior.coords)
        zc = a * coords[:, 0] + b * coords[:, 1] + c
        if not np.all(np.isfinite(zc)) or np.min(zc) <= z0:
            plane = None                       # bad fit → flat top
        else:
            height = float(np.max(zc))         # nominal flat height before tilt
    if plane is None:
        height = float(z1)

    try:
        mesh = trimesh.creation.extrude_polygon(poly, height=max(height - z0, 1e-6))
    except Exception:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if plane is not None:
        a, b, c = float(plane[0]), float(plane[1]), float(plane[2])
        top = verts[:, 2] > (max(height - z0, 1e-6) - 1e-6)   # extruded top cap
        verts[top, 2] = a * verts[top, 0] + b * verts[top, 1] + c - z0
    verts[:, 2] += z0
    return verts, faces


def prism_wall_vertices(vertices_top, faces, floor_val=0):
    open_edges = get_open_edges(faces)
    wall_vertices = []

    for edge in open_edges:
        i1, i2 = edge

        v_top_1 = vertices_top[i1]
        v_top_2 = vertices_top[i2]

        v_bot_1 = np.array([v_top_1[0], v_top_1[1], floor_val])
        v_bot_2 = np.array([v_top_2[0], v_top_2[1], floor_val])

        wall_vertices.append([v_top_1, v_bot_1, v_bot_2])
        wall_vertices.append([v_top_1, v_bot_2, v_top_2])

    return wall_vertices


def robust_triangulate(vertices, edges, holes=None):
    from shapely.geometry import MultiLineString

    # 1. Geometry Cleaning & Shell Formation
    lines = [vertices[edge] for edge in edges]
    merged_lines = unary_union(MultiLineString(lines))
    polygons = list(polygonize(merged_lines))

    if not polygons:
        raise ValueError("Could not form a closed polygon from edges.")

    boundary_poly = unary_union(polygons)

    # 2. Prep Holes
    holes_coords = []
    if holes:
        for h in holes:
            h_arr = np.array(h)
            if np.allclose(h_arr[0], h_arr[-1]):
                h_arr = h_arr[:-1]
            holes_coords.append(h_arr)

    # 3. Create the Formal Polygon with Holes
    poly = orient(Polygon(boundary_poly.exterior.coords, holes=holes_coords))

    # 4. Generate CDT and Handle GeometryCollection
    tri_output = constrained_delaunay_triangles(poly)

    if hasattr(tri_output, "geoms"):
        tri_list = tri_output.geoms
    else:
        tri_list = tri_output

    keep_coords = []
    for tri in tri_list:
        if poly.contains(tri.centroid):
            keep_coords.append(np.array(tri.exterior.coords)[:3])

    if not keep_coords:
        return np.array([]), np.array([])

    # 6. Re-indexing into Vertices and Faces
    flat_coords = np.vstack(keep_coords)
    unique_verts, inverse_indices = np.unique(np.round(flat_coords, 6), axis=0, return_inverse=True)
    faces = inverse_indices.reshape(-1, 3)

    return unique_verts, faces
