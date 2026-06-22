import numpy as np
import trimesh

from ..processing.extrusion import make_hollow_prism_solid, make_prism_solid


def make_base_border(width, b, m, base_n, height=1, offset_dist=5):

    pts_list = make_puzzle_pts(width, b, m, base_n, a=0, border_buffer=0)

    base_border = {}
    for n, pts in enumerate(pts_list):
        vs, fs = make_hollow_prism_solid(pts, offset_dist=offset_dist, z1=height, z0=0)

        base_border["Base" + str(n)] = vs, fs

    return base_border


def make_puzzle_model(width, b, m, base_n, a=0, z=100):

    pts_list = make_puzzle_pts(width, b, m, base_n, a=a)

    models = {}
    for n, pts in enumerate(pts_list):
        models[str(n)] = make_prism_solid(pts.copy(), z0=-base_n, z1=z)

    for key in models:
        vertices, faces = models[key]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.fix_normals()
        models[key] = mesh.vertices, mesh.faces

    return models


def make_puzzle_pts(width, b, m, base_n, a=0, border_buffer=None, tol=0.4):

    print(tol)

    if border_buffer is None:
        border_buffer = base_n

    width_x, width_y = width if isinstance(width, (list, tuple)) else (width, width)

    grid_range_x = list(np.arange(0, width_x, b))
    grid_range_y = list(np.arange(0, width_y, b))

    bx = min(b, width_x / len(grid_range_x))
    by = min(b, width_y / len(grid_range_y))
    b = (bx, by)

    pts_list = []
    for ni, i in enumerate(grid_range_x):
        for nj, j in enumerate(grid_range_y):
            temp = make_puzzle_piece(
                b, m, ni, nj, base_n, len(grid_range_x), len(grid_range_y), a=a, tol=tol
            )

            temp[:, 0] += ni * bx
            temp[:, 1] += nj * by
            pts_list.append(temp)

    for n, pts in enumerate(pts_list):
        pts[pts == 0] = -border_buffer
        pts[pts[:, 0] > (width_x - base_n), 0] = width_x + border_buffer
        pts[pts[:, 1] > (width_y - base_n), 1] = width_y + border_buffer
        pts_list[n] = pts

    return pts_list


def make_puzzle_piece(b, m, ni, nj, base_n, len_x, len_y, a=0, tol=1):

    def get_notched_square(b, m, n_left, n_bottom, n_right, n_top, a=0, tol=1):

        b_x, b_y = b if isinstance(b, (list, tuple)) else (b, b)

        m_l = m + tol if n_left < 0 else m
        m_b = m + tol if n_bottom < 0 else m
        m_r = m + tol if n_right < 0 else m
        m_t = m + tol if n_top < 0 else m

        x = np.array(
            [
                # Left Side
                [a, b_y],
                [a, b_y - m_l],
                [a + n_left, b_y - m_l],
                [a + n_left, m_l],
                [a, m_l],
                # Bottom Side
                [a, a],
                [m_b, a],
                [m_b, a + n_bottom],
                [b_x - m_b, a + n_bottom],
                [b_x - m_b, a],
                # Right Side
                [b_x, a],
                [b_x, m_r],
                [b_x - n_right, m_r],
                [b_x - n_right, b_y - m_r],
                [b_x, b_y - m_r],
                # Top Side
                [b_x, b_y],
                [b_x - m_t, b_y],
                [b_x - m_t, b_y - n_top],
                [m_t, b_y - n_top],
                [m_t, b_y],
                # Return to Start
                [a, b_y],
            ]
        )
        return x

    n_l = (1 if (nj) % 2 == 0 else -1) * base_n
    n_b = (1 if (ni) % 2 == 0 else -1) * base_n
    n_r = -n_l
    n_t = -n_b

    if ni == 0:
        n_l = 0
    if ni == len_x - 1:
        n_r = 0
    if nj == 0:
        n_b = 0
    if nj == len_y - 1:
        n_t = 0

    return get_notched_square(b, m, n_l, n_b, n_r, n_t, a, tol)
