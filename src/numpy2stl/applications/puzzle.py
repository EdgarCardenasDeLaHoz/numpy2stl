"""Jigsaw cutters: prisms that split a model into interlocking pieces.

Coordinates are model coordinates: x along ``width_x`` (array columns), y
along ``width_y`` (array rows), 1 unit = 1 mm. Piece ``r{row}c{col}`` covers
grid cell ``[col*w, (col+1)*w] x [row*h, (row+1)*h]`` with row 0 at min-y.
"""

import math

import numpy as np
import trimesh
from shapely.geometry import box
from shapely.geometry.polygon import orient


def make_jigsaw_cutters(
    width_x,
    width_y,
    cols,
    rows,
    knob_width,
    knob_depth,
    clearance=0.3,
    z0=-1.0,
    z1=1000.0,
    border=1.0,
):
    """Prism cutters that tile a ``width_x`` x ``width_y`` rectangle as a jigsaw.

    Parameters
    ----------
    width_x, width_y : float
        Model footprint (x = columns, y = rows).
    cols, rows : int
        Grid size; each piece is ``width_x/cols`` by ``width_y/rows``.
    knob_width, knob_depth : float
        Size of the rectangular tongue centred on every interior edge.
    clearance : float
        Gap between neighbours, taken from the groove side only: the groove
        piece's whole shared edge (groove included) is set back by this much.
    z0, z1 : float
        Bottom and top of the cutter prisms; must enclose the model.
    border : float
        How far outer edges extend past the rectangle, so the model's own
        side faces are fully inside the cutters.

    Returns
    -------
    dict[str, (vertices, faces)]
        Watertight, outward-facing prisms keyed ``r{row}c{col}``.
    """
    outlines = jigsaw_outlines(
        width_x, width_y, cols, rows, knob_width, knob_depth, clearance, border
    )
    return {key: _extrude(poly, z0, z1) for key, poly in outlines.items()}


def jigsaw_outlines(
    width_x, width_y, cols, rows, knob_width, knob_depth, clearance=0.3, border=1.0
):
    """Piece outlines (shapely Polygons, CCW) of the jigsaw grid, keyed ``r{row}c{col}``.

    Tongue direction alternates like a checkerboard: vertical edges point +x in
    even rows and -x in odd rows; horizontal edges point +y in even columns and
    -y in odd columns.
    """
    cols, rows = int(cols), int(rows)
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    w, h = width_x / cols, width_y / rows
    k2, d, c = knob_width / 2.0, float(knob_depth), float(clearance)
    if (cols > 1 or rows > 1) and (min(k2, d) <= 0 or d + k2 + 2 * c >= min(w, h) / 2):
        raise ValueError(
            f"knob {knob_width}x{knob_depth} (+{clearance} clearance) does not fit "
            f"pieces of {w:.3g}x{h:.3g}: need knob_depth + knob_width/2 + 2*clearance "
            f"< {min(w, h) / 2:.3g}"
        )
    xs = np.arange(cols + 1) * w
    ys = np.arange(rows + 1) * h
    xs[-1], ys[-1] = width_x, width_y

    pieces = {}
    for r in range(rows):
        for q in range(cols):
            x0 = xs[q] - (border if q == 0 else 0.0)
            x1 = xs[q + 1] + (border if q == cols - 1 else 0.0)
            y0 = ys[r] - (border if r == 0 else 0.0)
            y1 = ys[r + 1] + (border if r == rows - 1 else 0.0)
            pieces[(r, q)] = box(x0, y0, x1, y1)

    def join(tongue, groove, edge_box, tongue_box, groove_box):
        pieces[tongue] = pieces[tongue].union(tongue_box)
        pieces[groove] = pieces[groove].difference(edge_box.union(groove_box))

    for r in range(rows):
        yc = (ys[r] + ys[r + 1]) / 2
        for q in range(cols - 1):
            X = xs[q + 1]
            left, right = (r, q), (r, q + 1)
            if r % 2 == 0:  # tongue points +x, into the right piece
                join(
                    left,
                    right,
                    box(X, ys[r] - border, X + c, ys[r + 1] + border),
                    box(X, yc - k2, X + d, yc + k2),
                    box(X, yc - k2 - c, X + d + c, yc + k2 + c),
                )
            else:
                join(
                    right,
                    left,
                    box(X - c, ys[r] - border, X, ys[r + 1] + border),
                    box(X - d, yc - k2, X, yc + k2),
                    box(X - d - c, yc - k2 - c, X, yc + k2 + c),
                )
    for q in range(cols):
        xc = (xs[q] + xs[q + 1]) / 2
        for r in range(rows - 1):
            Y = ys[r + 1]
            low, high = (r, q), (r + 1, q)
            if q % 2 == 0:  # tongue points +y, into the upper piece
                join(
                    low,
                    high,
                    box(xs[q] - border, Y, xs[q + 1] + border, Y + c),
                    box(xc - k2, Y, xc + k2, Y + d),
                    box(xc - k2 - c, Y, xc + k2 + c, Y + d + c),
                )
            else:
                join(
                    high,
                    low,
                    box(xs[q] - border, Y - c, xs[q + 1] + border, Y),
                    box(xc - k2, Y - d, xc + k2, Y),
                    box(xc - k2 - c, Y - d - c, xc + k2 + c, Y),
                )

    out = {}
    for (r, q), poly in pieces.items():
        if poly.geom_type != "Polygon":
            raise ValueError(f"piece r{r}c{q} is not a single polygon ({poly.geom_type})")
        out[f"r{r}c{q}"] = orient(poly.simplify(0), 1.0)
    return out


def _extrude(poly, z0, z1):
    mesh = trimesh.creation.extrude_polygon(poly, height=z1 - z0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices[:, 2] += z0
    return vertices, np.asarray(mesh.faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Square-piece interface: piece size ``b``, knob margin ``m``, depth ``base_n``
# ---------------------------------------------------------------------------
def _grid_from_square(width, b, m, base_n, tol, border):
    width_x, width_y = width if isinstance(width, (list, tuple)) else (width, width)
    cols = max(1, math.ceil(width_x / b))
    rows = max(1, math.ceil(width_y / b))
    knob_width = min(width_x / cols, width_y / rows) - 2 * m
    outlines = jigsaw_outlines(
        width_x, width_y, cols, rows, knob_width, base_n, clearance=tol, border=border
    )
    # Old ordering: column-major, row index fastest.
    return [outlines[f"r{r}c{q}"] for q in range(cols) for r in range(rows)]


def make_base_border(width, b, m, base_n, height=1, offset_dist=5):
    """Hollow frames (``offset_dist`` wide, ``height`` tall) following each piece outline."""
    polys = _grid_from_square(width, b, m, base_n, tol=0.4, border=0.0)

    base_border = {}
    for n, poly in enumerate(polys):
        ring = poly.difference(poly.buffer(-offset_dist, join_style=2))
        base_border["Base" + str(n)] = _extrude(orient(ring, 1.0), 0.0, height)

    return base_border


def make_puzzle_model(width, b, m, base_n, a=0, z=100):
    """Jigsaw cutters for pieces of about ``b`` x ``b``; ``width = (width_x, width_y)``.

    The grid has ``ceil(width_x/b)`` x ``ceil(width_y/b)`` equal pieces, knobs
    are ``min(piece side) - 2*m`` wide and ``base_n`` deep. ``a`` is accepted
    for old call sites and has no effect. Prefer ``make_jigsaw_cutters``.
    """
    polys = _grid_from_square(width, b, m, base_n, tol=0.4, border=base_n)
    return {str(n): _extrude(poly, -base_n, z) for n, poly in enumerate(polys)}


def make_puzzle_pts(width, b, m, base_n, a=0, border_buffer=None, tol=0.4):
    """Closed (N, 2) outlines of ``make_puzzle_model``'s pieces, column-major order."""
    border = base_n if border_buffer is None else border_buffer
    polys = _grid_from_square(width, b, m, base_n, tol=tol, border=border)
    return [np.asarray(p.exterior.coords) for p in polys]


def make_puzzle_piece(b, m, ni, nj, base_n, len_x, len_y, a=0, tol=1):
    """Closed outline of piece (column ``ni``, row ``nj``) in its own cell frame."""
    b_x, b_y = b if isinstance(b, (list, tuple)) else (b, b)
    polys = jigsaw_outlines(
        b_x * len_x,
        b_y * len_y,
        len_x,
        len_y,
        min(b_x, b_y) - 2 * m,
        base_n,
        clearance=tol,
        border=0.0,
    )
    pts = np.asarray(polys[f"r{nj}c{ni}"].exterior.coords)
    return pts - [ni * b_x, nj * b_y]
