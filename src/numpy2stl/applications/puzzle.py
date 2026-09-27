"""Jigsaw puzzles: piece outlines, cutters, heightfield pieces, marks and plate layout.

Coordinates are model coordinates: x along ``width_x`` (array columns), y
along ``width_y`` (array rows), 1 unit = 1 mm. Piece ``r{row}c{col}`` covers
grid cell ``[col*w, (col+1)*w] x [row*h, (row+1)*h]`` with row 0 at min-y, or
``[col_edges[col], col_edges[col+1]] x [row_edges[row], row_edges[row+1]]`` for
a non-uniform grid.

* :func:`jigsaw_outlines` / :func:`make_jigsaw_cutters` - piece outlines and the
  prisms that cut a finished mesh (``processing.boolean.cut_jigsaw``).
* :func:`heightfield_pieces` - pieces meshed straight from a heightfield inside
  each outline: no boolean engine, for terrain-only models.
* :func:`underside_marks` + :func:`engrave_underside` - piece id and north arrow
  engraved into each piece's bottom.
* :func:`plate_layout` - pieces spread over one or more print beds.
"""

import logging
import math

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import polylabel

from ..core.extrude import orient_ccw, prism

logger = logging.getLogger(__name__)

KNOB_SHAPES = ("classic", "dovetail", "rectangular")


def _tongue(shape, k2, d):
    """Tongue outline in the edge frame: u in [0, d] away from the edge, v across.

    ``k2`` is half the widest part. ``classic`` is the rounded jigsaw knob (a
    neck half as wide as the oval head), ``dovetail`` widens from 0.6x at the
    edge to the full width at the tip, ``rectangular`` is a plain tab.
    """
    if shape == "rectangular":
        return box(0.0, -k2, d, k2)
    if shape == "dovetail":
        return Polygon([(0.0, -0.6 * k2), (d, -k2), (d, k2), (0.0, 0.6 * k2)])
    if shape == "classic":
        head = affinity.scale(Point(0.0, 0.0).buffer(1.0, quad_segs=8), 0.45 * d, k2,
                              origin=(0, 0))
        head = affinity.translate(head, 0.55 * d, 0.0)
        return shapely.union(box(0.0, -0.5 * k2, 0.55 * d, 0.5 * k2), head)
    raise ValueError(f"unknown knob_shape {shape!r}; use one of {KNOB_SHAPES}")


def _edge_frame(poly, axis, at, centre, sign):
    """Map an edge-frame shape to model coordinates.

    ``axis`` 0: vertical edge x = ``at``, u along ``sign`` * x, v along y about
    ``centre``; axis 1: horizontal edge y = ``at``, u along ``sign`` * y, v along x.
    """
    if axis == 0:
        return affinity.affine_transform(poly, [sign, 0, 0, 1, at, centre])
    return affinity.affine_transform(poly, [0, 1, sign, 0, centre, at])


def validate_edges(edges, width, count=None, name="edges"):
    """Cut positions as a float array ``[0, ..., width]``, strictly increasing.

    The first and last values must be within ``max(0.5, 1% of width)`` of 0 and
    ``width`` and are snapped to them (callers measure the model independently).
    """
    e = np.asarray(edges, dtype=np.float64).ravel()
    if e.size < 2 or not np.all(np.isfinite(e)):
        raise ValueError(f"{name}: need at least two finite positions")
    if count is not None and e.size != count + 1:
        raise ValueError(f"{name}: {e.size} positions for {count} pieces (need {count + 1})")
    slack = max(0.5, 0.01 * width)
    if abs(e[0]) > slack or abs(e[-1] - width) > slack:
        raise ValueError(f"{name} must run from 0 to the model size {width:.1f} mm "
                         f"(got {e[0]:.1f} .. {e[-1]:.1f})")
    e[0], e[-1] = 0.0, float(width)
    if np.any(np.diff(e) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return e


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
    knob_shape="rectangular",
    col_edges=None,
    row_edges=None,
):
    """Prism cutters that tile a ``width_x`` x ``width_y`` rectangle as a jigsaw.

    Parameters
    ----------
    width_x, width_y : float
        Model footprint (x = columns, y = rows).
    cols, rows : int
        Grid size; each piece is ``width_x/cols`` by ``width_y/rows``.
    knob_width, knob_depth : float
        Size (widest part x reach) of the tongue centred on every interior edge.
    clearance : float
        Gap between neighbours, taken from the groove side only: the groove
        piece's whole shared edge (groove included) is set back by this much.
    z0, z1 : float
        Bottom and top of the cutter prisms; must enclose the model.
    border : float
        How far outer edges extend past the rectangle, so the model's own
        side faces are fully inside the cutters.
    knob_shape : {"rectangular", "classic", "dovetail"}
        Tongue outline (see :func:`jigsaw_outlines`).
    col_edges, row_edges : sequence of float, optional
        Explicit cut positions ``[0, ..., width]`` (``cols + 1`` / ``rows + 1``
        values) for a non-uniform grid; default: equal pieces.

    Returns
    -------
    dict[str, (vertices, faces)]
        Watertight, outward-facing prisms keyed ``r{row}c{col}``.
    """
    outlines = jigsaw_outlines(
        width_x, width_y, cols, rows, knob_width, knob_depth, clearance, border,
        knob_shape=knob_shape, col_edges=col_edges, row_edges=row_edges,
    )
    return {key: _extrude(poly, z0, z1) for key, poly in outlines.items()}


def jigsaw_outlines(
    width_x, width_y, cols, rows, knob_width, knob_depth, clearance=0.3, border=1.0,
    knob_shape="rectangular", col_edges=None, row_edges=None,
):
    """Piece outlines (shapely Polygons, CCW) of the jigsaw grid, keyed ``r{row}c{col}``.

    Tongue direction alternates like a checkerboard: vertical edges point +x in
    even rows and -x in odd rows; horizontal edges point +y in even columns and
    -y in odd columns. The groove is the tongue grown by ``clearance`` (mitred
    for the straight-sided shapes, rounded for ``classic``) plus a ``clearance``
    strip along the whole shared edge, both taken from the groove piece.
    """
    cols, rows = int(cols), int(rows)
    if cols < 1 or rows < 1:
        raise ValueError("cols and rows must be >= 1")
    if knob_shape not in KNOB_SHAPES:
        raise ValueError(f"unknown knob_shape {knob_shape!r}; use one of {KNOB_SHAPES}")
    if col_edges is None:
        xs = np.arange(cols + 1) * (width_x / cols)
        xs[-1] = width_x
    else:
        xs = validate_edges(col_edges, width_x, cols, "col_edges")
    if row_edges is None:
        ys = np.arange(rows + 1) * (width_y / rows)
        ys[-1] = width_y
    else:
        ys = validate_edges(row_edges, width_y, rows, "row_edges")
    w, h = float(np.diff(xs).min()), float(np.diff(ys).min())
    k2, d, c = knob_width / 2.0, float(knob_depth), float(clearance)
    if (cols > 1 or rows > 1) and (min(k2, d) <= 0 or d + k2 + 2 * c >= min(w, h) / 2):
        raise ValueError(
            f"knob {knob_width}x{knob_depth} (+{clearance} clearance) does not fit "
            f"pieces of {w:.3g}x{h:.3g}: need knob_depth + knob_width/2 + 2*clearance "
            f"< {min(w, h) / 2:.3g}"
        )
    tongue = _tongue(knob_shape, k2, d) if min(k2, d) > 0 else None
    groove = None
    if tongue is not None:
        grown = tongue.buffer(c, join_style="round" if knob_shape == "classic" else "mitre",
                              quad_segs=4) if c > 0 else tongue
        # Only the far side of the edge; the near side is the strip below.
        groove = shapely.intersection(grown, box(0.0, -k2 - 2 * c - 1.0,
                                                 d + 2 * c + 1.0, k2 + 2 * c + 1.0))

    pieces = {}
    for r in range(rows):
        for q in range(cols):
            x0 = xs[q] - (border if q == 0 else 0.0)
            x1 = xs[q + 1] + (border if q == cols - 1 else 0.0)
            y0 = ys[r] - (border if r == 0 else 0.0)
            y1 = ys[r + 1] + (border if r == rows - 1 else 0.0)
            pieces[(r, q)] = box(x0, y0, x1, y1)

    def join(tongue_key, groove_key, axis, at, centre, sign, lo, hi):
        """Tongue from ``tongue_key`` across the edge ``at`` into ``groove_key``."""
        pieces[tongue_key] = pieces[tongue_key].union(_edge_frame(tongue, axis, at, centre, sign))
        cut = _edge_frame(groove, axis, at, centre, sign)
        if c > 0:
            strip = box(0.0, lo - border - centre, c, hi + border - centre)
            cut = cut.union(_edge_frame(strip, axis, at, centre, sign))
        pieces[groove_key] = pieces[groove_key].difference(cut)

    for r in range(rows):
        yc = (ys[r] + ys[r + 1]) / 2
        for q in range(cols - 1):
            left, right = (r, q), (r, q + 1)
            if r % 2 == 0:  # tongue points +x, into the right piece
                join(left, right, 0, xs[q + 1], yc, 1.0, ys[r], ys[r + 1])
            else:
                join(right, left, 0, xs[q + 1], yc, -1.0, ys[r], ys[r + 1])
    for q in range(cols):
        xc = (xs[q] + xs[q + 1]) / 2
        for r in range(rows - 1):
            low, high = (r, q), (r + 1, q)
            if q % 2 == 0:  # tongue points +y, into the upper piece
                join(low, high, 1, ys[r + 1], xc, 1.0, xs[q], xs[q + 1])
            else:
                join(high, low, 1, ys[r + 1], xc, -1.0, xs[q], xs[q + 1])

    out = {}
    for (r, q), poly in pieces.items():
        if poly.geom_type != "Polygon":
            raise ValueError(f"piece r{r}c{q} is not a single polygon ({poly.geom_type})")
        out[f"r{r}c{q}"] = orient(poly.simplify(0), 1.0)
    return out


def _extrude(poly, z0, z1):
    return prism(poly, z0, z1)


# ---------------------------------------------------------------------------
# Pieces straight from a heightfield (no boolean engine)
# ---------------------------------------------------------------------------
def _sample(z, x, y, s):
    """Bilinear height at model (x, y); pixel (i, j) is at (j*s, (H-1-i)*s)."""
    from scipy.ndimage import map_coordinates

    h = z.shape[0]
    return map_coordinates(z, [(h - 1) - np.asarray(y) / s, np.asarray(x) / s],
                           order=1, mode="nearest")


def heightfield_pieces(z, outlines, max_error, mm_per_px=1.0, floor=0.0, seed_step=8,
                       max_loss=0.01):
    """Jigsaw pieces meshed directly from a heightfield, one closed solid per outline.

    The fast path for a terrain-only puzzle: no boolean engine. The adaptive
    terrain vertices (``processing.decimate.heightfield_tin`` within
    ``max_error``) that fall inside each outline are found with one label
    raster (``raster.burn_polygons`` at the DEM grid), then triangulated
    together with the outline itself (constrained Delaunay, the outline split
    at the pixel spacing and sampled bilinearly). So piece walls follow the
    exact outline - knobs, clearance and all - not the pixel grid, and the top
    is the same surface the whole-model export has (to within ``max_error``).
    Each top gets walls down to a flat bottom at ``floor``, triangulated from the
    outline alone (``core.extrude.close_surface`` would copy the whole top).

    Parameters
    ----------
    z : (H, W) array
        Top surface; row 0 is north: pixel (i, j) is at x = j*s, y = (H-1-i)*s.
    outlines : dict[str, shapely Polygon]
        Piece outlines in the same frame, e.g. ``jigsaw_outlines((W-1)*s, (H-1)*s, ...)``.
        They are clipped to the heightfield's rectangle.
    max_loss : float or None
        As in ``cut_jigsaw``: the pieces must add up to the model's volume minus
        the clearance gaps to within this fraction; ``None`` skips the check.

    Returns
    -------
    dict[str, (vertices, faces)]
    """
    import triangle

    from ..processing.decimate import heightfield_tin
    from ..raster.burn import burn_polygons

    z = np.asarray(z, dtype=np.float64)
    h, w = z.shape
    s = float(mm_per_px)
    rect = box(0.0, 0.0, (w - 1) * s, (h - 1) * s)
    idx, tris = heightfield_tin(z, max_error, seed_step=seed_step)
    keys = list(outlines)
    clipped = [shapely.intersection(outlines[k], rect) for k in keys]
    # Pixel centres are cell centres of a grid whose outer edges sit half a pixel out.
    labels = burn_polygons([p.buffer(-0.05 * s) for p in clipped], (h, w),
                           bounds=(-0.5 * s, -0.5 * s, (w - 0.5) * s, (h - 0.5) * s),
                           values=np.arange(1, len(keys) + 1), mode="set",
                           dtype=np.int32).ravel()
    lab = labels[idx]
    ii, jj = np.divmod(idx, w)
    zf = z.ravel()

    pieces = {}
    for n, (key, poly) in enumerate(zip(keys, clipped, strict=True), start=1):
        if poly.is_empty or poly.area <= 0:
            continue
        if poly.geom_type != "Polygon" or poly.interiors:
            raise ValueError(f"piece {key}: outline is not a simple polygon after clipping")
        ring = np.asarray(shapely.segmentize(poly.exterior, s).coords)[:-1]
        inside = lab == n
        inner = np.column_stack([jj[inside] * s, (h - 1 - ii[inside]) * s])
        verts = np.vstack([ring, inner])
        verts, inv = np.unique(verts, axis=0, return_inverse=True)
        inv = inv.ravel()
        k = inv[: len(ring)]
        seg = np.column_stack([k, np.roll(k, -1)])
        seg = seg[seg[:, 0] != seg[:, 1]]
        out = triangle.triangulate({"vertices": verts, "segments": seg.astype(np.int32)}, "pQ")
        v2, f = out["vertices"], out["triangles"]
        top = np.column_stack([v2, _sample(z, v2[:, 0], v2[:, 1], s)])
        # Interior vertices are pixel centres: take the exact value, not an interpolation.
        cj, ci = v2[:, 0] / s, (h - 1) - v2[:, 1] / s
        on_px = (np.abs(cj - np.rint(cj)) < 1e-9) & (np.abs(ci - np.rint(ci)) < 1e-9)
        top[on_px, 2] = zf[np.rint(ci[on_px]).astype(np.int64) * w
                           + np.rint(cj[on_px]).astype(np.int64)]
        f = orient_ccw(top, f)
        pieces[key] = _close_flat(top, f, seg, float(floor))

    if max_loss is not None:
        _check_heightfield_volume(z, idx, tris, s, floor, clipped, rect, pieces, max_loss)
    return pieces


def _close_flat(top, faces, seg, floor):
    """Solid from an upward (CCW) top surface whose boundary is the segment loop
    ``seg``: walls down to ``floor`` and a bottom triangulated from the loop."""
    import triangle

    ring = np.unique(seg)
    local = np.searchsorted(ring, seg)
    out = triangle.triangulate({"vertices": top[ring, :2], "segments": local.astype(np.int32)}, "pQ")
    if len(out["vertices"]) != len(ring):
        raise ValueError("outline triangulation added vertices")
    n = len(top)
    bottom = np.column_stack([top[ring, :2], np.full(len(ring), floor)])
    below = np.full(n, -1)
    below[ring] = n + np.arange(len(ring))
    bf = orient_ccw(bottom, out["triangles"])[:, ::-1] + n     # facing down
    f = np.asarray(faces)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, inv, counts = np.unique(np.sort(edges, axis=1), axis=0,
                               return_inverse=True, return_counts=True)
    border = edges[counts[inv.ravel()] == 1]          # directed as in the CCW top faces
    a, b = border[:, 0], border[:, 1]
    walls = np.vstack([np.column_stack([a, below[b], b]), np.column_stack([a, below[a], below[b]])])
    return np.vstack([top, bottom]), np.vstack([f, bf, walls])


def _check_heightfield_volume(z, idx, tris, s, floor, clipped, rect, pieces, max_loss):
    """Pieces must add up to the model minus the gaps between the outlines.

    Model volume: the whole-model TIN. The gaps are the part of the model inside
    the outlines' convex hull but in no outline (as in ``cut_jigsaw``; model
    outside the hull was never covered and counts as lost); their volume is the
    gap area x the mean height sampled along the gap boundaries (they are
    clearance-wide strips). Overlapping outlines are rejected too.
    """
    from ..processing.boolean import mesh_volume

    h, w = z.shape
    ii, jj = np.divmod(idx, w)
    xy = np.column_stack([jj * s, (h - 1 - ii) * s])
    t = np.searchsorted(idx, tris)   # tris index pixels; idx is sorted
    a, b, c = xy[t[:, 0]], xy[t[:, 1]], xy[t[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    zt = z.ravel()[idx][t]
    v_in = float((area * (zt.mean(axis=1) - floor)).sum())
    union = shapely.union_all(clipped)
    gaps = shapely.difference(shapely.intersection(rect, union.convex_hull), union)
    v_gap = 0.0
    if not gaps.is_empty and gaps.area > 0:
        pts = shapely.get_coordinates(shapely.segmentize(gaps.boundary, s))
        v_gap = gaps.area * float(np.mean(_sample(z, pts[:, 0], pts[:, 1], s) - floor))
    overlap = sum(p.area for p in clipped) - union.area
    v_out = sum(abs(mesh_volume(v, f)) for v, f in pieces.values())
    missing = (v_in - v_gap - v_out) / v_in if v_in > 0 else 0.0
    logger.info("heightfield_pieces: %d pieces, clearance gaps %.3f%%, unaccounted %.3f%%",
                len(pieces), 100 * v_gap / max(v_in, 1e-300), 100 * missing)
    if overlap > 1e-6 * rect.area or abs(missing) > max_loss:
        raise ValueError(
            f"jigsaw pieces miss {missing:.2%} of the model volume beyond the clearance gaps "
            f"(limit {max_loss:.2%}); do the outlines cover the whole model without overlapping?"
        )


# ---------------------------------------------------------------------------
# Underside marks (piece id + north arrow)
# ---------------------------------------------------------------------------
def _text_polygons(text, height):
    """Outline of ``text`` (matplotlib's bundled DejaVu Sans Bold), cap height ~ ``height``.

    Returns a shapely (Multi)Polygon with its lower-left corner at the origin.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    # DejaVu's digits are ~0.73 em tall; size in em so the digits come out ``height``.
    path = TextPath((0, 0), text, size=height / 0.73,
                    prop=FontProperties(family="DejaVu Sans", weight="bold"))
    rings = [r for r in path.to_polygons() if len(r) >= 3]
    shape = Polygon()
    for r in rings:   # even-odd fill: counters (the holes in 0, 4, 6, 8, 9) cancel out
        shape = shapely.symmetric_difference(shape, shapely.make_valid(Polygon(r)))
    shape = shapely.make_valid(shape)
    if shape.is_empty:
        return shape
    x0, y0, _, _ = shape.bounds
    return affinity.translate(shape, -x0, -y0)


def _north_arrow(size):
    """Arrow pointing +y (north), ``size`` tall, base centre at the origin."""
    head = Polygon([(-0.45 * size, 0.5 * size), (0.45 * size, 0.5 * size), (0.0, size)])
    shaft = box(-0.14 * size, 0.0, 0.14 * size, 0.52 * size)
    return shapely.union(head, shaft)


def underside_marks(label, outline, text_height=None, min_text_mm=2.0, margin=1.0):
    """Piece id and north arrow for the bottom of one piece, mirrored to read from below.

    ``text_height`` defaults to ``min(8, 20% of the piece's shorter side)``. The
    marks are centred on the outline's pole of inaccessibility and shrunk until
    they sit ``margin`` inside it. Mirrored in x: flipped over (about the y
    axis) to read the underside, the text reads normally and the arrow still
    points north. Returns a shapely geometry, or None if it cannot fit at
    ``min_text_mm``.
    """
    x0, y0, x1, y1 = outline.bounds
    if text_height is None:
        text_height = min(8.0, 0.2 * min(x1 - x0, y1 - y0))
    inner = outline.buffer(-margin)
    if inner.is_empty:
        return None
    centre = (polylabel(inner, tolerance=0.5) if inner.geom_type == "Polygon"
              else inner.representative_point())
    th = float(text_height)
    while th >= min_text_mm:
        text = _text_polygons(label, th)
        tx, ty = text.bounds[2], text.bounds[3]
        # Arrow above the text, the pair centred on the origin.
        marks = shapely.union(affinity.translate(text, -tx / 2, 0.0),
                              affinity.translate(_north_arrow(th), 0.0, ty + 0.3 * th))
        x0, y0, x1, y1 = marks.bounds
        marks = affinity.translate(marks, -(x0 + x1) / 2, -(y0 + y1) / 2)
        marks = affinity.scale(marks, -1.0, 1.0, origin=(0, 0))
        marks = affinity.translate(marks, centre.x, centre.y)
        if inner.contains(marks):
            return marks
        th *= 0.8
    return None


def engrave_underside(vertices, faces, marks, depth=0.6, floor=None):
    """``marks`` (shapely area) cut ``depth`` into the mesh's bottom face.

    ``floor`` defaults to the mesh's lowest z. Uses manifold3d (the pieces are
    small, so this is quick). Returns (vertices, faces).
    """
    from ..processing.boolean import from_manifold, to_manifold, union

    v = np.asarray(vertices, dtype=np.float64)
    z0 = float(v[:, 2].min()) if floor is None else float(floor)
    solids = []
    for part in shapely.get_parts(shapely.make_valid(marks)):
        if part.geom_type == "Polygon" and part.area > 1e-6:
            p = prism(part, z0 - 1.0, z0 + depth)
            if p is not None:
                solids.append(p)
    cut, _ = union(solids)
    if cut is None:
        return v, np.asarray(faces)
    return from_manifold(to_manifold(v, faces, "piece") - cut)


# ---------------------------------------------------------------------------
# Plate layout
# ---------------------------------------------------------------------------
def plate_layout(pieces, bed_w, bed_h, gap=5.0, margin=5.0):
    """Spread pieces over print beds, row by row (shelf packing), each on z = 0.

    ``pieces``: {key: (vertices, faces)} in any frame. Pieces go left to right
    in key order, a new shelf when the row is full, a new plate when the bed
    is. A piece larger than the bed gets a plate of its own (and is reported).

    Returns (plates, oversize): ``plates`` is a list of {key: (vertices, faces)}
    translated onto the bed ``[margin, bed_w - margin] x [margin, bed_h - margin]``;
    ``oversize`` lists the keys that do not fit the bed at all.
    """
    plates, oversize = [], []
    cur, x, y, shelf = {}, margin, margin, 0.0
    for key, (v, f) in pieces.items():
        v = np.asarray(v, dtype=np.float64)
        lo, hi = v.min(axis=0), v.max(axis=0)
        pw, ph = hi[0] - lo[0], hi[1] - lo[1]
        if pw > bed_w - 2 * margin or ph > bed_h - 2 * margin:
            oversize.append(key)
        if x + pw > bed_w - margin and x > margin:         # next shelf
            x, y, shelf = margin, y + shelf + gap, 0.0
        if y + ph > bed_h - margin and y > margin and cur:  # next plate
            plates.append(cur)
            cur, x, y, shelf = {}, margin, margin, 0.0
        cur[key] = (v - [lo[0] - x, lo[1] - y, lo[2]], f)
        x += pw + gap
        shelf = max(shelf, ph)
    if cur:
        plates.append(cur)
    return plates, oversize


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
