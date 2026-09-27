"""Lossless simplification of flat mesh regions.

``simplify_mesh_surfaces`` replaces every planar, edge-connected patch of a
mesh by the fewest triangles that cover exactly the same polygon. Nothing
moves: interior vertices of a patch and vertices lying on a straight shared
edge between two patches are dropped, every other vertex stays, and no vertex
is added. Heightmap solids (1 px = 1 mm) shrink by orders of magnitude on
plateaus, flat sea and the walls/base, while slopes keep their full detail.
"""

import logging
import time

import numpy as np
import triangle as tr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..core.polygon import triangulate_polygon

logger = logging.getLogger(__name__)


def simplify_mesh_surfaces(vertices, faces, min_faces=2, tol=1e-9):
    """Retriangulate the flat regions of a mesh without changing its geometry.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    faces : ndarray of int, shape (M, 3)
        Consistently wound triangles. A watertight input gives a watertight
        output.
    min_faces : int
        Planar regions with fewer faces are left untouched.
    tol : float
        Relative tolerance (scaled by the bounding-box diagonal for offsets)
        for coplanarity and collinearity. Keep it tight: it bounds how far a
        dropped vertex may sit from the plane or line that replaces it.

    Returns
    -------
    faces : ndarray of int, shape (K, 3)
        New faces indexing the *input* ``vertices`` (unused vertices are simply
        no longer referenced). Every region keeps its exact boundary and the
        orientation of the faces it replaces.

    Notes
    -----
    - Regions are grouped by (normal, plane offset) and must be edge-connected;
      two coplanar patches that only touch at a vertex stay separate.
    - A boundary vertex is only dropped when exactly two regions meet at it
      and it is collinear with its neighbours on the shared edge, so both
      regions drop it together and no T-junction appears.
    - Holes (islands in a flat sea, courtyards in a roof) are handled: each
      region is triangulated as a planar straight-line graph and the inside is
      recovered by flood fill from the oriented boundary.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if len(F) == 0:
        return F.copy()

    t0 = time.perf_counter()
    F = _merge_coincident(V, F)
    used = np.unique(F)
    scale = float(np.linalg.norm(np.ptp(V[used], axis=0))) or 1.0

    mesh = _HalfEdges(V, F, scale, tol)
    labels, candidate = mesh.planar_regions(min_faces)
    t1 = time.perf_counter()

    was_open = (mesh.twin < 0).reshape(-1, 3)
    n_retry = 0
    while True:
        removed, segments = mesh.collapse_boundaries(labels, candidate)
        new, new_reg, failed = _retriangulate(mesh, labels, candidate, removed, segments)
        touched = np.zeros(len(candidate), dtype=bool)
        touched[new_reg] = True
        kept = np.flatnonzero(~touched[labels])
        out = np.vstack([F[kept], new])
        if not failed:
            failed = _cracked_regions(out, len(kept), new_reg, was_open[kept], segments, mesh.nv)
        if not failed:
            break
        # A region that could not be retriangulated keeps its faces, so its
        # neighbours must keep every vertex they share with it.
        candidate[failed] = False
        n_retry += 1
        logger.warning(f"simplify: {len(failed)} region(s) kept as-is, retrying")

    logger.info(
        f"simplify: {len(F)} -> {len(out)} faces "
        f"({len(np.unique(new_reg))} regions retriangulated, {n_retry} retries; "
        f"regions {t1 - t0:.2f}s, total {time.perf_counter() - t0:.2f}s)"
    )
    return out


# -----------------------------
# MESH TOPOLOGY
# -----------------------------
def _merge_coincident(V, F):
    """Point faces at one canonical input index per distinct position."""
    Vn = np.ascontiguousarray(V + 0.0)  # -0.0 -> 0.0
    view = Vn.view(np.dtype((np.void, Vn.dtype.itemsize * 3))).ravel()
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    if len(first) == len(V):
        return F
    return first[inverse.ravel()][F]


class _HalfEdges:
    """Half-edge arrays of an indexed triangle mesh (edge ``3f+i`` = corner i -> i+1)."""

    def __init__(self, V, F, scale, tol):
        self.V, self.F, self.scale, self.tol = V, F, scale, tol
        self.nv = len(V)
        self.a = F.ravel()
        self.b = F[:, [1, 2, 0]].ravel()
        self.face = np.repeat(np.arange(len(F)), 3)

        self.twin = _twins(self.a, self.b, self.nv)

        e1 = V[F[:, 1]] - V[F[:, 0]]
        e2 = V[F[:, 2]] - V[F[:, 0]]
        cross = np.cross(e1, e2)
        area2 = np.linalg.norm(cross, axis=1)
        self.degenerate = area2 <= (tol * scale) ** 2
        self.normal = cross / np.where(self.degenerate, 1.0, area2)[:, None]
        self.offset = np.einsum("ij,ij->i", self.normal, V[F[:, 0]])
        self.area2 = area2

    def planar_regions(self, min_faces):
        """Label edge-connected coplanar faces; flag the regions worth retriangulating."""
        h = np.flatnonzero(self.twin > np.arange(len(self.twin)))
        f, g = self.face[h], self.face[self.twin[h]]
        ok = ~self.degenerate[f] & ~self.degenerate[g]
        ok &= np.abs(self.normal[f] - self.normal[g]).max(axis=1) <= self.tol
        ok &= np.abs(self.offset[f] - self.offset[g]) <= self.tol * self.scale
        nf = len(self.F)
        graph = coo_matrix((np.ones(ok.sum(), dtype=np.int8), (f[ok], g[ok])), shape=(nf, nf))
        n_regions, labels = connected_components(graph, directed=False)
        labels = labels.astype(np.int64)

        counts = np.bincount(labels, minlength=n_regions)
        # Reference plane per region: its largest face.
        self.face_order = np.argsort(labels, kind="stable")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        area_sorted = self.area2[self.face_order]
        is_max = area_sorted == np.repeat(np.maximum.reduceat(area_sorted, starts), counts)
        idx = np.flatnonzero(is_max)
        grp = labels[self.face_order[idx]]
        ref = self.face_order[idx[np.r_[True, grp[1:] != grp[:-1]]]]
        self.ref_normal = self.normal[ref]
        self.ref_offset = self.offset[ref]

        # Every vertex of a region must lie on its reference plane.
        dist = np.abs(
            np.einsum("fcj,fj->fc", self.V[self.F], self.ref_normal[labels])
            - self.ref_offset[labels][:, None]
        ).max(axis=1)
        worst = np.maximum.reduceat(dist[self.face_order], starts)

        candidate = (counts >= max(min_faces, 2)) & (worst <= self.tol * self.scale)
        candidate[labels[self.degenerate]] = False
        return labels, candidate

    def collapse_boundaries(self, labels, candidate):
        """Pick removable vertices and chain each region's boundary past them.

        Returns ``removed`` (bool per vertex) and ``segments`` = (start, end,
        region) arrays: the oriented boundary edges of the candidate regions
        after the removed vertices are skipped.
        """
        V, nv = self.V, self.nv
        reg_he = labels[self.face]
        tw = self.twin
        boundary = (tw < 0) | (reg_he != reg_he[np.maximum(tw, 0)])
        self.boundary = boundary

        protected = np.zeros(nv, dtype=bool)
        protected[self.F[~candidate[labels]].ravel()] = True
        protected[self.a[tw < 0]] = True
        protected[self.b[tw < 0]] = True

        # Around a vertex, each outgoing boundary half-edge starts a new sector
        # of the fan: none means the vertex is inside one region.
        n_bout = np.bincount(self.a[boundary], minlength=nv)

        removed = (n_bout == 0) & ~protected

        # Straight-edge vertices between exactly two regions.
        bh = np.flatnonzero(boundary)
        bh = bh[np.argsort(self.a[bh], kind="stable")]
        starts = self.a[bh]
        two = (n_bout == 2) & ~protected
        first = np.flatnonzero(two[starts] & np.r_[True, starts[1:] != starts[:-1]])
        if len(first):
            v = starts[first]
            h1, h2 = bh[first], bh[first + 1]
            ev1 = V[self.b[h1]] - V[v]
            ev2 = V[self.b[h2]] - V[v]
            n1 = np.linalg.norm(ev1, axis=1)
            n2 = np.linalg.norm(ev2, axis=1)
            straight = np.linalg.norm(np.cross(ev1, ev2), axis=1) <= self.tol * n1 * n2
            straight &= np.einsum("ij,ij->i", ev1, ev2) < 0
            straight &= reg_he[h1] != reg_he[h2]
            removed[v[straight]] = True

        # Chain the candidate regions' boundary edges past removed vertices.
        bh = np.flatnonzero(boundary & candidate[reg_he])
        s, e, r = self.a[bh], self.b[bh].copy(), reg_he[bh]
        from_removed = removed[s]
        keys = r[from_removed] * nv + s[from_removed]
        korder = np.argsort(keys)
        keys, next_end = keys[korder], e[from_removed][korder]

        s, e, r = s[~from_removed], e[~from_removed], r[~from_removed]
        active = np.flatnonzero(removed[e])
        for _ in range(len(keys) + 1):
            if len(active) == 0:
                break
            q = r[active] * nv + e[active]
            pos = np.minimum(_sorted_lookup(keys, q), len(keys) - 1)
            hit = keys[pos] == q
            e[active[hit]] = next_end[pos[hit]]
            e[active[~hit]] = s[active[~hit]]  # broken chain: region will fail its checks
            active = active[hit]
            active = active[removed[e[active]]]

        return removed, (s, e, r)


# -----------------------------
# RETRIANGULATION
# -----------------------------
def _retriangulate(mesh, labels, candidate, removed, segments):
    """Triangulate every candidate region that lost a vertex.

    Returns ``(faces, region, failed)``: the new triangles (global vertex
    indices), the region each replaces, and the regions that could not be
    triangulated exactly (their triangles are not in ``faces``).
    """
    F, V, nv = mesh.F, mesh.V, mesh.nv
    n_reg = len(candidate)
    in_todo = np.zeros(n_reg, dtype=bool)
    in_todo[labels[removed[F].any(axis=1)]] = True
    in_todo &= candidate
    todo = np.flatnonzero(in_todo)
    empty = np.zeros((0, 3), dtype=np.int64)
    if len(todo) == 0:
        return empty, np.zeros(0, dtype=np.int64), []

    # (region, vertex) pairs, sorted by region then vertex.
    tf = np.flatnonzero(in_todo[labels])
    fr = labels[tf]
    pair = np.unique(np.repeat(fr, 3) * nv + F[tf].ravel())
    n_vert = np.bincount(pair // nv, minlength=n_reg)
    n_face = np.bincount(fr, minlength=n_reg)
    n_bnd = np.bincount(labels[mesh.face[mesh.boundary]], minlength=n_reg)
    # Euler characteristic 1 (2V - F - B == 2): a disk, no holes or pinches, so
    # Triangle's own concavity removal leaves exactly the region.
    simple = 2 * n_vert - n_face - n_bnd == 2

    pts = pair[~removed[pair % nv]]
    pts_reg, pts_v = pts // nv, pts % nv
    normal = mesh.ref_normal
    k = np.argmax(np.abs(normal), axis=1)
    flip = normal[np.arange(n_reg), k] < 0
    ax0 = np.where(flip, (k + 2) % 3, (k + 1) % 3)
    ax1 = np.where(flip, (k + 1) % 3, (k + 2) % 3)
    # Dropping the dominant axis keeps coordinates exact; the swap for downward
    # normals makes CCW in 2-D mean "facing the normal".
    p2 = np.stack([V[pts_v, ax0[pts_reg]], V[pts_v, ax1[pts_reg]]], axis=1)

    s, e, r = segments
    m = in_todo[r]
    s, e, r = s[m], e[m], r[m]
    order = np.argsort(r, kind="stable")
    s, e, r = s[order], e[order], r[order]
    p_lo, p_hi = np.searchsorted(pts_reg, todo), np.searchsorted(pts_reg, todo, side="right")
    s_lo, s_hi = np.searchsorted(r, todo), np.searchsorted(r, todo, side="right")
    # Segment ends as indices local to their region's point list.
    region_lo = np.zeros(n_reg, dtype=np.int64)
    region_lo[todo] = p_lo
    seg_local = np.stack([_sorted_lookup(pts, r * nv + x) for x in (s, e)], axis=1)
    seg_local -= region_lo[r][:, None]

    tris, tri_reg, failed = [], [], []
    for i, reg in enumerate(todo):
        a = p_lo[i]
        T = _triangulate_pslg(p2[a : p_hi[i]], seg_local[s_lo[i] : s_hi[i]], simple[reg])
        if T is None:
            failed.append(reg)
            continue
        tris.append(T + a)
        tri_reg.append(reg)
    if not tris:
        return empty, np.zeros(0, dtype=np.int64), failed

    new = pts_v[np.concatenate(tris)]
    new_reg = np.repeat(tri_reg, [len(t) for t in tris])

    # Same signed area (along the region normal) and no flipped triangles.
    area_new = _area_along(V[new], normal[new_reg])
    area_old = _area_along(V[F[tf]], normal[fr])
    sum_new = np.bincount(new_reg, weights=area_new, minlength=n_reg)
    sum_old = np.bincount(fr, weights=area_old, minlength=n_reg)
    bad = np.abs(sum_new - sum_old) > 1e-9 * np.abs(sum_old)
    bad[new_reg[area_new <= 0]] = True
    bad &= in_todo
    if bad.any():
        failed.extend(np.flatnonzero(bad).tolist())
        keep = ~bad[new_reg]
        new, new_reg = new[keep], new_reg[keep]
    return new, new_reg, failed


def _triangulate_pslg(p2, seg, simple):
    """Local triangles covering the region bounded by the oriented segments ``seg``."""
    if len(seg) < 3 or np.any(seg[:, 0] == seg[:, 1]):
        return None
    try:
        t = tr.triangulate({"vertices": p2, "segments": seg}, "pQ" if simple else "pnQ")
    except Exception:
        return None
    if "triangles" not in t or len(t["vertices"]) != len(p2):
        return None  # Triangle had to add Steiner points: segments cross
    T = t["triangles"].astype(np.int64)
    if simple:
        return T

    # Holes or pinch vertices: flood fill across unconstrained edges, seeded
    # from the oriented boundary (the region lies left of every segment).
    n = len(p2)
    N = t["neighbors"].astype(np.int64)
    seg_dir = seg[:, 0] * n + seg[:, 1]
    seg_und = np.minimum(seg[:, 0], seg[:, 1]) * n + np.maximum(seg[:, 0], seg[:, 1])
    tri_idx = np.arange(len(T))
    rows, cols = [], []
    inside_seed = np.zeros(len(T), dtype=bool)
    outside_seed = np.zeros(len(T), dtype=bool)
    for j in range(3):
        p, q = T[:, (j + 1) % 3], T[:, (j + 2) % 3]
        und = np.minimum(p, q) * n + np.maximum(p, q)
        free = (N[:, j] >= 0) & ~np.isin(und, seg_und)
        rows.append(tri_idx[free])
        cols.append(N[free, j])
        inside_seed |= np.isin(p * n + q, seg_dir)
        outside_seed |= np.isin(q * n + p, seg_dir)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(T),) * 2)
    _, comp = connected_components(graph, directed=False)
    comp_in = np.unique(comp[inside_seed])
    if np.intersect1d(comp_in, comp[outside_seed]).size:
        return None
    return T[np.isin(comp, comp_in)]


def _area_along(tri, normal):
    """Triangle areas signed by agreement with ``normal``; ``tri`` is (k, 3, 3)."""
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.einsum("ij,ij->i", cross, normal)


def _cracked_regions(out, n_kept, new_reg, kept_was_open, segments, nv):
    """Regions to blame for edges of ``out`` that lost their twin.

    ``out`` is the kept faces followed by the new ones. Every edge that was
    closed must still be closed; the culprit is the new region owning the
    open half-edge, or the region whose boundary segment should have met it.
    """
    open_now = _unmatched(out, nv)
    open_now[: 3 * n_kept] &= ~kept_was_open.ravel()
    crack = np.flatnonzero(open_now)
    if len(crack) == 0:
        return []
    he_face = crack // 3
    blamed = set(new_reg[he_face[he_face >= n_kept] - n_kept].tolist())
    a, b = out.ravel()[crack], out[:, [1, 2, 0]].ravel()[crack]
    seg_s, seg_e, seg_r = segments
    hit = np.isin(seg_s * nv + seg_e, np.r_[b * nv + a, a * nv + b])
    blamed |= set(seg_r[hit].tolist())
    if not blamed:
        raise RuntimeError("simplify opened an edge between untouched regions")
    return sorted(blamed)


def _twins(a, b, nv):
    """Twin of each half-edge ``a -> b`` (-1 if missing, duplicated or non-manifold)."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    und = lo * nv + hi
    order = np.argsort(und)
    su = und[order]
    start = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    size = np.diff(np.r_[start, len(su)])
    i, j = order[start[size == 2]], order[start[size == 2] + 1]
    ok = (a[i] == b[j]) & (b[i] == a[j]) & (a[i] != b[i])
    twin = np.full(len(a), -1, dtype=np.int64)
    twin[i[ok]] = j[ok]
    twin[j[ok]] = i[ok]
    return twin


def _sorted_lookup(sorted_keys, queries):
    """Positions of ``queries`` in ``sorted_keys`` (searched in sorted order: cache friendly)."""
    order = np.argsort(queries)
    pos = np.empty(len(queries), dtype=np.int64)
    pos[order] = np.searchsorted(sorted_keys, queries[order])
    return pos


def _unmatched(F, nv):
    """Half-edges (``3f+i``) without a proper twin."""
    return _twins(F.ravel(), F[:, [1, 2, 0]].ravel(), nv) < 0


# -----------------------------
# PERIMETER TRIANGULATION
# -----------------------------
def simplify_surface(vertices, perimeters, normal=None):
    """ """
    if normal is None:
        normal = np.array([0, 0, 1])

    sub_verts = vertices[np.concatenate(perimeters)]
    sub_peri = []
    end = 0
    for p in perimeters:
        sub_peri.append(np.arange(len(p)) + end)
        end += len(p)

    _, sub_faces = triangulate_polygon(sub_verts, sub_peri)
    faces = np.concatenate(perimeters)[sub_faces]

    return sub_verts, faces


def triangle_area_3d(p1, p2, p3):
    vector1 = np.array(p2) - np.array(p1)
    vector2 = np.array(p3) - np.array(p1)
    cross_product = np.cross(vector1, vector2)
    area = 0.5 * np.linalg.norm(cross_product)
    return area


def calculate_areas_of_triangles_list(triangles_list):
    areas = np.sum([triangle_area_3d(p1, p2, p3) for p1, p2, p3 in triangles_list])
    return areas
