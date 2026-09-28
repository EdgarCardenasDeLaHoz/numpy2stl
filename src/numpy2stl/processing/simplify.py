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
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

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
        orientation of the faces it replaces. Untouched faces are returned as
        given; coincident vertices are merged only internally, so copies the
        caller kept apart (solids touching at a point) stay apart.

    Notes
    -----
    - Regions are grouped by (normal, plane offset) and must be edge-connected;
      two coplanar patches that only touch at a vertex stay separate.
    - A boundary vertex is only dropped when exactly two regions meet at it
      and it is collinear with its neighbours on the shared edge, so both
      regions drop it together and no T-junction appears.
    - Each region's boundary is chained into rings (counter-clockwise shells,
      clockwise holes) in its plane, and GEOS's constrained Delaunay
      triangulation (``shapely.constrained_delaunay_triangles``) fills the
      polygon using its boundary vertices only. Holes (islands in a flat sea,
      courtyards in a roof), collinear boundary vertices and regions touching
      themselves at a vertex are handled. All regions go to GEOS in one
      vectorised call.
    - A region that cannot be triangulated exactly (invalid polygon, area
      mismatch, flipped triangle, or an edge that no longer meets its
      neighbour) keeps its original faces, and its neighbours are redone
      keeping every vertex they share with it.
    """
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    if len(F) == 0:
        return F.copy()

    t0 = time.perf_counter()
    F_in = F
    F = _merge_coincident(V, F)
    used = np.zeros(len(V), dtype=bool)
    used[F.ravel()] = True
    scale = float(np.linalg.norm(np.ptp(V[used], axis=0))) or 1.0

    mesh = _HalfEdges(V, F, scale, tol)
    labels, candidate = mesh.planar_regions(min_faces)
    mesh.region_boundaries(labels)
    t1 = time.perf_counter()

    was_open = (mesh.twin < 0).reshape(-1, 3)
    n_retry = 0
    new, new_reg, redo, removed = None, None, None, None
    while True:
        prev_removed = removed
        removed, segments = mesh.collapse_boundaries(labels, candidate)
        if prev_removed is not None:
            # Only regions around vertices whose fate changed need new triangles.
            changed = removed != prev_removed
            redo = np.zeros(len(candidate), dtype=bool)
            redo[labels[changed[F].any(axis=1)]] = True
            keep = candidate[new_reg] & ~redo[new_reg]
            new, new_reg = new[keep], new_reg[keep]
        add, add_reg, failed = _retriangulate(mesh, labels, candidate, removed, segments, redo)
        if new is None:
            new, new_reg = add, add_reg
        else:
            new, new_reg = np.vstack([new, add]), np.concatenate([new_reg, add_reg])
        touched = np.zeros(len(candidate), dtype=bool)
        touched[new_reg] = True
        kept = np.flatnonzero(~touched[labels])
        out = np.vstack([F[kept], new])
        if not failed:
            near = np.zeros(mesh.nv, dtype=bool)
            near[F[touched[labels]].ravel()] = True
            failed = _cracked_regions(out, len(kept), new_reg, was_open[kept], segments, near)
        if not failed:
            break
        # A region that could not be retriangulated keeps its faces, so its
        # neighbours must keep every vertex they share with it.
        candidate[failed] = False
        n_retry += 1
        logger.warning(f"simplify: {len(failed)} region(s) kept as-is, retrying")

    if F is not F_in:
        out = _unmerge(F_in, F, labels, kept, new, new_reg)

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


def _unmerge(F_in, F, labels, kept, new, new_reg):
    """Faces on the caller's own vertex indices where positions were merged.

    Kept faces get their input faces back; a new triangle takes, per corner,
    the index its region's input faces used at that position. Coincident
    copies (a touching point the caller kept apart) thus stay apart.
    """
    nv = int(F_in.max()) + 1
    replaced = np.zeros(int(labels.max()) + 1, dtype=bool)
    replaced[new_reg] = True
    faces = np.flatnonzero(replaced[labels])
    key = np.repeat(labels[faces], 3) * nv + F[faces].ravel()
    key, first = np.unique(key, return_index=True)
    orig = F_in[faces].ravel()[first]
    pos = np.searchsorted(key, new_reg[:, None] * nv + new)
    return np.vstack([F_in[kept], orig[pos]])


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

    def region_boundaries(self, labels):
        """Boundary half-edges of the regions and the vertices that could go.

        Independent of which regions are candidates, so it runs once; each
        retry only re-applies :meth:`collapse_boundaries` (which must then be
        called with a shrinking set of candidates).
        """
        V, nv = self.V, self.nv
        reg_he = labels[self.face]
        tw = self.twin
        boundary = (tw < 0) | (reg_he != reg_he[np.maximum(tw, 0)])
        self.boundary = boundary

        # Around a vertex, each outgoing boundary half-edge starts a new sector
        # of the fan: none means the vertex is inside one region.
        n_bout = np.bincount(self.a[boundary], minlength=nv)
        removable = n_bout == 0

        # Straight-edge vertices between exactly two regions.
        bh = np.flatnonzero(boundary)
        bh = bh[np.argsort(self.a[bh], kind="stable")]
        starts = self.a[bh]
        two = n_bout == 2
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
            removable[v[straight]] = True
        self.removable = removable
        self._protected = np.zeros(nv, dtype=bool)
        open_he = tw < 0
        self._protected[self.a[open_he]] = True
        self._protected[self.b[open_he]] = True
        self._was_candidate = np.ones(int(labels.max()) + 1, dtype=bool)

        # Oriented boundary segments, and for each the segment of the same
        # region leaving its end vertex (unique wherever that vertex can go:
        # a removable vertex has one outgoing boundary edge per region).
        bh = np.flatnonzero(boundary)
        s, e, r = self.a[bh], self.b[bh], reg_he[bh]
        key = r * nv + s
        order = np.argsort(key)
        sk = np.r_[key[order], -1]  # sentinel: a miss lands on a non-matching key
        q = r * nv + e
        pos = np.searchsorted(sk[:-1], q)
        self.seg = (s, e, r)
        self.seg_next = np.where(sk[pos] == q, order[np.minimum(pos, len(order) - 1)], -1)

    def collapse_boundaries(self, labels, candidate):
        """Pick removable vertices and chain each region's boundary past them.

        Returns ``removed`` (bool per vertex) and ``segments`` = (start, end,
        region) arrays: the oriented boundary edges of the candidate regions
        after the removed vertices are skipped.
        """
        # Vertices of kept (non-candidate) regions and of open edges stay.
        # Candidates only ever drop out, so extend the previous call's set.
        dropped = self._was_candidate & ~candidate
        self._protected[self.F[dropped[labels]].ravel()] = True
        self._was_candidate = candidate.copy()
        removed = self.removable & ~self._protected

        # Chain the candidate regions' boundary edges past removed vertices by
        # pointer jumping along ``seg_next``.
        s, e, r = self.seg
        follow = removed[e] & (self.seg_next >= 0) & candidate[r]
        succ = np.arange(len(s))
        act = np.flatnonzero(follow)
        succ[act] = self.seg_next[act]
        for _ in range(64):  # 2**64 steps: more than any chain
            act = act[succ[succ[act]] != succ[act]]
            if len(act) == 0:
                break
            succ[act] = succ[succ[act]]
        e = e[succ]
        e = np.where(removed[e], s, e)  # broken chain: region will fail its checks
        keep = ~removed[s] & candidate[r]
        return removed, (s[keep], e[keep], r[keep])


# -----------------------------
# RETRIANGULATION
# -----------------------------
def _retriangulate(mesh, labels, candidate, removed, segments, only=None):
    """Triangulate every candidate region that lost a vertex (and is in ``only``).

    Returns ``(faces, region, failed)``: the new triangles (global vertex
    indices), the region each replaces, and the regions that could not be
    triangulated exactly (their triangles are not in ``faces``).
    """
    F, V, nv = mesh.F, mesh.V, mesh.nv
    n_reg = len(candidate)
    look = candidate if only is None else candidate & only
    faces = np.flatnonzero(look[labels])
    in_todo = np.zeros(n_reg, dtype=bool)
    in_todo[labels[faces[removed[F[faces]].any(axis=1)]]] = True
    empty = np.zeros((0, 3), dtype=np.int64)
    if not in_todo.any():
        return empty, np.zeros(0, dtype=np.int64), []

    # Dropping the dominant axis keeps coordinates exact; the swap for downward
    # normals makes CCW in 2-D mean "facing the normal".
    normal = mesh.ref_normal
    k = np.argmax(np.abs(normal), axis=1)
    flip = normal[np.arange(n_reg), k] < 0
    ax0 = np.where(flip, (k + 2) % 3, (k + 1) % 3)
    ax1 = np.where(flip, (k + 1) % 3, (k + 2) % 3)

    s, e, r = segments
    m = in_todo[r]
    s, e, r = s[m], e[m], r[m]
    bad = np.zeros(n_reg, dtype=bool)
    bad[r[s == e]] = True  # broken chain
    xy_s = np.stack([V[s, ax0[r]], V[s, ax1[r]]], axis=1) + 0.0  # -0.0 -> 0.0
    xy_e = np.stack([V[e, ax0[r]], V[e, ax1[r]]], axis=1) + 0.0

    nxt, pinches = _link_boundary(s, e, r, xy_s, xy_e, nv, bad)
    new, new_reg = _triangulate_regions(s, r, xy_s, xy_e, nxt, pinches, bad, n_reg)

    tf = np.flatnonzero(in_todo[labels])
    fr = labels[tf]
    # Same signed area (along the region normal) and no flipped or flat triangles.
    area_new = _area_along(V[new], normal[new_reg])
    area_old = _area_along(V[F[tf]], normal[fr])
    sum_new = np.bincount(new_reg, weights=area_new, minlength=n_reg)
    sum_old = np.bincount(fr, weights=area_old, minlength=n_reg)
    bad |= np.abs(sum_new - sum_old) > 1e-9 * np.abs(sum_old)
    bad[new_reg[area_new <= 0]] = True
    bad &= in_todo
    keep = ~bad[new_reg]
    return new[keep], new_reg[keep], np.flatnonzero(bad).tolist()


def _link_boundary(s, e, r, xy_s, xy_e, nv, bad):
    """Index of the segment following each boundary segment of its region.

    Where a region touches itself at a vertex (several segments leave it),
    each incoming segment continues along the outgoing one that closes the
    same angular sector, so no two rings cross there. Returns ``(nxt,
    pinches)``, ``pinches`` being the incoming segments of each such vertex.
    Regions whose boundary does not close are flagged in ``bad``; their links
    are meaningless.
    """
    n = len(s)
    key = r * nv + s
    order = np.argsort(key, kind="stable")
    sk = key[order]
    q = r * nv + e
    lo = np.searchsorted(sk, q)
    hi = np.searchsorted(sk, q, side="right")
    cnt = hi - lo
    nxt = np.arange(n)
    one = cnt == 1
    nxt[one] = order[lo[one]]
    bad[r[cnt == 0]] = True

    multi = np.flatnonzero(cnt > 1)
    pinches = []
    if len(multi):
        multi = multi[np.argsort(q[multi], kind="stable")]
        cuts = np.flatnonzero(np.diff(q[multi])) + 1
        for grp in np.split(multi, cuts):
            outs = order[lo[grp[0]] : hi[grp[0]]]
            if len(outs) != len(grp):
                bad[r[grp[0]]] = True
                continue
            v = xy_s[outs[0]]
            a_in = np.arctan2(*(xy_s[grp] - v).T[::-1])
            a_out = np.arctan2(*(xy_e[outs] - v).T[::-1])
            # First outgoing segment clockwise from the reversed incoming one.
            gap = np.mod(a_in[:, None] - a_out[None, :], 2 * np.pi)
            gap[gap == 0] = 2 * np.pi
            pick = np.argmin(gap, axis=1)
            if len(np.unique(pick)) != len(pick):
                bad[r[grp[0]]] = True
                continue
            nxt[grp] = outs[pick]
            pinches.append(grp)
    return nxt, pinches


def _triangulate_regions(s, r, xy_s, xy_e, nxt, pinches, bad, n_reg):
    """Constrained Delaunay triangulation (GEOS) of every linked region.

    Each region's boundary segments form rings: counter-clockwise shells and
    clockwise holes. GEOS triangulates the polygons without adding vertices;
    every triangle corner is mapped back to its input vertex by its exact
    coordinates. Regions that cannot be built or triangulated are flagged in
    ``bad``. Returns ``(faces, region)`` with faces wound counter-clockwise
    in 2-D (i.e. facing the region normal).
    """
    empty = np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    keep = np.flatnonzero(~bad[r])
    if len(keep) == 0:
        return empty
    remap = np.full(len(s), -1)
    remap[keep] = np.arange(len(keep))
    s, r, xy_s, xy_e = s[keep], r[keep], xy_s[keep], xy_e[keep]
    nxt = remap[nxt[keep]]
    n = len(s)  # links stay within a region, so ``nxt`` has no -1 left
    pinches = [remap[g] for g in pinches if remap[g[0]] >= 0]

    # Rings = cycles of the permutation ``nxt``.
    idx = np.arange(n)
    ring = _cycles(nxt)
    if pinches:
        nxt, ring = _split_touching_rings(nxt, ring, pinches, r, bad)
    # Rank each segment in its ring by pointer jumping from the ring's lowest index.
    _, root = np.unique(ring, return_index=True)
    last = nxt == root[ring]
    succ = np.where(last, idx, nxt)
    dist = (~last).astype(np.int64)
    while True:
        s2 = succ[succ]
        if np.array_equal(s2, succ):
            break
        dist += dist[succ]
        succ = s2
    ring_len = np.bincount(ring)
    pos = ring_len[ring] - 1 - dist
    ring_reg = r[root]

    # Signed ring areas (shoelace, relative to the ring's first vertex).
    o = xy_s[root[ring]]
    a, b = xy_s - o, xy_e - o
    ring_area = np.bincount(ring, weights=a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    tiny = np.abs(ring_area) <= 1e-12 * np.bincount(ring, weights=np.abs(a).sum(axis=1)) ** 2
    bad[ring_reg[(ring_len < 3) | tiny]] = True
    # An edge-connected region has one outer ring (counter-clockwise) and any
    # number of holes; anything else keeps its faces.
    shell = ring_area > 0
    n_shell = np.bincount(ring_reg[shell], minlength=n_reg)
    bad[ring_reg[n_shell[ring_reg] != 1]] = True
    use = np.flatnonzero(~bad[ring_reg])
    if len(use) == 0:
        return empty
    # One polygon per region: its rings in order, shell first.
    use = use[np.lexsort((~shell[use], ring_reg[use]))]
    rank = np.full(len(root), -1)
    rank[use] = np.arange(len(use))
    seg_use = np.flatnonzero(rank[ring] >= 0)
    seg_use = seg_use[np.lexsort((pos[seg_use], rank[ring[seg_use]]))]
    rings = shapely.linearrings(xy_s[seg_use], indices=rank[ring[seg_use]])
    poly_reg, poly_idx = np.unique(ring_reg[use], return_inverse=True)
    polys = shapely.polygons(rings, indices=poly_idx)

    valid = shapely.is_valid(polys)
    bad[poly_reg[~valid]] = True
    ok = np.flatnonzero(valid)
    tri_geoms = np.empty(len(polys), dtype=object)
    try:
        tri_geoms[ok] = shapely.constrained_delaunay_triangles(polys[ok])
    except Exception:  # one bad polygon aborts the batch: redo them one by one
        for i in ok:
            try:
                tri_geoms[i] = shapely.constrained_delaunay_triangles(polys[i])
            except Exception:
                bad[poly_reg[i]] = True
    ok = ok[~bad[poly_reg[ok]]]
    parts, part_poly = shapely.get_parts(tri_geoms[ok], return_index=True)
    corner = shapely.get_coordinates(parts)
    if len(corner) != 4 * len(parts):
        # GEOS always returns closed 4-point triangle rings; anything else is unusable.
        bad[poly_reg[ok]] = True
        return empty
    corner = corner.reshape(-1, 4, 2)[:, :3] + 0.0
    tri_reg = poly_reg[ok][part_poly]

    # Corners back to vertex indices: exact (region, x, y) lookup.
    kdt = np.dtype((np.void, 24))
    known = np.column_stack([r.astype(np.float64), xy_s])
    kv = np.ascontiguousarray(known).view(kdt).ravel()
    korder = np.argsort(kv)
    kv = kv[korder]
    query = np.column_stack([np.repeat(tri_reg.astype(np.float64), 3), corner.reshape(-1, 2)])
    qv = np.ascontiguousarray(query).view(kdt).ravel()
    pos_q = np.minimum(np.searchsorted(kv, qv), len(kv) - 1)
    hit = (kv[pos_q] == qv).reshape(-1, 3).all(axis=1)
    bad[tri_reg[~hit]] = True
    tri = s[korder[pos_q]].reshape(-1, 3)

    # Wind every triangle counter-clockwise in 2-D.
    d1 = corner[:, 1] - corner[:, 0]
    d2 = corner[:, 2] - corner[:, 0]
    cw = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] < 0
    tri[cw] = tri[cw][:, ::-1]
    good = ~bad[tri_reg]
    return tri[good], tri_reg[good]


def _cycles(nxt):
    """Cycle label of each element of the permutation ``nxt``."""
    n = len(nxt)
    graph = coo_matrix((np.ones(n, dtype=np.int8), (np.arange(n), nxt)), shape=(n, n))
    return connected_components(graph, directed=True, connection="weak")[1]


def _split_touching_rings(nxt, ring, pinches, r, bad):
    """Split rings that pass twice through a pinch vertex into two rings.

    Sector pairing never makes rings cross, but a ring can still touch itself
    (e.g. a hole meeting the outer boundary at a vertex), which GEOS rejects.
    Swapping the continuations of two visits of the same ring cuts it at the
    touching vertex into two simple rings (a shell and a hole, or two shells).
    """
    nxt = nxt.copy()
    for _ in range(64):
        used, changed = set(), False
        for grp in pinches:
            lab = ring[grp]
            if len(np.unique(lab)) == len(lab):
                continue
            i, j = next(
                (i, j) for i in range(len(lab)) for j in range(i + 1, len(lab)) if lab[i] == lab[j]
            )
            if lab[i] in used:  # the ring changes this round: revisit next round
                changed = True
                continue
            used.add(lab[i])
            a, b = grp[i], grp[j]
            nxt[a], nxt[b] = nxt[b], nxt[a]
            changed = True
        if not changed:
            return nxt, ring
        ring = _cycles(nxt)
    for grp in pinches:  # did not settle: give up on these regions
        if len(np.unique(ring[grp])) != len(grp):
            bad[r[grp[0]]] = True
    return nxt, ring


def _area_along(tri, normal):
    """Triangle areas signed by agreement with ``normal``; ``tri`` is (k, 3, 3)."""
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.einsum("ij,ij->i", cross, normal)


def _cracked_regions(out, n_kept, new_reg, kept_was_open, segments, near):
    """Regions to blame for edges of ``out`` that lost their twin.

    ``out`` is the kept faces followed by the new ones. Every edge that was
    closed must still be closed; the culprit is the new region owning the
    open half-edge, or the region whose boundary segment should have met it.
    Only edges between two ``near`` vertices (those of the replaced faces) can
    have changed: every other edge keeps both its faces, so it is not checked.
    """
    nv = len(near)
    a_all, b_all = out.ravel(), out[:, [1, 2, 0]].ravel()
    check = np.flatnonzero(near[a_all] & near[b_all])
    open_now = np.zeros(len(a_all), dtype=bool)
    open_now[check] = _twins(a_all[check], b_all[check], nv) < 0
    open_now[: 3 * n_kept] &= ~kept_was_open.ravel()
    crack = np.flatnonzero(open_now)
    if len(crack) == 0:
        return []
    he_face = crack // 3
    blamed = set(new_reg[he_face[he_face >= n_kept] - n_kept].tolist())
    a, b = a_all[crack], b_all[crack]
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
