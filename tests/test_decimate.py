"""processing.decimate: error-bounded reduction."""

import numpy as np

from numpy2stl.processing.decimate import heightfield_tin


def _interp_error(z, idx, tris):
    import matplotlib.tri as mtri

    h, w = z.shape
    ii, jj = np.divmod(idx, w)
    remap = np.full(h * w, -1)
    remap[idx] = np.arange(len(idx))
    t = mtri.Triangulation(jj.astype(float), ii.astype(float), remap[tris])
    yy, xx = np.mgrid[0:h, 0:w]
    zi = mtri.LinearTriInterpolator(t, z.ravel()[idx])(xx, yy)
    return np.nanmax(np.abs(zi - z))


def test_tin_stays_within_the_error_bound():
    y, x = np.mgrid[0:90, 0:120]
    z = 5 + 20 * np.exp(-(((x - 60) / 20) ** 2 + ((y - 45) / 15) ** 2)) + 0.3 * np.sin(x / 3)
    idx, tris = heightfield_tin(z, max_error=0.1)
    assert _interp_error(z, idx, tris) <= 0.1 + 1e-9
    assert len(idx) < z.size / 2


def test_flat_field_needs_only_the_border_and_seeds():
    z = np.full((64, 80), 3.0)
    idx, tris = heightfield_tin(z, max_error=0.05, seed_step=8)
    border = 2 * 64 + 2 * 80 - 4
    assert len(idx) <= border + (64 // 8 + 1) * (80 // 8 + 1)


def test_budget_raises_the_bound_until_the_mesh_fits():
    from numpy2stl.processing.decimate import heightfield_tin_budget

    y, x = np.mgrid[0:120, 0:160]
    z = 5 + 20 * np.exp(-(((x - 80) / 25) ** 2 + ((y - 60) / 20) ** 2)) + 0.5 * np.sin(x / 2.5)
    full, _ = heightfield_tin(z, max_error=0.02)
    idx, tris, err = heightfield_tin_budget(z, 0.02, max_vertices=len(full) // 3)
    assert len(idx) <= len(full) // 3
    assert err > 0.02
    assert _interp_error(z, idx, tris) <= err + 1e-9
    # A budget the exact mesh fits returns the exact mesh's bound.
    idx2, _, err2 = heightfield_tin_budget(z, 0.02, max_vertices=10 * len(full))
    assert err2 <= 0.02 and len(idx2) <= len(full) * 1.05


def test_tiled_tin_keeps_the_bound_and_conforming_seams():
    y, x = np.mgrid[0:130, 0:170]
    z = 5 + 20 * np.exp(-(((x - 80) / 25) ** 2 + ((y - 60) / 20) ** 2)) + 0.5 * np.sin(x / 2.5)
    idx, tris = heightfield_tin(z, max_error=0.05, tile=48)   # 3 x 4 tiles, odd remainders
    assert _interp_error(z, idx, tris) <= 0.05 + 1e-9
    # conforming: every edge is used by one (outer border) or two triangles, and the
    # triangles cover the grid exactly once (areas add up to the rectangle)
    e = np.sort(tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, uses = np.unique(e[:, 0] * z.size + e[:, 1], return_counts=True)
    assert uses.max() == 2
    ii, jj = np.divmod(tris, z.shape[1])
    area = 0.5 * np.abs((jj[:, 1] - jj[:, 0]) * (ii[:, 2] - ii[:, 0])
                        - (jj[:, 2] - jj[:, 0]) * (ii[:, 1] - ii[:, 0]))
    assert area.sum() == (z.shape[0] - 1) * (z.shape[1] - 1)
    untiled, _ = heightfield_tin(z, max_error=0.05, tile=None)
    assert len(idx) < 1.2 * len(untiled)
