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
