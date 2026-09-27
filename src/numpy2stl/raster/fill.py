"""NaN fill for height rasters.

``fill_nan(arr, method=...)`` is the one NaN-fill helper for numpy2stl:

- ``"nearest"``  copy the nearest valid cell (scipy EDT); ``interior_only=True``
                 leaves NaN regions that touch the border alone (isotropic
                 padding, water outside the model).  Falls back to ``"median"``
                 without scipy.
- ``"median"``   the median of the valid cells.
- ``"constant"`` ``value`` (default 0).
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def fill_nan(
    arr: np.ndarray,
    method: str = "nearest",
    value: float = 0.0,
    interior_only: bool = False,
) -> np.ndarray:
    """Return ``arr`` with its NaN cells filled.

    Parameters
    ----------
    arr : 2-D float array (any float dtype; the dtype is kept).
    method : {'nearest', 'median', 'constant'}
    value : fill value for ``method='constant'``.
    interior_only : ``'nearest'`` only — fill just the NaN regions that do not
        touch the array border (8-connected); border-connected NaN stays NaN.

    Returns
    -------
    A filled copy, or ``arr`` itself when it has no NaN.  An all-NaN array
    comes back unchanged for ``'median'`` and ``'nearest'``.
    """
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    if method == "constant":
        filled = arr.copy()
        filled[nan_mask] = value
        return filled
    if method not in ("nearest", "median"):
        raise ValueError(f"method must be 'nearest', 'median' or 'constant', got {method!r}")
    if nan_mask.all():
        return arr

    if method == "nearest":
        try:
            from scipy.ndimage import distance_transform_edt, label
        except ImportError:
            method = "median"   # scipy not available — median fill as last resort
        else:
            target = nan_mask
            if interior_only:
                lbl, _ = label(nan_mask, structure=np.ones((3, 3), dtype=int))
                border_ids = set(np.unique(np.concatenate([
                    lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])).tolist())
                border_ids.discard(0)
                if border_ids:
                    target = nan_mask & ~np.isin(lbl, list(border_ids))
            filled = arr.copy()
            if target.any():
                _, idx = distance_transform_edt(nan_mask, return_indices=True)
                filled[target] = arr[idx[0][target], idx[1][target]]
            logger.debug("fill_nan(nearest): filled %d px; left %d border-connected px",
                         int(target.sum()), int((nan_mask & ~target).sum()))
            return filled

    filled = arr.copy()
    filled[nan_mask] = np.nanmedian(arr)
    return filled
