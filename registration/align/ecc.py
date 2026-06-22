"""ECC refinement, coarse-to-fine search, projection discovery.

Part of the align/ subpackage (split from the former align.py).
"""
from __future__ import annotations

import logging
import time
from math import atan2, degrees, sqrt

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from scipy.ndimage import sobel, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    sobel = gaussian_filter = None
    HAS_SCIPY = False

from .metrics import _mask_sdf, _tolerant_iou
from .segmentation import building_edges, building_mask
from .transform import _preprocess_for_registration, apply_transform

def refine_transform(
    source: np.ndarray,
    target: np.ndarray,
    init_transform: np.ndarray,
    signal: str = "edge",
    motion: str = "affine",
    ecc_iterations: int = 500,
    ecc_eps: float = 1e-7,
    blur_sigma: float = 1.5,
) -> dict:
    """
    Refine an existing source→target transform with a second ECC pass.

    Use this to chain stages, e.g. a robust binary-mask alignment followed by
    an edge-based fine pass, or to add rotation / perspective freedom.

    Parameters
    ----------
    source, target : (rows, cols) float64 heightmaps
    init_transform : (2, 3) affine from a prior registration
    signal         : 'edge'  — Sobel gradient image (sharp building outlines)
                     'mask'  — binary footprint signed-distance field
                     'sdf'   — alias for 'mask'
    motion         : 'translation' | 'euclidean' | 'affine' | 'homography'
    ecc_iterations : ECC max iterations
    ecc_eps        : ECC convergence threshold
    blur_sigma     : Gaussian blur on the signal (edge mode only)

    Returns
    -------
    dict: {'transform' (2,3 or 3,3), 'confidence', 'converged', 'motion', 'signal'}
    """
    if not HAS_CV2:
        raise ImportError("opencv-python is required.")

    tgt_h, tgt_w = target.shape

    # Build the signal images
    if signal in ("mask", "sdf"):
        s_img = _mask_sdf(building_mask(source, source="stl"))
        t_img = _mask_sdf(building_mask(target, source="osm"))
    else:  # edge
        s_img = _preprocess_for_registration(source).astype(np.float32)
        t_img = _preprocess_for_registration(target).astype(np.float32)
        if blur_sigma > 0:
            k = max(3, int(blur_sigma * 3) | 1)
            s_img = cv2.GaussianBlur(s_img, (k, k), blur_sigma)
            t_img = cv2.GaussianBlur(t_img, (k, k), blur_sigma)

    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
        "homography": cv2.MOTION_HOMOGRAPHY,
    }
    motion_type = motion_map[motion]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, ecc_eps)

    if motion == "homography":
        warp = np.eye(3, dtype=np.float32)
        warp[:2, :] = init_transform.astype(np.float32)
    else:
        warp = init_transform.astype(np.float32).copy()

    cc, converged = 0.0, False
    try:
        cc, warp = cv2.findTransformECC(
            templateImage=t_img, inputImage=s_img,
            warpMatrix=warp, motionType=motion_type, criteria=criteria,
        )
        if not isinstance(cc, float):
            warp = cc
            cc = 0.0
        converged = True
    except cv2.error as e:
        logger.warning("refine ECC (%s/%s) did not converge: %s", signal, motion, e)
        warp = (np.eye(3, dtype=np.float32) if motion == "homography"
                else init_transform.astype(np.float32))

    return {
        "transform": warp.astype(np.float64),
        "confidence": float(cc),
        "converged": converged,
        "motion": motion,
        "signal": signal,
    }


def discover_projection(
    source: np.ndarray,
    target: np.ndarray,
    base_transform: np.ndarray,
    candidates: tuple = ("affine", "euclidean", "homography"),
    cv_fraction: float = 0.5,
) -> dict:
    """
    Pick the projection model the *data* actually supports — per run, not baked in.

    Different STL sources can have different projections relative to the OSM map
    (none, slight affine skew, or true perspective). A higher-DOF model always
    fits the training footprints at least as well, so choosing by raw IoU would
    always pick homography and risk overfitting. Instead we **cross-validate**:
    fit each candidate's refinement, then score it on a held-out half of the
    target footprints. The model that generalizes best wins.

    Parameters
    ----------
    source, target : (rows, cols) heightmaps
    base_transform : (2,3) affine from the coarse base registration
    candidates     : projection models to try, in increasing DOF
    cv_fraction    : fraction of target columns held out for validation

    Returns
    -------
    dict:
        'transform'    : best transform (2,3 or 3,3)
        'projection'   : name of the chosen model
        'train_iou'    : IoU on the fit region
        'val_iou'      : IoU on the held-out region (the deciding score)
        'all'          : list of per-candidate {projection, train_iou, val_iou}
    """
    h, w = target.shape

    # Hold out a contiguous vertical band of the target for validation.
    val_w = max(8, int(w * cv_fraction))
    val_c0 = (w - val_w) // 2
    val_cols = np.zeros(w, dtype=bool)
    val_cols[val_c0:val_c0 + val_w] = True

    # Score on EDGE masks — sparse outlines with a low random baseline, so the
    # cross-validation reflects genuine alignment, not dense blob overlap.
    t_edge = building_edges(target, source="osm")

    def _split_iou(M):
        aligned = apply_transform(source, M, output_shape=target.shape)
        s_edge = building_edges(aligned, source="stl")
        out = {}
        for name, colsel in (("train", ~val_cols), ("val", val_cols)):
            out[name] = _tolerant_iou(s_edge[:, colsel], t_edge[:, colsel], tol_px=2)
        return out

    results = []
    # Always include the base (no extra refinement) as a candidate.
    base_scores = _split_iou(base_transform)
    results.append({
        "projection": "base", "transform": base_transform,
        "train_iou": base_scores["train"], "val_iou": base_scores["val"],
    })

    for motion in candidates:
        r = refine_transform(source, target, base_transform,
                             signal="edge", motion=motion)
        sc = _split_iou(r["transform"])
        results.append({
            "projection": motion, "transform": r["transform"],
            "train_iou": sc["train"], "val_iou": sc["val"],
        })

    # Choose by held-out (validation) IoU — the model that generalizes.
    best = max(results, key=lambda r: r["val_iou"])

    for r in results:
        logger.info("  projection %-11s train=%.3f  val=%.3f%s",
                    r["projection"], r["train_iou"], r["val_iou"],
                    "  <- chosen" if r is best else "")

    return {
        "transform": best["transform"],
        "projection": best["projection"],
        "train_iou": best["train_iou"],
        "val_iou": best["val_iou"],
        "all": [{k: r[k] for k in ("projection", "train_iou", "val_iou")} for r in results],
    }
