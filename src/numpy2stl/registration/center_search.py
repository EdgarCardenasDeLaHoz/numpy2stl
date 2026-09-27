"""Centre search: find the reference frame whose buildings match a city STL.

``find_best_target`` scores candidate frames (from the reference source's
``candidate_targets``) with a coarse ``register_global`` pass and keeps the best
one that locks.  Generating the candidates (a ring of lon/lat centres for OSM) is
the source's job, so this module stays geo-free.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def find_best_target(
    source,
    stl_hm: np.ndarray,
    candidates: list[tuple[Any, Any]],
    osm_margin: float = 1.5,
    probe_resolution: int = 256,
) -> dict:
    """Score ``candidates`` = [(label, target), ...] and return the best locked one.

    Used when a single guessed centre may be in the wrong part of a
    partial-coverage city model (measured wrong for Barcelona, Paris, Lisbon and
    likely Bilbao with a "Downtown {city}" geocode).  The first candidate is
    conventionally the initial guess.

    Each candidate is scored by a COARSE, cheap register_global() pass at
    ``probe_resolution`` against ``source.building_heightmap(target, ...)``, with
    the scale prior fixed at the geometric anchor 1/osm_margin (the probe does not
    need semantic masks: locking only reads the returned scale_sweep/rot_sweep).
    A candidate is `is_locked` when BOTH its Dice-vs-scale and edge-IoU-vs-rotation
    sweeps show a sharp, non-boundary peak -- see
    stages._common._is_locked_registration(), shared with the pipeline's own
    coarse-probe gate so "locked" means the same thing in both places.

    Among locked candidates, the winner has the highest `edge_iou`.  When NO
    candidate locks this returns `resolved=False` and does not guess a winner from
    unlocked (noisy) candidates -- the caller keeps its initial frame.  A
    per-candidate fetch failure is logged and treated as "not locked".

    Returns
    -------
    dict: {
        "center": label | None,             # winning candidate's label (OSM: (lat, lon))
        "resolved": bool,
        "candidates_tried": int,
        "best_dice_sharpness": float,
        "best_rot_sharpness": float,
        "reg_dict": dict | None,            # winning candidate's register_global() dict
        "osm_fetch_target": Any | None,     # winning candidate's target
    }
    """
    from .align import register_global
    from .config import CENTER_SEARCH_DICE_MARGIN, CENTER_SEARCH_ROT_MARGIN
    from .stages import _is_locked_registration

    name = getattr(source, "name", "reference")
    if not candidates:
        logger.warning("find_best_target(%r): no candidates; cannot search.", name)
        return {
            "center": None, "resolved": False, "candidates_tried": 0,
            "best_dice_sharpness": 0.0, "best_rot_sharpness": 0.0,
            "reg_dict": None, "osm_fetch_target": None,
        }
    initial_label = candidates[0][0]
    logger.info("find_best_target(%r): searching %d candidates", name, len(candidates))

    # Resize the STL heightmap once to the probe resolution (register_global
    # itself will also resize `source` if shapes mismatch, but doing it once up
    # front avoids repeating a possibly-large resize per candidate).
    stl_probe = stl_hm
    if stl_hm.shape != (probe_resolution, probe_resolution):
        import cv2 as _cv2
        _filled = np.nan_to_num(stl_hm.astype(np.float32), nan=0.0)
        stl_probe = _cv2.resize(_filled, (probe_resolution, probe_resolution),
                                interpolation=_cv2.INTER_LINEAR)

    best: dict | None = None
    best_locked: dict | None = None
    for idx, (label, target) in enumerate(candidates):
        offset_desc = "initial" if idx == 0 else f"#{idx} {label}"
        try:
            osm = source.building_heightmap(target, probe_resolution)
            geometric_anchor = 1.0 / osm_margin
            reg_dict = register_global(
                stl_probe, osm["heightmap"], scale_prior=geometric_anchor, scale_search=0.0,
            )
        except Exception as exc:
            logger.warning("  candidate %s: fetch/register failed (%s); treating as not locked",
                           offset_desc, exc)
            continue

        lock = _is_locked_registration(
            reg_dict, dice_margin=CENTER_SEARCH_DICE_MARGIN, rot_margin=CENTER_SEARCH_ROT_MARGIN,
        )
        logger.info("  candidate %s: dice_sharpness=%.3f rot_sharpness=%.3f "
                    "edge_iou=%.3f locked=%s", offset_desc, lock["dice_sharpness"],
                    lock["rot_sharpness"], reg_dict.get("edge_iou", 0.0), lock["is_locked"])

        entry = {
            "center": label, "bbox": target, "reg_dict": reg_dict,
            "dice_sharpness": lock["dice_sharpness"], "rot_sharpness": lock["rot_sharpness"],
            "is_locked": lock["is_locked"], "edge_iou": float(reg_dict.get("edge_iou", 0.0)),
        }
        if best is None or entry["edge_iou"] > best["edge_iou"]:
            best = entry
        if lock["is_locked"] and (best_locked is None or entry["edge_iou"] > best_locked["edge_iou"]):
            best_locked = entry

    candidates_tried = len(candidates)
    if best_locked is None:
        logger.warning("find_best_target(%r): 0/%d candidates locked; keeping the "
                       "initial frame %s.", name, candidates_tried, initial_label)
        return {
            "center": initial_label, "resolved": False, "candidates_tried": candidates_tried,
            "best_dice_sharpness": float(best["dice_sharpness"]) if best else 0.0,
            "best_rot_sharpness": float(best["rot_sharpness"]) if best else 0.0,
            "reg_dict": None, "osm_fetch_target": None,
        }

    logger.info("find_best_target(%r): resolved to %s "
                "(edge_iou=%.3f, dice_sharpness=%.3f, rot_sharpness=%.3f) after %d candidates",
                name, best_locked["center"],
                best_locked["edge_iou"], best_locked["dice_sharpness"],
                best_locked["rot_sharpness"], candidates_tried)
    return {
        "center": best_locked["center"], "resolved": True, "candidates_tried": candidates_tried,
        "best_dice_sharpness": float(best_locked["dice_sharpness"]),
        "best_rot_sharpness": float(best_locked["rot_sharpness"]),
        "reg_dict": best_locked["reg_dict"], "osm_fetch_target": best_locked["bbox"],
    }
