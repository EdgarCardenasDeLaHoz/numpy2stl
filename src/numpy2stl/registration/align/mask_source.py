"""The mask-producer seam — one swappable place that turns a heightmap into a
building mask.

Every consumer in the registration pipeline that needs "which cells of this STL
heightmap are buildings?" goes through `produce_mask` / `produce_edges` here
instead of calling `building_mask` / `building_edges` directly.  With no
producer installed these are exactly the classic calls, so default behaviour is
unchanged; installing one (see `use_mask_producer`) swaps segmentation for the
whole pipeline without touching registration, which stays classic.

Why a seam at all: measured registration failures split into two very different
causes.  Where the STL segmentation is clean (Barcelona, Paris) the classic
search finds the right transform; where it is wrong (Valencia overlap-IoU 0.025,
Salzburg 0.039, Miami 0.155) no amount of search fixes it.  That makes
segmentation the axis worth replacing, and a single seam makes it replaceable
and measurable (`_align_tool/eval_registration.py` reports segmentation IoU
separately from registration quality for exactly this reason).

OSM masks never route through the producer.  `building_mask(source='osm')` is
literally `~np.isnan(heightmap)` — the rasterized footprints are ground truth,
not an estimate, so there is nothing for a learned model to improve and
everything to corrupt.  Calls with `source='osm'` fall through to the classic
implementation even when a producer is installed.

The active producer is process-wide (thread-local) rather than threaded through
every signature, matching config.py's stated convention that "nothing has to
thread the whole object".  Install it for a bounded scope with the context
manager; there is deliberately no global setter.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Callable, Iterator

import numpy as np

from .segmentation import HAS_CV2, building_edges, building_mask

if HAS_CV2:
    import cv2

__all__ = ["MaskProducer", "active_mask_producer", "use_mask_producer",
           "use_config", "produce_mask", "produce_edges"]

# A producer has building_mask's signature: (heightmap, source=..., **kwargs)
# -> bool ndarray of the same shape.  Keyword arguments it does not understand
# must be accepted and ignored (take **kwargs), because call sites pass the
# classic tuning knobs (cell_size_m, threshold_method, exclude_mask, ...) that a
# learned segmenter has no use for.
MaskProducer = Callable[..., np.ndarray]

_state = threading.local()


def active_mask_producer() -> MaskProducer | None:
    """The producer installed for the current thread, or None (classic)."""
    return getattr(_state, "producer", None)


@contextlib.contextmanager
def use_mask_producer(producer: MaskProducer | None) -> Iterator[None]:
    """Install `producer` for the duration of the block, then restore.

    Passing None inside an outer scope restores classic behaviour for the inner
    block, which is what makes A/B comparisons in one process possible.
    """
    previous = active_mask_producer()
    _state.producer = producer
    try:
        yield
    finally:
        _state.producer = previous


@contextlib.contextmanager
def use_config(config: Any) -> Iterator[None]:
    """Install a `RegistrationConfig`'s mask_producer for the duration."""
    with use_mask_producer(getattr(config, "mask_producer", None)):
        yield


def _validated(mask: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Check a producer's output before the pipeline consumes it.

    A mis-shaped or non-boolean mask fails far downstream (usually as a silent
    broadcast or an all-True frame) where the cause is unrecoverable, so reject
    it here while the producer is still named in the traceback.
    """
    mask = np.asarray(mask)
    if mask.shape != like.shape:
        raise ValueError(
            f"mask producer returned shape {mask.shape}, expected {like.shape}")
    return mask.astype(bool)


def produce_mask(heightmap: np.ndarray, source: str = "stl", **kwargs) -> np.ndarray:
    """`building_mask`, routed through the active producer (STL sources only)."""
    producer = active_mask_producer()
    if producer is None or source == "osm":
        return building_mask(heightmap, source=source, **kwargs)
    return _validated(producer(heightmap, source=source, **kwargs), heightmap)


def produce_edges(heightmap: np.ndarray, source: str = "stl", **kwargs) -> np.ndarray:
    """`building_edges`, routed through the active producer (STL sources only).

    The outline is recomputed here rather than delegated so a producer only ever
    has to return a filled mask.  The erode-and-subtract is duplicated from
    `building_edges` deliberately: the two must stay pixel-identical, and the
    default path below still calls `building_edges` itself so any drift shows up
    as a test failure rather than a silent difference in the search.
    """
    producer = active_mask_producer()
    if producer is None or source == "osm":
        return building_edges(heightmap, source=source, **kwargs)
    mask = produce_mask(heightmap, source=source, **kwargs).astype(np.uint8)
    if not HAS_CV2:
        gx = np.abs(np.diff(mask, axis=1, prepend=0))
        gy = np.abs(np.diff(mask, axis=0, prepend=0))
        return (gx + gy) > 0
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8))
    return (mask - eroded).astype(bool)
