"""Registration pipeline stages, extracted from the monolithic ``pipeline.py``.

The orchestrator (``register_city_stl``) imports these stage helpers; they are
not part of the public API but are grouped here for navigability.
"""
from .simplify import _simplify_stage
from .registration import _polygon_register_dict, _run_registration
from .compare import _run_comparison

__all__ = [
    "_simplify_stage",
    "_polygon_register_dict",
    "_run_registration",
    "_run_comparison",
]
