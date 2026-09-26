"""numpy2stl.registration — city STL to OSM image registration pipeline.

Quick start::

    from numpy2stl.registration import register_city_stl
    report = register_city_stl("philadelphia.stl", "Philadelphia, PA, USA",
                                out_dir="./report")

Step-by-step::

    from numpy2stl.registration.align import register, apply_transform
    from numpy2stl.registration.compare import compare
    from numpy2stl.registration.html_report import write_registration_report

This package `__init__` is a thin façade: the orchestrator and its stage helpers
live in `pipeline.py`; the comparison, transforms, report and types live in their
own modules (`compare`, `align`, `html_report`, `types`).
"""
from __future__ import annotations

from .align import apply_transform, register
from .compare import compare
from .html_report import write_registration_report
from .pipeline import RUNS_DIR, register_city_stl
from .types import CityRegistrationReport, ComparisonResult, RegistrationResult

__all__ = [
    "register_city_stl",
    "register",
    "apply_transform",
    "compare",
    "write_registration_report",
    "CityRegistrationReport",
    "RegistrationResult",
    "ComparisonResult",
    "RUNS_DIR",
]
