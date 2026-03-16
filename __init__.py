"""Compatibility shim for `strm2stl.numpy2stl` package.

This package directory contains a nested `numpy2stl` module; re-export
its public API at the expected import path.
"""
import sys
import importlib
from .numpy2stl import *
import os

# Ensure submodule imports like `strm2stl.numpy2stl.oceans` resolve to the
# nested `numpy2stl` directory by adding it to this package's __path__.
_inner_path = os.path.join(os.path.dirname(__file__), 'numpy2stl')
if os.path.isdir(_inner_path) and _inner_path not in __path__:
    __path__.insert(0, _inner_path)

# Ensure the inner package is importable via sys.modules for dotted imports
try:
    inner_mod = importlib.import_module(f"{__name__}.numpy2stl")
    sys.modules[f"{__name__}.numpy2stl"] = inner_mod
    # Also ensure common submodule aliases resolve for tests/imports
    try:
        oceans_mod = importlib.import_module(f"{__name__}.numpy2stl.oceans")
        sys.modules[f"{__name__}.numpy2stl.oceans"] = oceans_mod
        sys.modules[f"{__name__}.oceans"] = oceans_mod
    except Exception:
        pass
except Exception:
    pass
