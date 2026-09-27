"""Where numpy2stl writes caches and reports, defined once.

    NUMPY2STL_CACHE    compute caches (STL heightmaps).
                       Default: registration/runs/ inside the package tree, gitignored.
    NUMPY2STL_REPORTS  generated registration reports.
                       Default: the shared, gitignored Code/_reports/ beside the repos.
"""

import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent            # src/numpy2stl/
_WORKSPACE = _PACKAGE.parents[2]                      # Code/ (holds numpy2stl/ and map2stl/)

CACHE_ROOT = Path(os.environ.get("NUMPY2STL_CACHE") or _PACKAGE / "registration" / "runs")
REPORTS_ROOT = Path(os.environ.get("NUMPY2STL_REPORTS") or _WORKSPACE / "_reports")
