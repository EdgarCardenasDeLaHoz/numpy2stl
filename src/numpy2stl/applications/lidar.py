"""Removed: 3DEP lidar nDSM fetching left numpy2stl (numpy2stl is geo-free).

``get_ndsm`` lives in strm2stl's ``city2stl.height.providers.lidar_3dep_ept``.
numpy2stl never imports strm2stl, so this module cannot forward to it; importing it
raises this ImportError for one release, then it goes.
"""

raise ImportError(
    "numpy2stl.applications.lidar was removed (numpy2stl is geo-free). Use strm2stl's "
    "city2stl.height.providers.lidar_3dep_ept.get_ndsm."
)
