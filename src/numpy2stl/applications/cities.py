"""Removed: OSM fetching left numpy2stl (numpy2stl is geo-free).

The OSM building-height / semantic rasters and the city helpers
(``get_osm_building_heightmap``, ``get_osm_semantic_masks``, ``get_city_bbox``,
``get_city_center_point``, ``estimate_bbox_from_stl``, ``tight_bbox_from_extent``,
``derive_scale_m_per_unit``, ``get_philadelphia_heightmap``) live in map2stl's
``city2stl.osm_raster``; registering against a city name is
``city2stl.registration.register_city_stl``.  numpy2stl never imports map2stl, so
this module cannot forward to them; importing it raises this ImportError for one
release, then it goes.
"""

raise ImportError(
    "numpy2stl.applications.cities was removed (numpy2stl is geo-free). Use map2stl's "
    "city2stl.osm_raster (OSM rasters, city bbox/centre helpers) and "
    "city2stl.registration.register_city_stl (register an STL against a city name or bbox)."
)
