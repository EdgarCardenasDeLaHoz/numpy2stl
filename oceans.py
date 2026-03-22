"""
Ocean and Terrain Processing Module

This module contains functions for processing geographic elevation data
into 3D terrain models, extracted and refactored from the Oceans.ipynb notebook.
"""

from .solid import triangles_to_facets
from .save import writeSTL
from .generate import numpy2stl
from .tools import resize_max, rescale
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import filters, io
from typing import Tuple, Optional, List, Union
import os
from pathlib import Path

# Import from existing modules
import sys
import os

# Add the strm2stl directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
strm2stl_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if strm2stl_dir not in sys.path:
    sys.path.insert(0, strm2stl_dir)

# Import geo2stl separately - it doesn't require ee
try:
    from geo2stl import geo2stl as g2s
except ImportError as e:
    import importlib
    try:
        g2s = importlib.import_module('strm2stl.geo2stl.geo2stl')
    except Exception:
        try:
            from strm2stl.geo2stl import geo2stl as g2s
        except Exception as e2:
            print(
                f"Warning: Could not import geo2stl.geo2stl: {e}; fallback failed: {e2}")
            g2s = None

# Import sat2stl separately - requires ee (Google Earth Engine)
try:
    from geo2stl.sat2stl import get_aquatic_regions
except ImportError as e:
    try:
        from strm2stl.geo2stl.sat2stl import get_aquatic_regions
    except Exception as e2:
        print(
            f"Warning: Could not import sat2stl (requires ee): {e}; fallback failed: {e2}")
        get_aquatic_regions = None

try:
    from .view import view3D_napari
except ImportError:
    view3D_napari = None


def make_dem_image(
    target_bbox: Tuple[float, float, float, float],
    dim: int = 600,
    depth_scale: float = 0.5,
    sat_scale: float = 400,
    water_scale: float = 0.1,
    base: float = 0.1,
    height: float = 25,
    subtract_water: bool = True,
    clip: Optional[List[float]] = None,
    smooth: Optional[int] = None,
    projection: str = 'cosine',
    maintain_dimensions: bool = True
) -> np.ndarray:
    """
    Create a DEM (Digital Elevation Model) image from geographic bounding box.

    Args:
        target_bbox: (N, S, E, W) bounding box coordinates
        dim: Output dimension size
        depth_scale: Scale factor for depth values
        sat_scale: Scale factor for satellite data
        water_scale: Scale factor for water subtraction
        base: Base height offset
        height: Maximum height
        subtract_water: Whether to subtract water bodies
        clip: Percentile clipping [low, high]
        smooth: Gaussian smoothing sigma
        projection: Map projection type ('none', 'cosine', 'mercator', 'equidistant', 'lambert', 'sinusoidal')
        maintain_dimensions: If True, output has predictable dimensions

    Returns:
        Processed DEM array
    """
    N, S, E, W = target_bbox

    # Stitch elevation tiles
    result = g2s.stitch_tiles_no_rasterio(target_bbox)
    im = result.copy()

    # Ensure predictable output dimensions: resize to fit within `dim` if requested
    try:
        if maintain_dimensions and dim and im is not None:
            h, w = im.shape[:2]
            max_side = max(h, w)
            if max_side > dim:
                scale = float(dim) / float(max_side)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                im = cv2.resize(im, (new_w, new_h),
                                interpolation=cv2.INTER_LINEAR)
    except Exception:
        # If resizing fails, keep original image
        pass

    # Apply depth scaling
    im[im < 0] = im[im < 0] * depth_scale

    # Get satellite water data if needed
    if subtract_water and get_aquatic_regions is not None:
        try:
            # Calculate scale based on output image size to avoid EE limits
            target_dim = min(max(im.shape[0], im.shape[1]), 500)
            img = get_aquatic_regions(
                N, S, E, W, dataset="jrc", scale=None, target_dim=target_dim)
            if img is not None:
                img2 = img.copy().astype(np.uint8)
                img2 = filters.median(img2, np.ones((3, 3)))
                img2 = cv2.resize(img2, (im.shape[1], im.shape[0]),
                                  interpolation=cv2.INTER_LINEAR).astype(int)

                # Apply water masking
                img2[im < 0] = 200
                img2[im > 500] = 0
                img2 = img2 / 100
                img2 = img2.clip(0, 1)

                # Subtract water from elevation
                im = im - img2 * np.ptp(im.ravel()) * water_scale
        except Exception as e:
            print(f"Warning: Could not process water data: {e}")
    elif subtract_water and get_aquatic_regions is None:
        pass  # Silently skip - Earth Engine not available

    # Project to 2D coordinates using selected projection
    if projection != 'none' and 0:
        try:
            from geo2stl.projections import project_coordinates
            im, proj_metadata = project_coordinates(
                im,
                (N, S, E, W),
                projection=projection,
                maintain_dimensions=maintain_dimensions,
                fill_value=np.nan
            )
        except ImportError:
            # Fallback to original projection if new module not available
            im = g2s.proj_map_geo_to_2D(im, np.array((N, S, E, W)))

    # Note: resize and rescale are intentionally NOT applied here
    # Raw elevation data is returned for client-side rendering
    # Resize/rescale should only be applied during model creation

    return im


def create_dem_model(
    im: np.ndarray,
    simplify: bool = True,
    max_faces: int = 50000,
    puzzle_cut: bool = False,
    **kwargs
) -> List:
    """
    Create 3D model from DEM array.

    Args:
        im: DEM array
        simplify: Whether to simplify mesh
        max_faces: Maximum number of faces for simplification
        puzzle_cut: Whether to apply puzzle cutting for large models
        **kwargs: Additional arguments for numpy2stl

    Returns:
        List of 3D models
    """
    if puzzle_cut and im.size > 1000000:  # Large model threshold
        # Implement puzzle cutting logic here
        # For now, just process as single model
        pass

    # Convert to STL
    vertices, faces = numpy2stl(im, **kwargs)

    models = [{
        'vertices': vertices,
        'faces': faces,
        'name': 'terrain'
    }]

    if simplify:
        # Apply mesh simplification if available
        try:
            from .simplify import simplify_mesh
            models[0]['vertices'], models[0]['faces'] = simplify_mesh(
                vertices, faces, max_faces=max_faces
            )
        except ImportError:
            print("Warning: Mesh simplification not available")

    return models


def process_region(
    name: str,
    bbox: Tuple[float, float, float, float],
    output_dir: Union[str, Path],
    **processing_kwargs
) -> str:
    """
    Process a geographic region into 3D terrain model.

    Args:
        name: Region name for output files
        bbox: (N, S, E, W) bounding box
        output_dir: Output directory path
        **processing_kwargs: Arguments for make_dem_image and create_dem_model

    Returns:
        Path to saved STL file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DEM image
    im = make_dem_image(bbox, **processing_kwargs)

    # Create 3D model
    models = create_dem_model(im, **processing_kwargs)

    # Save as STL using Solid class
    from .solid import Solid, triangles_to_facets
    from .save import writeSTL

    stl_path = output_dir / f"{name}.stl"
    vertices = models[0]['vertices']
    faces = models[0]['faces']
    triangles = vertices[faces]
    facets = triangles_to_facets(triangles)
    writeSTL(facets, str(stl_path))

    return str(stl_path)


def savefile(
    out_dir: Union[str, Path],
    name: str,
    im: np.ndarray,
    format: str = 'stl'
) -> str:
    """
    Save processed terrain data to file.

    Args:
        out_dir: Output directory
        name: File name (without extension)
        im: Terrain array
        format: Output format ('stl', '3mf', or 'npy')

    Returns:
        Path to saved file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if format.lower() == 'npy':
        filepath = out_dir / f"{name}.npy"
        np.save(str(filepath), im)
    elif format.lower() == 'stl':
        filepath = out_dir / f"{name}.stl"
        vertices, faces = numpy2stl(im)
        triangles = vertices[faces]
        facets = triangles_to_facets(triangles)
        writeSTL(facets, str(filepath))
    elif format.lower() == '3mf':
        # 3MF support would need additional implementation
        raise NotImplementedError("3MF format not yet implemented")
    else:
        raise ValueError(f"Unsupported format: {format}")

    return str(filepath)


# Region-specific processing functions
def process_appalachians(output_dir: Union[str, Path] = "output") -> str:
    """Process Appalachian Mountains region."""
    bbox = (42.0, 33.0, -74.0, -84.0)  # Approximate bounds
    return process_region("Appalachians", bbox, output_dir,
                          dim=800, depth_scale=0.3, height=40)


def process_great_lakes(output_dir: Union[str, Path] = "output") -> str:
    """Process Great Lakes region."""
    bbox = (49.0, 41.0, -76.0, -92.0)  # Approximate bounds
    return process_region("GreatLakes", bbox, output_dir,
                          dim=600, depth_scale=0.5, height=30)


def process_caribbean(output_dir: Union[str, Path] = "output") -> str:
    """Process Caribbean region."""
    bbox = (26.0, 8.0, -59.0, -85.0)  # Approximate bounds
    return process_region("Caribbean", bbox, output_dir,
                          dim=800, depth_scale=0.8, height=25)


def process_andes(output_dir: Union[str, Path] = "output") -> str:
    """Process Andes Mountains region."""
    bbox = (13.0, -56.0, -35.0, -80.0)  # Approximate bounds
    return process_region("Andes", bbox, output_dir,
                          dim=1000, depth_scale=0.4, height=50)


def process_amazon(output_dir: Union[str, Path] = "output") -> str:
    """Process Amazon region with river processing."""
    bbox = (5.0, -20.0, -44.0, -80.0)  # Approximate bounds
    return process_region("Amazon", bbox, output_dir,
                          dim=800, depth_scale=0.3, water_scale=0.05,
                          subtract_water=True, height=20)


def process_mediterranean(output_dir: Union[str, Path] = "output") -> str:
    """Process Mediterranean Sea region."""
    bbox = (48.5, 28.5, 45.2, -15.0)  # Approximate bounds
    return process_region("Mediterranean", bbox, output_dir,
                          dim=800, depth_scale=0.8, height=30)


def process_africa(output_dir: Union[str, Path] = "output") -> str:
    """Process Africa continent."""
    bbox = (37.0, -35.0, 52.0, -18.0)  # Approximate bounds
    return process_region("Africa", bbox, output_dir,
                          dim=1200, depth_scale=0.2, height=40)


def process_japan(output_dir: Union[str, Path] = "output") -> str:
    """Process Japan region."""
    bbox = (48.0, 30.0, 150.0, 122.0)  # Approximate bounds
    return process_region("Japan", bbox, output_dir,
                          dim=800, depth_scale=0.2, height=35)
