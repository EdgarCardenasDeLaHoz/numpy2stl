# Plotting helpers need matplotlib/napari: import numpy2stl.utils.visualization explicitly.
from .image import rescale, resize_max

__all__ = ["resize_max", "rescale"]
