import numpy as np
from .view import *
from .polygon import *
from collections import defaultdict
import cv2
from skimage import filters

####################### 3D Object Processing tools #######################################


def resize_max(im, max_size=1000):
    """
    Resize an image to fit within a maximum size while maintaining aspect ratio.

    Parameters:
    - im: Input image as a 2D numpy array.
    - max_size: Maximum size for the longest dimension of the image.

    Returns:
    - Resized image as a 2D numpy array.
    """
    height, width = im.shape
    scale = max_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    resized_im = cv2.resize(im, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_im


def rescale(im, max_size=600, height=20, base=10, clip=None, smooth=None):
    """
    Rescale and process an elevation image for 3D printing.

    Parameters:
    - im: Input elevation image
    - max_size: Maximum dimension size
    - height: Maximum height of the model
    - base: Base height offset
    - clip: Percentile clipping [low, high] or single value for symmetric clip
    - smooth: Median filter size for smoothing

    Returns:
    - Processed elevation image
    """
    im = resize_max(im, max_size=max_size)

    if clip is not None:
        if len(clip) == 1:
            clip = [clip, 100 - clip]
        lo, hi = np.percentile(im.ravel(), clip)
        im = im.clip(lo, hi)

    im = im - im.min()
    im = im / im.ptp() * height
    im = im + base
    return im
