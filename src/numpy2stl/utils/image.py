
import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    from skimage import filters

    HAS_SKIMAGE = True
except ImportError:
    filters = None
    HAS_SKIMAGE = False

__all__ = ["resize_max", "rescale"]


def resize_max(im, max_size=1000):
    """
    Resize an image to fit within a maximum size while maintaining aspect ratio.

    Parameters:
    - im: Input image as a 2D numpy array.
    - max_size: Maximum size for the longest dimension of the image.

    Returns:
    - Resized image as a 2D numpy array.

    Raises:
    - ImportError: If opencv-python is not installed.
    """
    if not HAS_CV2:
        raise ImportError(
            "opencv-python is required for resize_max(). "
            "Install with: pip install numpy2stl[tools]"
        )

    height, width = im.shape
    if max(height, width) <= max_size:
        return im
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
    im = im / np.ptp(im) * height
    im = im + base
    return im
