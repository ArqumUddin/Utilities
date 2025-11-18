"""
Base model interface and shared utilities for object detection models.
"""
import numpy as np
from typing import Tuple
from ..data.padding import ImagePaddingInfo

def pad_image_to_square(image: np.ndarray) -> Tuple[np.ndarray, ImagePaddingInfo]:
    """
    Pad image to square by adding padding to the right and bottom.
    The largest dimension is kept constant, smaller dimension is padded.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        Tuple of (padded_image, padding_info)
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)
    pad_right = max_dim - width
    pad_bottom = max_dim - height

    padding_info = ImagePaddingInfo(
        original_width=width,
        original_height=height,
        padded_width=max_dim,
        padded_height=max_dim,
        pad_right=pad_right,
        pad_bottom=pad_bottom
    )

    if pad_right == 0 and pad_bottom == 0:
        return image, padding_info

    padded_image = np.zeros((max_dim, max_dim, image.shape[2]), dtype=image.dtype)
    padded_image[:height, :width] = image

    return padded_image, padding_info
