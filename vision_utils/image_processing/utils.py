"""
Utility functions for image processing.
"""
import torch
import numpy as np
import cv2
from typing import Optional, Union, Tuple
from PIL import Image

def auto_detect_device(device: Optional[str] = None) -> str:
    """
    Auto-detect CUDA/CPU device.

    Args:
        device: Optional device string. If None, auto-detects.

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def load_image_as_pil(image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
    """
    Load image from various formats and convert to PIL Image (RGB).

    Args:
        image: Input image as file path, numpy array, or PIL Image

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If image type is unsupported or file cannot be loaded
    """
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
        if img is None:
            raise ValueError(f"Failed to load image from {image}")
        return img
    elif isinstance(image, np.ndarray):
        return Image.fromarray(image.astype('uint8'), 'RGB')
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def load_image_as_cv2(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Load image from various formats and convert to OpenCV format (BGR).

    Args:
        image: Input image as file path, numpy array, or PIL Image

    Returns:
        Numpy array in BGR format (OpenCV)

    Raises:
        ValueError: If image type is unsupported or file cannot be loaded
    """
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image from {image}")
        return img
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return image
    elif isinstance(image, Image.Image):
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

def load_mask(mask: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load mask from file path or numpy array as grayscale.

    Args:
        mask: Mask as file path or numpy array

    Returns:
        Grayscale mask as numpy array

    Raises:
        ValueError: If mask type is unsupported
    """
    if isinstance(mask, str):
        return cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    elif isinstance(mask, np.ndarray):
        if mask.ndim == 2:
            return mask
        else:
            return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported mask type: {type(mask)}")

def compute_mask_bbox(
    mask_array: np.ndarray,
    expand_factor: float = 1.2,
    threshold: float = 0.8
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute bounding box from mask with optional expansion.

    Args:
        mask_array: Grayscale mask as numpy array
        expand_factor: Factor to expand bounding box (default 1.2 = 20% larger)
        threshold: Threshold for mask binarization (0-1, default 0.8)

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) or None if mask is empty
    """
    bbox_coords = np.argwhere(mask_array > threshold * 255)
    if len(bbox_coords) == 0:
        print("Warning: Mask is empty")
        return None

    x_min = np.min(bbox_coords[:, 1])
    y_min = np.min(bbox_coords[:, 0])
    x_max = np.max(bbox_coords[:, 1])
    y_max = np.max(bbox_coords[:, 0])

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    size = max(x_max - x_min, y_max - y_min)
    size = int(size * expand_factor)

    crop_x_min = int(center_x - size // 2)
    crop_y_min = int(center_y - size // 2)
    crop_x_max = int(center_x + size // 2)
    crop_y_max = int(center_y + size // 2)

    return (crop_x_min, crop_y_min, crop_x_max, crop_y_max)

def crop_pil_image_to_mask(
    image: Image.Image,
    mask: Union[str, np.ndarray],
    expand_factor: float = 1.2,
    threshold: float = 0.8
) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
    """
    Crop PIL image to mask bounding box with expansion.

    Args:
        image: PIL Image to crop
        mask: Mask as file path or numpy array
        expand_factor: Factor to expand bounding box
        threshold: Threshold for mask binarization (0-1)

    Returns:
        Tuple of (cropped_image, bbox) where bbox is (x_min, y_min, x_max, y_max)
        Returns original image and None if mask is empty
    """
    mask_array = load_mask(mask)
    bbox = compute_mask_bbox(mask_array, expand_factor, threshold)

    if bbox is None:
        return image, None

    x_min, y_min, x_max, y_max = bbox
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return cropped, bbox

def crop_cv2_image_to_mask(
    image: np.ndarray,
    mask: Union[str, np.ndarray],
    expand_factor: float = 1.2,
    threshold: float = 0.8
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """
    Crop OpenCV image to mask bounding box with expansion.

    Args:
        image: OpenCV image (BGR numpy array) to crop
        mask: Mask as file path or numpy array
        expand_factor: Factor to expand bounding box
        threshold: Threshold for mask binarization (0-1)

    Returns:
        Tuple of (cropped_image, bbox) where bbox is (x_min, y_min, x_max, y_max)
        Returns original image and None if mask is empty
    """
    mask_array = load_mask(mask)
    bbox = compute_mask_bbox(mask_array, expand_factor, threshold)

    if bbox is None:
        return image, None

    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped, (x_min, y_min, x_max, y_max)
