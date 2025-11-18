"""
Image padding information and utilities.
"""
from dataclasses import dataclass
import numpy as np
from ..utils.bbox import BoundingBox

@dataclass
class ImagePaddingInfo:
    """
    Information about padding applied to an image for model inference.
    """
    original_width: int
    original_height: int
    padded_width: int
    padded_height: int
    pad_right: int
    pad_bottom: int

    def unpad_bbox(self, bbox: BoundingBox) -> BoundingBox:
        """
        Convert bounding box from padded image coordinates to original image coordinates.

        Note:
        Bounding boxes in padded image are already in the correct coordinate space
        since padding only adds to right and bottom, coordinates remain the same
        Just clamp to original dimensions to ensure no boxes extend into padding
        
        Args:
            bbox: BoundingBox in padded image coordinates

        Returns:
            BoundingBox in original image coordinates
        """

        return BoundingBox(
            x_min=min(bbox.x_min, self.original_width),
            y_min=min(bbox.y_min, self.original_height),
            x_max=min(bbox.x_max, self.original_width),
            y_max=min(bbox.y_max, self.original_height)
        )

    def pad_image(self, image: np.ndarray) -> np.ndarray:
        """
        Pad image according to padding info.

        Args:
            image: Original image (H, W, C)

        Returns:
            Padded image
        """
        if self.pad_right == 0 and self.pad_bottom == 0:
            return image

        padded = np.zeros((self.padded_height, self.padded_width, image.shape[2]), dtype=image.dtype)
        padded[:self.original_height, :self.original_width] = image
        return padded
