"""
Bounding box data structure and utilities.
"""
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BoundingBox:
    """
    Bounding box representation.
    Uses [x_min, y_min, x_max, y_max] format (absolute coordinates).
    """
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Get bounding box width."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Get bounding box height."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates (x, y)."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        return (self.x_min, self.y_min, self.width, self.height)

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x_min, y_min, x_max, y_max) format."""
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> 'BoundingBox':
        """Create bounding box from (x, y, width, height) format."""
        return cls(x_min=x, y_min=y, x_max=x + w, y_max=y + h)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> 'BoundingBox':
        """Create bounding box from (center_x, center_y, width, height) format."""
        return cls(x_min=cx - w/2, y_min=cy - h/2, x_max=cx + w/2, y_max=cy + h/2)

    def scale(self, width_scale: float, height_scale: float) -> 'BoundingBox':
        """
        Scale bounding box coordinates.

        Args:
            width_scale: Scale factor for x coordinates
            height_scale: Scale factor for y coordinates

        Returns:
            New scaled BoundingBox
        """
        return BoundingBox(
            x_min=self.x_min * width_scale,
            y_min=self.y_min * height_scale,
            x_max=self.x_max * width_scale,
            y_max=self.y_max * height_scale
        )

    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another bounding box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU value between 0 and 1
        """
        x_left = max(self.x_min, other.x_min)
        y_top = max(self.y_min, other.y_min)
        x_right = min(self.x_max, other.x_max)
        y_bottom = min(self.y_max, other.y_max)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area + other.area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'x_min': float(self.x_min),
            'y_min': float(self.y_min),
            'x_max': float(self.x_max),
            'y_max': float(self.y_max),
            'width': float(self.width),
            'height': float(self.height)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BoundingBox':
        """Create BoundingBox from dictionary."""
        return cls(
            x_min=data['x_min'],
            y_min=data['y_min'],
            x_max=data['x_max'],
            y_max=data['y_max']
        )

    def __repr__(self) -> str:
        return f"BoundingBox(x_min={self.x_min:.1f}, y_min={self.y_min:.1f}, x_max={self.x_max:.1f}, y_max={self.y_max:.1f})"


def calculate_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box
        bbox2: Second bounding box

    Returns:
        IoU value between 0 and 1
    """
    return bbox1.iou(bbox2)
