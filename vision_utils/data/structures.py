"""
Core data structures for object detection results.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import cv2

from ..utils.bbox import BoundingBox
from .padding import ImagePaddingInfo

@dataclass
class Detection:
    """
    Single object detection result.
    """
    bbox: BoundingBox
    class_name: str
    confidence: float
    class_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'bbox': self.bbox.to_dict(),
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'class_id': int(self.class_id) if self.class_id is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """Create Detection from dictionary."""
        bbox = BoundingBox.from_dict(data['bbox'])
        return cls(
            bbox=bbox,
            class_name=data['class_name'],
            confidence=data['confidence'],
            class_id=data.get('class_id')
        )

    def __repr__(self) -> str:
        return f"Detection(class={self.class_name}, confidence={self.confidence:.3f}, bbox={self.bbox})"

@dataclass
class FrameDetections:
    """
    All detections for a single frame/image.
    """
    frame_id: int
    frame_path: Optional[str]
    detections: List[Detection] = field(default_factory=list)
    ground_truth: Optional[List[Detection]] = None
    padding_info: Optional[ImagePaddingInfo] = None 

    @property
    def num_detections(self) -> int:
        """Get number of detections in frame."""
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        """Check if frame has any detections."""
        return len(self.detections) > 0

    @property
    def class_counts(self) -> dict:
        """Get count of detections per class."""
        counts = {}
        for det in self.detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'frame_path': self.frame_path,
            'num_detections': len(self.detections),
            'detections': [det.to_dict() for det in self.detections]
        }

    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Get all detections for a specific class."""
        return [det for det in self.detections if det.class_name == class_name]

    def __repr__(self) -> str:
        return f"FrameDetections(frame_id={self.frame_id}, num_detections={self.num_detections})"

@dataclass
class DataAnnotation:
    """
    Data annotation for a single frame/image.
    """
    frame_id: int
    frame_path: Optional[str]
    annotations: List[Detection] = field(default_factory=list)

    @property
    def num_annotations(self) -> int:
        """Get number of annotations."""
        return len(self.annotations)

    @property
    def class_counts(self) -> dict:
        """Get count of annotations per class."""
        counts = {}
        for ann in self.annotations:
            counts[ann.class_name] = counts.get(ann.class_name, 0) + 1
        return counts

    def __repr__(self) -> str:
        return f"DataAnnotation(frame_id={self.frame_id}, num_annotations={self.num_annotations})"

@dataclass
class Classification:
    """
    Single classification result for an image.
    """
    class_name: str
    confidence: float
    class_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'class_id': int(self.class_id) if self.class_id is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Classification':
        """Create Classification from dictionary."""
        return cls(
            class_name=data['class_name'],
            confidence=data['confidence'],
            class_id=data.get('class_id')
        )

    def __repr__(self) -> str:
        return f"Classification(class={self.class_name}, confidence={self.confidence:.3f})"

@dataclass
class SegmentationMask:
    """
    Segmentation mask result with associated class information.
    """
    mask: np.ndarray  # Binary mask (H, W) with values 0 or 255, or float mask with values [0, 1]
    class_name: str
    confidence: float
    class_id: Optional[int] = None

    def to_bounding_box(self, threshold: float = 0.5) -> Optional[BoundingBox]:
        """
        Convert segmentation mask to bounding box.

        Args:
            threshold: Threshold for binary mask (default: 0.5)

        Returns:
            BoundingBox enclosing the mask region, or None if mask is empty
        """
        # Normalize mask to [0, 1] range
        if self.mask.max() > 1.0:
            mask_normalized = self.mask / 255.0
        else:
            mask_normalized = self.mask

        binary_mask = (mask_normalized > threshold).astype(np.uint8)
        coords = np.argwhere(binary_mask > 0)

        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        return BoundingBox(
            x_min=float(x_min),
            y_min=float(y_min),
            x_max=float(x_max),
            y_max=float(y_max)
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'mask': self.mask.tolist(),
            'mask_shape': self.mask.shape,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'class_id': int(self.class_id) if self.class_id is not None else None,
            'bounding_box': self.to_bounding_box().to_dict() if self.to_bounding_box() is not None else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SegmentationMask':
        """Create SegmentationMask from dictionary."""
        mask = np.array(data['mask']).reshape(data['mask_shape'])
        return cls(
            mask=mask,
            class_name=data['class_name'],
            confidence=data['confidence'],
            class_id=data.get('class_id')
        )

    def __repr__(self) -> str:
        bbox = self.to_bounding_box()
        bbox_str = f", bbox={bbox}" if bbox is not None else ""
        return f"SegmentationMask(class={self.class_name}, confidence={self.confidence:.3f}, shape={self.mask.shape}{bbox_str})"

@dataclass
class FrameClassifications:
    """
    All classifications for a single frame/image.
    """
    frame_id: int
    frame_path: Optional[str] = None
    classifications: List[Classification] = field(default_factory=list)
    ground_truth: Optional[List[Classification]] = None
    padding_info: Optional[ImagePaddingInfo] = None

    @property
    def num_classifications(self) -> int:
        """Get number of classifications."""
        return len(self.classifications)

    @property
    def has_classifications(self) -> bool:
        """Check if frame has any classifications."""
        return len(self.classifications) > 0

    @property
    def top_prediction(self) -> Optional[Classification]:
        """Get top prediction (highest confidence)."""
        return self.classifications[0] if self.classifications else None

    @property
    def class_confidences(self) -> dict:
        """Get confidence per class."""
        return {c.class_name: c.confidence for c in self.classifications}

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'frame_path': self.frame_path,
            'num_classifications': len(self.classifications),
            'classifications': [c.to_dict() for c in self.classifications],
            'top_prediction': self.top_prediction.to_dict() if self.top_prediction else None
        }

    def get_classification_by_name(self, class_name: str) -> Optional[Classification]:
        """Get classification for a specific class."""
        for c in self.classifications:
            if c.class_name == class_name:
                return c
        return None

    def __repr__(self) -> str:
        return f"FrameClassifications(frame_id={self.frame_id}, num_classifications={self.num_classifications})"

@dataclass
class FrameSegmentations:
    """
    All segmentation masks for a single frame/image.
    """
    frame_id: int
    frame_path: Optional[str] = None
    segmentation_masks: List[SegmentationMask] = field(default_factory=list)
    ground_truth: Optional[List[SegmentationMask]] = None
    padding_info: Optional[ImagePaddingInfo] = None

    @property
    def num_masks(self) -> int:
        """Get number of segmentation masks."""
        return len(self.segmentation_masks)

    @property
    def has_masks(self) -> bool:
        """Check if frame has any segmentation masks."""
        return len(self.segmentation_masks) > 0

    @property
    def class_names(self) -> List[str]:
        """Get list of class names in masks."""
        return [mask.class_name for mask in self.segmentation_masks]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'frame_id': self.frame_id,
            'frame_path': self.frame_path,
            'num_masks': len(self.segmentation_masks),
            'segmentation_masks': [mask.to_dict() for mask in self.segmentation_masks]
        }

    def get_mask_by_class(self, class_name: str) -> Optional[SegmentationMask]:
        """Get segmentation mask for a specific class."""
        for mask in self.segmentation_masks:
            if mask.class_name == class_name:
                return mask
        return None

    def __repr__(self) -> str:
        return f"FrameSegmentations(frame_id={self.frame_id}, num_masks={self.num_masks})"
