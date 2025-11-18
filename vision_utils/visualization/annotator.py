"""
Frame annotator for drawing bounding boxes and labels on images.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from ..data.structures import Detection, FrameDetections
from ..data.padding import ImagePaddingInfo

class FrameAnnotator:
    """
    Annotate frames with bounding boxes and class labels.
    """
    def __init__(
        self,
        bbox_color_correct: Tuple[int, int, int] = (0, 255, 0),
        bbox_color_incorrect: Tuple[int, int, int] = (255, 0, 0),
        bbox_color_default: Tuple[int, int, int] = (0, 255, 255),
        bbox_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1
    ):
        """
        Initialize frame annotator.

        Args:
            bbox_color_correct: BGR color for correct predictions (green)
            bbox_color_incorrect: BGR color for incorrect predictions (red)
            bbox_color_default: BGR color when no ground truth (cyan)
            bbox_thickness: Thickness of bounding box lines
            font_scale: Font scale for labels
            font_thickness: Font thickness for labels
        """
        self.bbox_color_correct = bbox_color_correct
        self.bbox_color_incorrect = bbox_color_incorrect
        self.bbox_color_default = bbox_color_default
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_frame(
        self,
        image: np.ndarray,
        frame_detections: FrameDetections,
        draw_on_padded: bool = False
    ) -> np.ndarray:
        """
        Annotate a frame with bounding boxes and labels.

        Args:
            image: Original image (RGB numpy array)
            frame_detections: FrameDetections object with detections and optional ground truth
            draw_on_padded: If True, draw on padded image; otherwise draw on original

        Returns:
            Annotated image (RGB numpy array)
        """
        annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        if draw_on_padded and frame_detections.padding_info:
            annotated = frame_detections.padding_info.pad_image(annotated)

        if frame_detections.ground_truth:
            for gt_det in frame_detections.ground_truth:
                self._draw_detection(
                    annotated,
                    gt_det,
                    color=(0, 255, 0),  # Green for ground truth
                    label_prefix="GT: ",
                    dashed=True
                )
        has_ground_truth = frame_detections.ground_truth is not None

        for detection in frame_detections.detections:
            if has_ground_truth:
                is_correct = self._is_detection_correct(
                    detection,
                    frame_detections.ground_truth
                )
                color = self.bbox_color_correct if is_correct else self.bbox_color_incorrect
            else:
                color = self.bbox_color_default

            self._draw_detection(annotated, detection, color)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        return annotated

    def _draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Tuple[int, int, int],
        label_prefix: str = "",
    ):
        """
        Draw a single detection on the image.

        Args:
            image: Image to draw on (BGR format)
            detection: Detection object
            color: BGR color tuple
            label_prefix: Prefix for label text
            dashed: Whether to draw dashed bounding box
        """
        bbox = detection.bbox
        x1, y1 = int(bbox.x_min), int(bbox.y_min)
        x2, y2 = int(bbox.x_max), int(bbox.y_max)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.bbox_thickness)

        label = f"{label_prefix}{detection.class_name}"
        if detection.confidence < 1.0:  # Only show confidence for predictions
            label += f" {detection.confidence:.2f}"

        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            self.font,
            self.font_scale,
            self.font_thickness
        )

        label_y = max(y1, label_height + 10)
        cv2.rectangle(
            image,
            (x1, label_y - label_height - 10),
            (x1 + label_width + 10, label_y),
            color,
            -1
        )

        cv2.putText(
            image,
            label,
            (x1 + 5, label_y - 5),
            self.font,
            self.font_scale,
            (255, 255, 255),
            self.font_thickness,
            cv2.LINE_AA
        )

    def _is_detection_correct(
        self,
        detection: Detection,
        ground_truth: List[Detection],
        iou_threshold: float = 0.5
    ) -> bool:
        """
        Check if a detection is correct (matches ground truth).

        Args:
            detection: Predicted detection
            ground_truth: List of ground truth detections
            iou_threshold: IoU threshold for matching

        Returns:
            True if detection matches ground truth, False otherwise
        """
        for gt in ground_truth:
            if detection.class_name != gt.class_name:
                continue
            iou = detection.bbox.iou(gt.bbox)
            if iou >= iou_threshold:
                return True

        return False

    def annotate_batch(
        self,
        images: List[np.ndarray],
        frame_detections_list: List[FrameDetections],
        draw_on_padded: bool = False
    ) -> List[np.ndarray]:
        """
        Annotate a batch of frames.

        Args:
            images: List of images
            frame_detections_list: List of FrameDetections objects
            draw_on_padded: If True, draw on padded images

        Returns:
            List of annotated images
        """
        annotated_images = []
        for image, frame_detections in zip(images, frame_detections_list):
            annotated = self.annotate_frame(image, frame_detections, draw_on_padded)
            annotated_images.append(annotated)

        return annotated_images
