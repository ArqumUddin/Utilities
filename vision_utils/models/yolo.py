"""
Ultralytics YOLO model wrapper.
Supports YOLOv8, YOLO-World, and other Ultralytics models.
"""
import os
import torch
import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO

from ..data.structures import Detection
from ..utils.bbox import BoundingBox
from ..data.padding import ImagePaddingInfo
from .base import pad_image_to_square

class YOLOModel:
    """
    Wrapper for Ultralytics YOLO models (YOLOv8, YOLO-World, etc.).
    Provides the same interface as ObjectDetectionModel.
    """
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        text_prompts: Optional[List[str]] = None
    ):
        """
        Initialize Ultralytics YOLO model.

        Args:
            model_name: Model name (e.g., 'yolov8n.pt', 'yolov8l-world.pt') or HF repo
            confidence_threshold: Minimum confidence score for detections
            device: Device for inference ('cuda' or 'cpu')
            revision: Not used for YOLO models (kept for interface compatibility)
            text_prompts: Text prompts for YOLO-World models (ignored for regular YOLO)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.revision = revision
        self.text_prompts = text_prompts  # Store for fallback

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading YOLO model: {model_name} on {self.device}")

        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model {model_name}: {e}")
        self.is_yolo_world = hasattr(self.model, 'set_classes')

        if self.is_yolo_world and text_prompts is not None:
            print(f"YOLO-World model detected. Setting classes: {text_prompts}")
            self.model.set_classes(text_prompts)

        print(f"YOLO model loaded successfully: {model_name}")

        self.id2label = {i: name for i, name in enumerate(self.model.names.values())}
        self.label2id = {name: i for i, name in enumerate(self.model.names.values())}
        self.class_names = list(self.label2id.keys())

    @staticmethod
    def pad_image_to_square(image: np.ndarray) -> Tuple[np.ndarray, ImagePaddingInfo]:
        """Pad image to square - delegates to base function."""
        return pad_image_to_square(image)

    def predict(self, image: np.ndarray, return_padding_info: bool = False, text_prompts: Optional[List[str]] = None) -> Tuple[List[Detection], Optional[ImagePaddingInfo]]:
        """
        Run inference on a single image.
        Image is padded to square before inference, and bounding boxes are returned
        in the original (unpadded) image coordinate system.

        Args:
            image: RGB image as numpy array (H, W, 3)
            return_padding_info: If True, return padding info along with detections
            text_prompts: Optional text prompts for YOLO-World models (ignored for regular YOLO)

        Returns:
            Tuple of (detections, padding_info) if return_padding_info is True,
            otherwise just (detections, None)
        """
        if self.is_yolo_world:
            if text_prompts is None:
                if self.text_prompts is not None:
                    self.model.set_classes(self.text_prompts)
            else:
                self.model.set_classes(text_prompts)

            self.id2label = {i: name for i, name in enumerate(self.model.names.values())}
            self.label2id = {name: i for i, name in enumerate(self.model.names.values())}
            self.class_names = list(self.label2id.keys())

        padded_image, padding_info = self.pad_image_to_square(image)
        results = self.model.predict(
            padded_image,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device
        )
        detections = []
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # [N, 4]
            confidences = result.boxes.conf.cpu().numpy()  # [N]
            class_ids = result.boxes.cls.cpu().numpy()  # [N]

            for box_coords, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
                class_name = self.id2label.get(int(cls_id), f"class_{int(cls_id)}")
                bbox_padded = BoundingBox(
                    x_min=float(box_coords[0]),
                    y_min=float(box_coords[1]),
                    x_max=float(box_coords[2]),
                    y_max=float(box_coords[3])
                )

                bbox = padding_info.unpad_bbox(bbox_padded)

                detection = Detection(
                    bbox=bbox,
                    class_name=class_name,
                    confidence=float(conf),
                    class_id=int(cls_id)
                )
                detections.append(detection)

        if return_padding_info:
            return detections, padding_info
        else:
            return detections, None

    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8,
        return_padding_info: bool = False,
        text_prompts: Optional[List[str]] = None
    ) -> Tuple[List[List[Detection]], Optional[List[ImagePaddingInfo]]]:
        """
        Run inference on a batch of images.
        Processes images sequentially using predict() to avoid code duplication.

        Args:
            images: List of RGB images as numpy arrays
            batch_size: Batch size for inference (unused, kept for interface compatibility)
            return_padding_info: If True, return padding info for each image
            text_prompts: Optional text prompts for YOLO-World models

        Returns:
            Tuple of (all_detections, padding_infos) if return_padding_info is True,
            otherwise (all_detections, None)
        """
        all_detections = []
        all_padding_infos = [] if return_padding_info else None

        for img in images:
            detections, padding_info = self.predict(
                img,
                return_padding_info=return_padding_info,
                text_prompts=text_prompts
            )
            all_detections.append(detections)

            if return_padding_info:
                all_padding_infos.append(padding_info)

        return all_detections, all_padding_infos

    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        return self.class_names

    def get_num_classes(self) -> int:
        """Get number of classes the model can detect."""
        return len(self.class_names)

    def get_model_size_mb(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes
        """
        try:
            if hasattr(self.model, 'ckpt_path') and self.model.ckpt_path:
                if os.path.exists(self.model.ckpt_path):
                    size_mb = os.path.getsize(self.model.ckpt_path) / (1024 ** 2)
                    return float(size_mb)

            if os.path.exists(self.model_name):
                size_mb = os.path.getsize(self.model_name) / (1024 ** 2)
                return float(size_mb)

            if hasattr(self.model, 'model'):
                num_params = sum(p.numel() for p in self.model.model.parameters())
                size_mb = (num_params * 4) / (1024 ** 2)
                return float(size_mb)

            return 0.0
        except Exception as e:
            print(f"Warning: Could not calculate YOLO model size: {e}")
            return 0.0

    def __repr__(self) -> str:
        return (f"YOLOModel(model={self.model_name}, "
                f"device={self.device}, "
                f"threshold={self.confidence_threshold}, "
                f"num_classes={self.get_num_classes()})")
