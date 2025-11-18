"""
HuggingFace Transformers object detection model wrapper.
Supports models like DETR, RT-DETR, and other HF detection models.
"""
import torch
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from ..data.structures import Detection
from ..utils.bbox import BoundingBox
from ..data.padding import ImagePaddingInfo
from .base import pad_image_to_square

class ObjectDetectionModel:
    """Wrapper for HuggingFace Transformers object detection models."""
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        revision: Optional[str] = None
    ):
        """
        Initialize object detection model.

        Args:
            model_name: HuggingFace model identifier
            confidence_threshold: Minimum confidence score for detections
            device: Device for inference ('cuda' or 'cpu')
            revision: Specific model revision
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.revision = revision

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading model: {model_name} on {self.device}")
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                revision=revision
            )
            self.model = AutoModelForObjectDetection.from_pretrained(
                model_name,
                revision=revision
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")

        print(f"Model loaded successfully: {model_name}")
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

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
            text_prompts: Not used for standard models (kept for API consistency)

        Returns:
            Tuple of (detections, padding_info) if return_padding_info is True,
            otherwise just (detections, None)
        """
        padded_image, padding_info = self.pad_image_to_square(image)

        if isinstance(padded_image, np.ndarray):
            pil_image = Image.fromarray(padded_image.astype('uint8'), 'RGB')
        else:
            pil_image = padded_image

        img_width, img_height = pil_image.size
        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[img_height, img_width]]).to(self.device)
        results = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            score_val = score.item()
            label_val = label.item()
            box_coords = box.cpu().numpy()

            class_name = self.id2label.get(label_val, f"class_{label_val}")

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
                confidence=float(score_val),
                class_id=int(label_val)
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
        return_padding_info: bool = False
    ) -> Tuple[List[List[Detection]], Optional[List[ImagePaddingInfo]]]:
        """
        Run inference on a batch of images.
        Processes images sequentially using predict() to avoid code duplication.

        Args:
            images: List of RGB images as numpy arrays
            batch_size: Batch size for inference (unused, kept for interface compatibility)
            return_padding_info: If True, return padding info for each image

        Returns:
            Tuple of (all_detections, padding_infos) if return_padding_info is True,
            otherwise (all_detections, None)
        """
        all_detections = []
        all_padding_infos = [] if return_padding_info else None

        for img in images:
            detections, padding_info = self.predict(
                img,
                return_padding_info=return_padding_info
            )
            all_detections.append(detections)

            if return_padding_info:
                all_padding_infos.append(padding_info)

        return all_detections, all_padding_infos

    def get_class_names(self) -> List[str]:
        """Get list of class names the model can detect."""
        return list(self.label2id.keys())

    def get_num_classes(self) -> int:
        """Get number of classes the model can detect."""
        return len(self.id2label)

    def get_model_size_mb(self) -> float:
        """
        Get model size in MB.

        Returns:
            Model size in megabytes
        """
        try:
            num_params = sum(p.numel() for p in self.model.parameters())
            size_mb = (num_params * 4) / (1024 ** 2)
            return float(size_mb)
        except Exception as e:
            print(f"Warning: Could not calculate model size: {e}")
            return 0.0

    def __repr__(self) -> str:
        return (f"ObjectDetectionModel(model={self.model_name}, "
                f"device={self.device}, "
                f"threshold={self.confidence_threshold}, "
                f"num_classes={self.get_num_classes()})")
