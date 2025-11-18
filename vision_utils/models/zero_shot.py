"""
HuggingFace zero-shot object detection model wrapper.

Supports models using AutoModelForZeroShotObjectDetection including:
- Grounding DINO (IDEA-Research/grounding-dino-*)
- OWL-ViT (google/owlvit-*)
- OWLv2 (google/owlv2-*)
- LLMDet (iSEE-Laboratory/llmdet_*)
"""
import torch
import numpy as np
from typing import List, Optional, Tuple, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from ..data.structures import Detection
from ..utils.bbox import BoundingBox
from ..data.padding import ImagePaddingInfo
from .base import pad_image_to_square

class ZeroShotDetectionModel:
    """
    Wrapper for HuggingFace zero-shot object detection models.

    Supports any model using AutoModelForZeroShotObjectDetection,
    including Grounding DINO, OWL-ViT, OWLv2, and LLMDet.

    Requires text prompts to specify which objects to detect.
    """
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        text_prompts: Optional[List[str]] = None
    ):
        """
        Initialize zero-shot detection model.

        Args:
            model_name: HuggingFace model identifier (e.g., IDEA-Research/grounding-dino-base,
                        google/owlvit-base-patch32, google/owlv2-base-patch16)
            confidence_threshold: Minimum confidence score for detections
            device: Device for inference ('cuda' or 'cpu')
            revision: Specific model revision
            text_prompts: List of class names to detect
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.revision = revision

        if text_prompts is not None:
            self.text_prompts = text_prompts
        else:
            self.text_prompts = ["person", "car", "dog", "cat", "bicycle", "motorcycle", "truck", "bus", "boat", "chair"]

        self.class_names = self.text_prompts

        if isinstance(self.text_prompts, list):
            self.text_prompt = ". ".join(self.text_prompts) + "."
        else:
            self.text_prompt = self.text_prompts

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading zero-shot detection model: {model_name} on {self.device}")
        print(f"Text prompts ({len(self.class_names)} classes): {self.class_names[:5]}..." if len(self.class_names) > 5 else f"Text prompts: {self.class_names}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                revision=revision
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_name,
                revision=revision
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load zero-shot detection model {model_name}: {e}")

        print(f"Zero-shot detection model loaded successfully: {model_name}")

    @staticmethod
    def pad_image_to_square(image: np.ndarray) -> Tuple[np.ndarray, ImagePaddingInfo]:
        """Pad image to square - delegates to base function."""
        return pad_image_to_square(image)

    def predict(
        self,
        image: np.ndarray,
        return_padding_info: bool = False,
        text_prompts: Optional[List[str]] = None
    ) -> Union[List[Detection], Tuple[List[Detection], ImagePaddingInfo]]:
        """
        Run object detection on an image using text prompts.

        Args:
            image: RGB image as numpy array
            return_padding_info: If True, return (detections, padding_info) tuple
            text_prompts: Optional text prompts to override initialization prompts

        Returns:
            List of Detection objects, or tuple of (detections, padding_info) if return_padding_info is True
        """
        if text_prompts is None:
            text_prompts = self.text_prompts
            class_names = self.class_names
            text_prompt = self.text_prompt
        else:
            class_names = text_prompts
            if isinstance(text_prompts, list):
                text_prompt = ". ".join(text_prompts) + "."
            else:
                text_prompt = text_prompts

        padded_image, padding_info = self.pad_image_to_square(image)
        pil_image = Image.fromarray(padded_image)
        inputs = self.processor(
            images=pil_image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.confidence_threshold,
            target_sizes=[(padding_info.padded_height, padding_info.padded_width)]
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]
        class_names_lower = {name.lower().strip(): name for name in class_names}

        for box, score, label in zip(boxes, scores, labels):
            label_lower = label.lower().strip()
            matched_class_name = class_names_lower.get(label_lower, label)
            x_min, y_min, x_max, y_max = box
            bbox_padded = BoundingBox(
                x_min=float(x_min),
                y_min=float(y_min),
                x_max=float(x_max),
                y_max=float(y_max)
            )
            bbox = padding_info.unpad_bbox(bbox_padded)

            detection = Detection(
                bbox=bbox,
                class_name=matched_class_name,
                confidence=float(score)
            )
            detections.append(detection)

        if return_padding_info:
            return detections, padding_info
        return detections

    def get_num_classes(self) -> int:
        """
        Get number of classes.
        For zero-shot models, this depends on the text prompt.
        """
        return len([c.strip() for c in self.text_prompt.split('.') if c.strip()])

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
        return (f"ZeroShotDetectionModel(model={self.model_name}, "
                f"device={self.device}, "
                f"threshold={self.confidence_threshold}, "
                f"prompt='{self.text_prompt[:50]}...')")
