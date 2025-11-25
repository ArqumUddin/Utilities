"""
Generic vision server that works with any vision_utils model.

This module provides a unified server wrapper that supports:
- Object detection models (YOLO, DETR, RT-DETR, Grounding DINO, etc.)
- Classification models (DINOtxt, etc.)
- Segmentation models (DINOtxt, etc.)
"""
from typing import List, Optional, Union
import numpy as np

from .base import str_to_image
from ..models.factory import create_model
from ..data.structures import (
    FrameDetections,
    FrameClassifications,
    FrameSegmentations,
    Detection,
    Classification,
    SegmentationMask
)

class VisionServer:
    """
    Generic server wrapper for any vision_utils model.

    This unified server can host any model supported by the vision_utils factory:
    - Detection models: YOLO, DETR, RT-DETR, Grounding DINO, OWL-ViT, etc.
    - Classification models: DINOtxt, etc.
    - Segmentation models: DINOtxt, etc.

    The server automatically detects the model's output type and returns
    the appropriate response format.
    """
    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
        text_prompts: Optional[List[str]] = None,
        **model_kwargs
    ):
        """
        Initialize vision server.

        Args:
            model_name: Model identifier (auto-detected by factory)
                - Detection: "yolov8n.pt", "IDEA-Research/grounding-dino-base"
                - Classification/Segmentation: "dinov3_vitl16_dinotxt"
            confidence_threshold: Minimum confidence score (default: 0.5)
            device: Device for inference ('cuda' or 'cpu', default: auto-detect)
            text_prompts: Text prompts for zero-shot/vision-language models
            **model_kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        print(f"Initializing model: {model_name}")
        self.model = create_model(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            text_prompts=text_prompts,
            **model_kwargs
        )
        print(f"Model initialized successfully!")

    def process_payload(self, payload: dict) -> dict:
        """
        Process incoming vision request (detection/classification/segmentation).

        Args:
            payload: Dictionary containing request data
                - image: Base64 encoded image string (required)
                - confidence_threshold: Optional confidence threshold override
                - caption: Optional text prompts (string or list format)
                - text_prompts: Optional text prompts (list format)
                - top_k: For classification, number of top predictions (default: 5)
                - output_size: For segmentation, output mask size [H, W]

        Returns:
            Dictionary containing results (format depends on model type):
                - Detection models: {"detections": [...], "num_detections": N}
                - Classification models: {"classifications": [...], "top_prediction": {...}}
                - Segmentation models: {"segmentation_masks": [...], "num_masks": N}
        """
        img_np = str_to_image(payload["image"])
        if "confidence_threshold" in payload:
            original_confidence = self.model.confidence_threshold
            self.model.confidence_threshold = payload["confidence_threshold"]

        caption = payload.get("caption", None)
        prompts = None
        if caption is not None:
            if isinstance(caption, str):
                prompts = [c.strip() for c in caption.rstrip(" .").split(" . ")]
            elif isinstance(caption, list):
                prompts = caption
        elif "text_prompts" in payload:
            prompts = payload["text_prompts"]

        if "output_size" in payload and hasattr(self.model, 'predict_segmentation'):
            results = self.model.predict_segmentation(
                img_np,
                text_prompts=prompts,
                output_size=tuple(payload["output_size"]) if payload["output_size"] else None
            )
        else:
            extra_kwargs = {}
            if "top_k" in payload:
                extra_kwargs["top_k"] = payload["top_k"]

            results = self.model.predict(
                img_np,
                text_prompts=prompts,
                **extra_kwargs
            )

        if "confidence_threshold" in payload:
            self.model.confidence_threshold = original_confidence

        if isinstance(results, tuple):
            results = results[0]

        if not results:
            result = {
                "detections": [],
                "model_name": self.model_name,
                "warning": f"Empty results: {type(results)}"
            }
        elif isinstance(results[0], Detection):
            frame_detections = FrameDetections(
                frame_id=0,
                frame_path=None,
                detections=results
            )
            result = frame_detections.to_dict()
            result["model_name"] = self.model_name
        elif isinstance(results[0], Classification):
            frame_classifications = FrameClassifications(
                frame_id=0,
                frame_path=None,
                classifications=results
            )
            result = frame_classifications.to_dict()
            result["model_name"] = self.model_name
        elif isinstance(results[0], SegmentationMask):
            frame_segmentations = FrameSegmentations(
                frame_id=0,
                frame_path=None,
                segmentation_masks=results
            )
            result = frame_segmentations.to_dict()
            result["model_name"] = self.model_name
        else:
            result = {
                "detections": [str(r) for r in results],
                "model_name": self.model_name,
                "warning": f"Unknown result type: {type(results[0])}"
            }

        return result
