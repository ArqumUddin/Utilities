"""
Factory function for creating object detection models.
"""
from typing import Optional, List, Union

from .object_detection import ObjectDetectionModel
from .zero_shot import ZeroShotDetectionModel
from .yolo import YOLOModel
from .dinotxt import DINOtxtModel

def create_model(
    model_name: str,
    confidence_threshold: float = 0.5,
    device: Optional[str] = None,
    revision: Optional[str] = None,
    config_file: Optional[str] = None,
    checkpoint: Optional[str] = None,
    text_prompts: Optional[List[str]] = None,
    max_detections: int = 100,
    nms_threshold: float = 0.5
) -> Union[ObjectDetectionModel, ZeroShotDetectionModel, YOLOModel, DINOtxtModel]:
    """
    Factory function to create the appropriate model type based on model_name.

    Automatically detects whether to use:
    - DINOtxtModel (for DINOtxt vision-language models)
    - YOLOModel (for YOLO models)
    - ZeroShotDetectionModel (for zero-shot models: Grounding DINO, OWL-ViT, OWLv2, etc.)
    - ObjectDetectionModel (for standard HuggingFace Transformers models)

    Args:
        model_name: Model identifier
        confidence_threshold: Minimum confidence score for detections
        device: Device for inference ('cuda' or 'cpu')
        revision: Specific model revision (HuggingFace only)
        config_file: Config file path (reserved for future use)
        checkpoint: Checkpoint path (reserved for future use)
        text_prompts: List of class names (zero-shot models, DINOtxt, and YOLO-World)
        max_detections: Maximum detections (reserved for future use)
        nms_threshold: NMS threshold (reserved for future use)

    Returns:
        Model instance (DINOtxtModel, YOLOModel, ZeroShotDetectionModel, or ObjectDetectionModel)
    """
    model_name_lower = model_name.lower()
    is_dinotxt = (
        'dinotxt' in model_name_lower or
        'dinov3_vitl16_dinotxt' in model_name_lower
    )

    is_zero_shot = (
        ('grounding' in model_name_lower and 'dino' in model_name_lower) or
        'owlvit' in model_name_lower or
        'owlv2' in model_name_lower or
        'llmdet' in model_name_lower
    )

    is_yolo = (
        'yolo' in model_name_lower or
        model_name.endswith('.pt') or
        model_name.endswith('.yaml')
    )

    if is_dinotxt:
        return DINOtxtModel(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            text_prompts=text_prompts
        )
    elif is_zero_shot:
        return ZeroShotDetectionModel(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            revision=revision,
            text_prompts=text_prompts
        )
    elif is_yolo:
        return YOLOModel(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            revision=revision,
            text_prompts=text_prompts
        )
    else:
        return ObjectDetectionModel(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            device=device,
            revision=revision
        )
