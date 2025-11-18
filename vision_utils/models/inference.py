"""
High-level inference engine for running vision models on datasets.

Supports detection, classification, and segmentation models.
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from tqdm import tqdm

from ..data.structures import Detection, Classification, SegmentationMask
from ..data.padding import ImagePaddingInfo
from .object_detection import ObjectDetectionModel
from .zero_shot import ZeroShotDetectionModel
from .yolo import YOLOModel
from .dinotxt import DINOtxtModel

class ModelInferenceEngine:
    """
    High-level interface for running inference on datasets.

    Supports detection, classification, and segmentation models.
    """
    def __init__(self, model: Union[ObjectDetectionModel, ZeroShotDetectionModel, YOLOModel, DINOtxtModel]):
        """
        Initialize inference engine.

        Args:
            model: Model instance (any supported vision model type)
        """
        self.model = model
        self.total_inferences = 0
        self.total_detections = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        return_padding_info: bool = False
    ) -> Tuple[Union[List[Detection], List[Classification], List[SegmentationMask]], Optional[ImagePaddingInfo]]:
        """
        Process a single frame.

        Args:
            frame: RGB image as numpy array
            frame_id: Frame identifier
            return_padding_info: If True, return padding info

        Returns:
            Tuple of (results, padding_info) where results can be:
                - List[Detection] for detection models
                - List[Classification] for classification models
                - List[SegmentationMask] for segmentation models
        """
        results, padding_info = self.model.predict(frame, return_padding_info=return_padding_info)
        self.total_inferences += 1
        self.total_detections += len(results)
        return results, padding_info

    def process_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8,
        show_progress: bool = True,
        return_padding_info: bool = False
    ) -> Tuple[List[Union[List[Detection], List[Classification], List[SegmentationMask]]], Optional[List[ImagePaddingInfo]]]:
        """
        Process multiple frames.

        Args:
            frames: List of RGB images
            batch_size: Batch size for inference
            show_progress: Show progress bar
            return_padding_info: If True, return padding info for each frame

        Returns:
            Tuple of (all_results, padding_infos) where all_results is a list of result lists,
            one per frame. Each result list contains Detection, Classification, or SegmentationMask objects.
        """
        if show_progress:
            frames = tqdm(frames, desc="Processing frames")

        all_results, padding_infos = self.model.predict_batch(
            frames,
            batch_size=batch_size,
            return_padding_info=return_padding_info
        )
        self.total_inferences += len(frames)
        self.total_detections += sum(len(results) for results in all_results)

        return all_results, padding_infos

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            'total_inferences': self.total_inferences,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': (
                self.total_detections / self.total_inferences
                if self.total_inferences > 0 else 0
            )
        }

    def reset_statistics(self):
        """Reset inference statistics."""
        self.total_inferences = 0
        self.total_detections = 0

    def __repr__(self) -> str:
        return f"ModelInferenceEngine(model={self.model})"
