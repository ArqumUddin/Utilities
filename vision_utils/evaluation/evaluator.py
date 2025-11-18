"""
Main evaluator class for object detection model evaluation.
"""
import os
import numpy as np
from typing import Optional, Dict, Any, List

from ..io.config import EvaluationConfig
from ..data.structures import FrameDetections, Detection
from ..metrics.advanced import AdvancedMetrics
from ..io.results import ResultsWriter
from ..visualization.annotator import FrameAnnotator
from .base_evaluator import BaseEvaluator

class ObjectDetectionEvaluator(BaseEvaluator):
    """
    Main evaluator for object detection models.
    Supports both basic (no ground truth) and advanced (with ground truth) evaluation modes.
    """
    def _create_metrics(self):
        """Create detection-specific advanced metrics."""
        return AdvancedMetrics(iou_threshold=self.config.iou_threshold)

    def _create_annotator(self):
        """Create detection-specific annotator."""
        return FrameAnnotator(
            bbox_color_correct=self.config.bbox_color_correct,
            bbox_color_incorrect=self.config.bbox_color_incorrect,
            bbox_color_default=self.config.bbox_color_default,
            bbox_thickness=self.config.bbox_thickness,
            font_scale=self.config.font_scale
        )

    def _process_predictions(
        self,
        predictions: List[Detection],
        frame_id: int,
        image_path: Optional[str],
        padding_info,
        inference_time_ms: float
    ) -> FrameDetections:
        """
        Process detection predictions into FrameDetections.

        Args:
            predictions: List of Detection objects
            frame_id: Frame identifier
            image_path: Path to image file
            padding_info: Image padding information
            inference_time_ms: Inference time in milliseconds

        Returns:
            FrameDetections object
        """
        return FrameDetections(
            frame_id=frame_id,
            frame_path=image_path,
            detections=predictions,
            padding_info=padding_info
        )

    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile detection evaluation results.

        Returns:
            Complete results dictionary with detection metrics
        """
        basic_metrics = self.basic_metrics.get_summary()
        advanced_metrics = None
        if self.advanced_metrics:
            advanced_metrics = self.advanced_metrics.get_summary()

        config_info = {
            'model_name': self.config.model_name,
            'confidence_threshold': self.config.confidence_threshold,
            'input_path': self.config.input_path,
            'input_type': self.config.input_type,
            'frame_sampling_rate': self.config.frame_sampling_rate
        }

        if self.config.has_ground_truth:
            config_info['ground_truth_path'] = self.config.ground_truth_path
            config_info['ground_truth_format'] = self.config.ground_truth_format
            config_info['iou_threshold'] = self.config.iou_threshold

        gpu_memory_summary = self.gpu_memory_tracker.get_summary()

        self._save_results({
            'basic_metrics': basic_metrics,
            'advanced_metrics': advanced_metrics,
            'config': config_info,
            'gpu_memory': gpu_memory_summary
        })

        return ResultsWriter.load_results(self.config.results_path)

    def __repr__(self) -> str:
        eval_type = "advanced" if self.advanced_metrics else "basic"
        return f"ObjectDetectionEvaluator(model={self.config.model_name}, type={eval_type})"
