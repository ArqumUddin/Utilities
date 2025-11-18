"""
Evaluator class for image classification model evaluation.

Supports:
- Top-k accuracy metrics
- Confusion matrix
- Per-class accuracy
- Classification reports
"""
import numpy as np
from typing import Optional, Dict, Any, List
from collections import defaultdict

from ..io.config import EvaluationConfig
from ..data.structures import FrameClassifications, Classification
from ..metrics.basic import BasicMetrics
from ..io.results import ResultsWriter
from ..visualization.annotator import FrameAnnotator
from .base_evaluator import BaseEvaluator

class ClassificationMetrics:
    """
    Metrics for image classification evaluation.

    Computes:
    - Top-1, Top-3, Top-5 accuracy
    - Per-class accuracy
    - Confusion matrix
    """
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize classification metrics.

        Args:
            class_names: List of all possible class names
        """
        self.class_names = class_names or []
        self.correct_top1 = 0
        self.correct_top3 = 0
        self.correct_top5 = 0
        self.total_samples = 0
        self.class_correct = defaultdict(int)
        self.class_total = defaultdict(int)
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))

    def add_sample_result(
        self,
        predictions: List[Classification],
        ground_truth: Classification
    ):
        """
        Add results for a single sample.

        Args:
            predictions: List of Classification objects (sorted by confidence)
            ground_truth: Ground truth Classification
        """
        if not predictions:
            return

        self.total_samples += 1
        gt_class = ground_truth.class_name
        self.class_total[gt_class] += 1

        pred_top1 = predictions[0].class_name
        if pred_top1 == gt_class:
            self.correct_top1 += 1
            self.class_correct[gt_class] += 1

        self.confusion_matrix[pred_top1][gt_class] += 1

        if len(predictions) >= 3:
            top3_classes = [p.class_name for p in predictions[:3]]
            if gt_class in top3_classes:
                self.correct_top3 += 1
        elif gt_class in [p.class_name for p in predictions]:
            self.correct_top3 += 1

        if len(predictions) >= 5:
            top5_classes = [p.class_name for p in predictions[:5]]
            if gt_class in top5_classes:
                self.correct_top5 += 1
        elif gt_class in [p.class_name for p in predictions]:
            self.correct_top5 += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of classification metrics.

        Returns:
            Dictionary with all classification metrics
        """
        if self.total_samples == 0:
            return {
                'top1_accuracy': 0.0,
                'top3_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'total_samples': 0,
                'per_class_accuracy': {},
                'confusion_matrix': {}
            }

        top1_acc = self.correct_top1 / self.total_samples
        top3_acc = self.correct_top3 / self.total_samples
        top5_acc = self.correct_top5 / self.total_samples

        per_class_acc = {}
        for class_name in self.class_total:
            total = self.class_total[class_name]
            correct = self.class_correct[class_name]
            per_class_acc[class_name] = {
                'accuracy': correct / total if total > 0 else 0.0,
                'correct': correct,
                'total': total
            }

        confusion_dict = {}
        for pred_class in self.confusion_matrix:
            confusion_dict[pred_class] = dict(self.confusion_matrix[pred_class])

        return {
            'top1_accuracy': float(top1_acc),
            'top3_accuracy': float(top3_acc),
            'top5_accuracy': float(top5_acc),
            'total_samples': self.total_samples,
            'correct_top1': self.correct_top1,
            'correct_top3': self.correct_top3,
            'correct_top5': self.correct_top5,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion_dict
        }

class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for image classification models.

    Supports:
    - Top-k accuracy evaluation
    - Per-class performance analysis
    - Confusion matrix generation
    """
    def _create_metrics(self):
        """Create classification-specific advanced metrics."""
        class_names = self.config.text_prompts if self.config.text_prompts else None
        return ClassificationMetrics(class_names=class_names)

    def _create_annotator(self):
        """Create classification-specific annotator."""
        return FrameAnnotator(
            bbox_color_default=self.config.bbox_color_default,
            bbox_thickness=self.config.bbox_thickness,
            font_scale=self.config.font_scale
        )

    def _process_predictions(
        self,
        predictions: List[Classification],
        frame_id: int,
        image_path: Optional[str],
        padding_info,
        inference_time_ms: float
    ) -> FrameClassifications:
        """
        Process classification predictions into FrameClassifications.

        Args:
            predictions: List of Classification objects
            frame_id: Frame identifier
            image_path: Path to image file
            padding_info: Image padding information
            inference_time_ms: Inference time in milliseconds

        Returns:
            FrameClassifications object
        """
        return FrameClassifications(
            frame_id=frame_id,
            frame_path=image_path,
            classifications=predictions,
            padding_info=padding_info
        )

    def _update_advanced_metrics(self, predictions: List[Classification], ground_truth: List[Classification]):
        """
        Update classification metrics with predictions and ground truth.
        Ground truth should be a single classification (not a list), but we receive it as a list from the parser, so take the first element

        Args:
            predictions: List of predicted Classification objects (sorted by confidence)
            ground_truth: List containing single ground truth Classification
        """
        if self.advanced_metrics and ground_truth:
            gt_classification = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
            self.advanced_metrics.add_sample_result(predictions, gt_classification)

    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile classification evaluation results.

        Returns:
            Complete results dictionary with classification metrics
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
            'frame_sampling_rate': self.config.frame_sampling_rate,
            'text_prompts': self.config.text_prompts
        }

        if self.config.has_ground_truth:
            config_info['ground_truth_path'] = self.config.ground_truth_path
            config_info['ground_truth_format'] = self.config.ground_truth_format

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
        return f"ClassificationEvaluator(model={self.config.model_name}, type={eval_type})"
