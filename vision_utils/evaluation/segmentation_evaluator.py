"""
Evaluator class for semantic segmentation model evaluation.

Supports:
- Pixel accuracy
- Mean Intersection over Union (mIoU)
- Per-class IoU
- Dice coefficient
"""
import numpy as np
from typing import Optional, Dict, Any, List
from collections import defaultdict
import cv2

from ..io.config import EvaluationConfig
from ..data.structures import FrameSegmentations, SegmentationMask
from ..metrics.basic import BasicMetrics
from ..io.results import ResultsWriter
from ..visualization.annotator import FrameAnnotator
from .base_evaluator import BaseEvaluator

class SegmentationMetrics:
    """
    Metrics for semantic segmentation evaluation.

    Computes:
    - Pixel accuracy
    - Mean IoU (mIoU)
    - Per-class IoU
    - Dice coefficient
    """
    def __init__(self, class_names: Optional[List[str]] = None, threshold: float = 0.5):
        """
        Initialize segmentation metrics.

        Args:
            class_names: List of all possible class names
            threshold: Threshold for converting continuous masks to binary (default: 0.5)
        """
        self.class_names = class_names or []
        self.threshold = threshold
        self.total_pixels_correct = 0
        self.total_pixels = 0
        self.class_intersection = defaultdict(int)
        self.class_union = defaultdict(int)
        self.class_dice_sum = defaultdict(float)
        self.class_dice_count = defaultdict(int)

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert mask to binary (0 or 1).

        Args:
            mask: Input mask (either [0, 255] or [0, 1])

        Returns:
            Binary mask with values 0 or 1
        """
        if mask.max() > 1.0:
            mask_normalized = mask / 255.0
        else:
            mask_normalized = mask

        return (mask_normalized > self.threshold).astype(np.uint8)

    def add_frame_result(
        self,
        pred_masks: List[SegmentationMask],
        gt_masks: List[SegmentationMask]
    ):
        """
        Add results for a single frame/image.

        Args:
            pred_masks: List of predicted SegmentationMask objects
            gt_masks: List of ground truth SegmentationMask objects
        """
        pred_dict = {mask.class_name: mask for mask in pred_masks}
        gt_dict = {mask.class_name: mask for mask in gt_masks}
        all_classes = set(pred_dict.keys()) | set(gt_dict.keys())

        for class_name in all_classes:
            pred_mask = pred_dict.get(class_name)
            gt_mask = gt_dict.get(class_name)
            if pred_mask is None or gt_mask is None:
                continue

            if pred_mask.mask.shape != gt_mask.mask.shape:
                pred_mask_resized = cv2.resize(
                    pred_mask.mask,
                    (gt_mask.mask.shape[1], gt_mask.mask.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            else:
                pred_mask_resized = pred_mask.mask

            pred_binary = self._binarize_mask(pred_mask_resized)
            gt_binary = self._binarize_mask(gt_mask.mask)

            correct_pixels = np.sum(pred_binary == gt_binary)
            total_pixels = pred_binary.size
            self.total_pixels_correct += correct_pixels
            self.total_pixels += total_pixels

            intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
            union = np.sum((pred_binary == 1) | (gt_binary == 1))
            self.class_intersection[class_name] += intersection
            self.class_union[class_name] += union

            if union > 0:
                dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(gt_binary))
                self.class_dice_sum[class_name] += dice
                self.class_dice_count[class_name] += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of segmentation metrics.

        Returns:
            Dictionary with all segmentation metrics
        """
        if self.total_pixels == 0:
            return {
                'pixel_accuracy': 0.0,
                'mean_iou': 0.0,
                'mean_dice': 0.0,
                'per_class_metrics': {},
                'total_pixels': 0
            }

        pixel_accuracy = self.total_pixels_correct / self.total_pixels
        per_class_metrics = {}
        ious = []
        dices = []

        for class_name in set(self.class_union.keys()) | set(self.class_dice_count.keys()):
            metrics = {}
            if class_name in self.class_union and self.class_union[class_name] > 0:
                iou = self.class_intersection[class_name] / self.class_union[class_name]
                metrics['iou'] = float(iou)
                ious.append(iou)
            else:
                metrics['iou'] = 0.0

            if class_name in self.class_dice_count and self.class_dice_count[class_name] > 0:
                dice = self.class_dice_sum[class_name] / self.class_dice_count[class_name]
                metrics['dice'] = float(dice)
                dices.append(dice)
            else:
                metrics['dice'] = 0.0

            metrics['samples'] = self.class_dice_count.get(class_name, 0)
            per_class_metrics[class_name] = metrics

        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_dice = float(np.mean(dices)) if dices else 0.0

        return {
            'pixel_accuracy': float(pixel_accuracy),
            'mean_iou': mean_iou,
            'mean_dice': mean_dice,
            'per_class_metrics': per_class_metrics,
            'total_pixels': self.total_pixels,
            'total_correct_pixels': self.total_pixels_correct,
            'num_classes_evaluated': len(per_class_metrics)
        }

class SegmentationEvaluator(BaseEvaluator):
    """
    Evaluator for semantic segmentation models.

    Supports:
    - Pixel-level accuracy evaluation
    - IoU and Dice metrics
    - Per-class performance analysis
    """
    def _create_metrics(self):
        """Create segmentation-specific advanced metrics."""
        class_names = self.config.text_prompts if self.config.text_prompts else None
        threshold = getattr(self.config, 'mask_threshold', 0.5)
        return SegmentationMetrics(class_names=class_names, threshold=threshold)

    def _create_annotator(self):
        """Create segmentation-specific annotator."""
        return FrameAnnotator(
            bbox_color_default=self.config.bbox_color_default,
            bbox_thickness=self.config.bbox_thickness,
            font_scale=self.config.font_scale
        )

    def _process_predictions(
        self,
        predictions: List[SegmentationMask],
        frame_id: int,
        image_path: Optional[str],
        padding_info,
        inference_time_ms: float
    ) -> FrameSegmentations:
        """
        Process segmentation predictions into FrameSegmentations.

        Args:
            predictions: List of SegmentationMask objects
            frame_id: Frame identifier
            image_path: Path to image file
            padding_info: Image padding information
            inference_time_ms: Inference time in milliseconds

        Returns:
            FrameSegmentations object
        """
        return FrameSegmentations(
            frame_id=frame_id,
            frame_path=image_path,
            segmentation_masks=predictions,
            padding_info=padding_info
        )

    def _update_advanced_metrics(self, predictions: List[SegmentationMask], ground_truth: List[SegmentationMask]):
        """
        Update segmentation metrics with predictions and ground truth.

        Args:
            predictions: List of predicted SegmentationMask objects
            ground_truth: List of ground truth SegmentationMask objects
        """
        if self.advanced_metrics and ground_truth:
            self.advanced_metrics.add_frame_result(predictions, ground_truth)

    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile segmentation evaluation results.

        Returns:
            Complete results dictionary with segmentation metrics
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
            'text_prompts': self.config.text_prompts,
            'mask_threshold': getattr(self.config, 'mask_threshold', 0.5)
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
        return f"SegmentationEvaluator(model={self.config.model_name}, type={eval_type})"
