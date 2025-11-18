"""
Advanced metrics calculation for object detection with ground truth annotations.
"""
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
from ..data.structures import Detection, FrameDetections
from ..utils.matching import match_detections_to_ground_truth

class AdvancedMetrics:
    """
    Calculate advanced metrics for object detection with ground truth.
    Includes precision, recall, F1-score, mAP, and confusion matrices.
    """
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize advanced metrics tracker.

        Args:
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.iou_threshold = iou_threshold
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.all_classes = set()
        self.confusion_data = []  # List of (predicted_class, actual_class) tuples

        self.frame_results = []

    def add_frame_results(
        self,
        predictions: List[Detection],
        ground_truth: List[Detection]
    ):
        """
        Add results from a single frame.

        Args:
            predictions: List of predicted detections
            ground_truth: List of ground truth detections
        """
        for det in predictions + ground_truth:
            self.all_classes.add(det.class_name)

        matched_pairs, unmatched_preds, unmatched_gt = match_detections_to_ground_truth(
            predictions, ground_truth, self.iou_threshold
        )

        for pred_idx, gt_idx in matched_pairs:
            pred = predictions[pred_idx]
            gt = ground_truth[gt_idx]

            self.true_positives[pred.class_name] += 1
            self.confusion_data.append((pred.class_name, gt.class_name))

        for pred_idx in unmatched_preds:
            pred = predictions[pred_idx]
            self.false_positives[pred.class_name] += 1
            self.confusion_data.append((pred.class_name, None))  # None = no ground truth

        for gt_idx in unmatched_gt:
            gt = ground_truth[gt_idx]
            self.false_negatives[gt.class_name] += 1
            self.confusion_data.append((None, gt.class_name))  # None = missed detection

        self.frame_results.append({
            'predictions': predictions,
            'ground_truth': ground_truth,
            'matched_pairs': matched_pairs
        })

    def calculate_precision(self, class_name: str) -> float:
        """
        Calculate precision for a specific class.

        Args:
            class_name: Class name

        Returns:
            Precision value between 0 and 1
        """
        tp = self.true_positives[class_name]
        fp = self.false_positives[class_name]

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def calculate_recall(self, class_name: str) -> float:
        """
        Calculate recall for a specific class.

        Args:
            class_name: Class name

        Returns:
            Recall value between 0 and 1
        """
        tp = self.true_positives[class_name]
        fn = self.false_negatives[class_name]

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def calculate_f1_score(self, class_name: str) -> float:
        """
        Calculate F1-score for a specific class.

        Args:
            class_name: Class name

        Returns:
            F1-score value between 0 and 1
        """
        precision = self.calculate_precision(class_name)
        recall = self.calculate_recall(class_name)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def calculate_average_precision(self, class_name: str) -> float:
        """
        Calculate Average Precision (AP) for a specific class.

        Args:
            class_name: Class name

        Returns:
            Average Precision value between 0 and 1
        """
        all_predictions = []
        for frame_result in self.frame_results:
            for det in frame_result['predictions']:
                if det.class_name == class_name:
                    all_predictions.append(det)

        if len(all_predictions) == 0:
            return 0.0

        all_predictions.sort(key=lambda x: x.confidence, reverse=True)

        tp_count = 0
        fp_count = 0
        precisions = []
        recalls = []

        total_gt = sum(
            len([d for d in frame['ground_truth'] if d.class_name == class_name])
            for frame in self.frame_results
        )

        if total_gt == 0:
            return 0.0

        for pred in all_predictions:
            is_tp = False
            for frame_result in self.frame_results:
                for pred_idx, gt_idx in frame_result['matched_pairs']:
                    if frame_result['predictions'][pred_idx] == pred:
                        is_tp = True
                        break
                if is_tp:
                    break

            if is_tp:
                tp_count += 1
            else:
                fp_count += 1

            precision = tp_count / (tp_count + fp_count)
            recall = tp_count / total_gt

            precisions.append(precision)
            recalls.append(recall)

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if len([r for r in recalls if r >= t]) == 0:
                p = 0.0
            else:
                p = max([precisions[i] for i, r in enumerate(recalls) if r >= t])
            ap += p / 11.0

        return ap

    def calculate_map(self) -> float:
        """
        Calculate Mean Average Precision (mAP) across all classes.

        Returns:
            mAP value between 0 and 1
        """
        if len(self.all_classes) == 0:
            return 0.0

        aps = [self.calculate_average_precision(class_name) for class_name in self.all_classes]
        return sum(aps) / len(aps)

    def get_confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get confusion matrix as numpy array.

        Returns:
            Tuple of (confusion_matrix, class_labels)
        """
        classes = sorted(list(self.all_classes))
        n_classes = len(classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Rows: predicted, Columns: actual
        confusion_matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=int)

        for pred_class, actual_class in self.confusion_data:
            if pred_class is None:
                pred_idx = n_classes
            else:
                pred_idx = class_to_idx[pred_class]

            if actual_class is None:
                actual_idx = n_classes
            else:
                actual_idx = class_to_idx[actual_class]

            confusion_matrix[pred_idx, actual_idx] += 1

        labels = classes + ['background']

        return confusion_matrix, labels

    def calculate_overall_accuracy(self) -> float:
        """
        Calculate overall accuracy across all classes.

        Returns:
            Overall accuracy between 0 and 1
        """
        total_tp = sum(self.true_positives.values())
        total_fp = sum(self.false_positives.values())
        total_fn = sum(self.false_negatives.values())

        total = total_tp + total_fp + total_fn

        if total == 0:
            return 0.0

        return total_tp / total

    def get_per_class_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get complete metrics for each class.

        Returns:
            Dictionary with metrics per class
        """
        metrics = {}

        for class_name in self.all_classes:
            metrics[class_name] = {
                'precision': self.calculate_precision(class_name),
                'recall': self.calculate_recall(class_name),
                'f1_score': self.calculate_f1_score(class_name),
                'average_precision': self.calculate_average_precision(class_name),
                'true_positives': self.true_positives[class_name],
                'false_positives': self.false_positives[class_name],
                'false_negatives': self.false_negatives[class_name]
            }

        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete summary of advanced metrics.

        Returns:
            Dictionary with all advanced metrics
        """
        confusion_matrix, labels = self.get_confusion_matrix()

        return {
            'iou_threshold': self.iou_threshold,
            'overall_accuracy': self.calculate_overall_accuracy(),
            'mean_average_precision': self.calculate_map(),
            'per_class_metrics': self.get_per_class_metrics(),
            'confusion_matrix': confusion_matrix.tolist(),
            'confusion_matrix_labels': labels,
            'total_true_positives': sum(self.true_positives.values()),
            'total_false_positives': sum(self.false_positives.values()),
            'total_false_negatives': sum(self.false_negatives.values())
        }

    def reset(self):
        """Reset all metrics."""
        self.true_positives.clear()
        self.false_positives.clear()
        self.false_negatives.clear()
        self.all_classes.clear()
        self.confusion_data.clear()
        self.frame_results.clear()

    def __repr__(self) -> str:
        return (f"AdvancedMetrics(iou_threshold={self.iou_threshold}, "
                f"mAP={self.calculate_map():.4f}, "
                f"accuracy={self.calculate_overall_accuracy():.4f})")
