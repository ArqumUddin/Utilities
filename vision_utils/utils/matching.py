"""
Detection matching utilities for comparing predictions with ground truth.
"""
import numpy as np
from typing import List, Tuple

from ..data.structures import Detection

def match_detections_to_ground_truth(
    predictions: List[Detection],
    ground_truth: List[Detection],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predicted detections to ground truth annotations using IoU.

    Args:
        predictions: List of predicted detections
        ground_truth: List of ground truth detections
        iou_threshold: Minimum IoU for a match

    Returns:
        Tuple of:
        - matched_pairs: List of (pred_idx, gt_idx) tuples
        - unmatched_predictions: List of prediction indices
        - unmatched_ground_truth: List of ground truth indices
    """
    if len(predictions) == 0 or len(ground_truth) == 0:
        return [], list(range(len(predictions))), list(range(len(ground_truth)))

    iou_matrix = np.zeros((len(predictions), len(ground_truth)))
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            if pred.class_name == gt.class_name:  # Only match same class
                iou_matrix[i, j] = pred.bbox.iou(gt.bbox)

    matched_pairs = []
    unmatched_predictions = set(range(len(predictions)))
    unmatched_ground_truth = set(range(len(ground_truth)))

    while True:
        if iou_matrix.size == 0 or iou_matrix.max() < iou_threshold:
            break

        i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matched_pairs.append((int(i), int(j)))
        unmatched_predictions.discard(int(i))
        unmatched_ground_truth.discard(int(j))

        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    return matched_pairs, sorted(list(unmatched_predictions)), sorted(list(unmatched_ground_truth))
