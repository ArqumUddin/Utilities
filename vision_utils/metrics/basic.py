"""
Basic metrics calculation for object detection evaluation (no ground truth required).
"""
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

from ..data.structures import FrameDetections

class BasicMetrics:
    """
    Calculate basic metrics for object detection without ground truth.
    Tracks detection counts, frame statistics, and execution time.
    """
    def __init__(self):
        """Initialize basic metrics tracker."""
        self.total_frames = 0
        self.frames_with_detections = 0
        self.detection_counts = defaultdict(int)
        self.frame_detection_counts = []
        self.all_detections = [] 

        self.frame_inference_times = []
        self.all_confidence_scores = []

    def add_frame_detections(self, frame_detections: FrameDetections):
        """
        Add detections from a single frame.

        Args:
            frame_detections: FrameDetections object
        """
        self.total_frames += 1
        num_detections = frame_detections.num_detections

        if num_detections > 0:
            self.frames_with_detections += 1

        self.frame_detection_counts.append(num_detections)
        for detection in frame_detections.detections:
            self.detection_counts[detection.class_name] += 1
            self.all_confidence_scores.append(detection.confidence)

        self.all_detections.append(frame_detections)

    def get_detection_rate(self) -> float:
        """
        Get the detection rate (percentage of frames with detections).

        Returns:
            Detection rate between 0 and 1
        """
        if self.total_frames == 0:
            return 0.0
        return self.frames_with_detections / self.total_frames

    def get_average_detections_per_frame(self) -> float:
        """
        Get average number of detections per frame.

        Returns:
            Average detections per frame
        """
        if self.total_frames == 0:
            return 0.0
        total_detections = sum(self.frame_detection_counts)
        return total_detections / self.total_frames

    def get_class_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get per-class detection statistics.

        Returns:
            Dictionary with statistics per class
        """
        stats = {}
        total_detections = sum(self.detection_counts.values())

        for class_name, count in self.detection_counts.items():
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            stats[class_name] = {
                'count': count,
                'percentage': percentage
            }

        return stats

    def add_frame_inference_time(self, duration_ms: float):
        """
        Add inference time for a frame.

        Args:
            duration_ms: Inference duration in milliseconds
        """
        self.frame_inference_times.append(duration_ms)

    def get_inference_time_statistics(self) -> Dict[str, float]:
        """
        Get inference time statistics.

        Returns:
            Dictionary with timing statistics
        """
        if not self.frame_inference_times:
            return {
                'average_inference_time_ms': 0.0,
                'min_inference_time_ms': 0.0,
                'max_inference_time_ms': 0.0,
                'std_inference_time_ms': 0.0,
                'inference_fps': 0.0
            }

        times_array = np.array(self.frame_inference_times)
        avg_time = float(np.mean(times_array))

        return {
            'average_inference_time_ms': avg_time,
            'min_inference_time_ms': float(np.min(times_array)),
            'max_inference_time_ms': float(np.max(times_array)),
            'std_inference_time_ms': float(np.std(times_array)),
            'inference_fps': 1000.0 / avg_time if avg_time > 0 else 0.0
        }

    def get_confidence_statistics(self) -> Dict[str, Any]:
        """
        Get confidence score statistics.

        Returns:
            Dictionary with confidence statistics
        """
        if not self.all_confidence_scores:
            return {
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'quartiles': [0.0, 0.0, 0.0]
            }

        scores_array = np.array(self.all_confidence_scores)

        return {
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'std': float(np.std(scores_array)),
            'quartiles': [
                float(np.percentile(scores_array, 25)),
                float(np.percentile(scores_array, 50)),
                float(np.percentile(scores_array, 75))
            ]
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete summary of basic metrics.

        Returns:
            Dictionary with all basic metrics
        """
        summary = {
            'total_frames': self.total_frames,
            'frames_with_detections': self.frames_with_detections,
            'frames_without_detections': self.total_frames - self.frames_with_detections,
            'detection_rate': self.get_detection_rate(),
            'average_detections_per_frame': self.get_average_detections_per_frame(),
            'total_detections': sum(self.detection_counts.values()),
            'detection_counts': dict(self.detection_counts),
            'class_statistics': self.get_class_statistics()
        }

        summary.update(self.get_inference_time_statistics())
        summary['confidence_distribution'] = self.get_confidence_statistics()

        return summary

    def reset(self):
        """Reset all metrics."""
        self.total_frames = 0
        self.frames_with_detections = 0
        self.detection_counts.clear()
        self.frame_detection_counts.clear()
        self.all_detections.clear()
        self.frame_inference_times.clear()
        self.all_confidence_scores.clear()

    def __repr__(self) -> str:
        return (f"BasicMetrics(total_frames={self.total_frames}, "
                f"frames_with_detections={self.frames_with_detections}, "
                f"total_detections={sum(self.detection_counts.values())})")
