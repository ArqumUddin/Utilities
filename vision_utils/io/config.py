"""
Configuration parser and validator for object detection evaluation.
"""
import os
import torch
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class EvaluationConfig:
    """
    Configuration class for object detection evaluation.
    Parses and validates YAML configuration files.
    """
    def __init__(self, config_path: str):
        """
        Initialize configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {self.config_path}")

        return config

    def _validate_config(self):
        """Validate required configuration parameters."""
        required_fields = ['model', 'input', 'output']

        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

        if 'name' not in self.config.get('model', {}):
            raise ValueError("Model configuration must include 'name'")

        if 'path' not in self.config.get('input', {}):
            raise ValueError("Input configuration must include 'path'")

        input_path = self.config.get('input', {}).get('path')
        if input_path and not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if 'directory' not in self.config.get('output', {}):
            raise ValueError("Output configuration must include 'directory'")

    @property
    def model_name(self) -> str:
        """Get model name/identifier."""
        return self.config.get('model', {}).get('name')

    @property
    def display_name(self) -> Optional[str]:
        """Get display name for the model (optional)."""
        return self.config.get('model', {}).get('display_name', None)

    @property
    def model_revision(self) -> Optional[str]:
        """Get model revision (optional)."""
        return self.config.get('model', {}).get('revision', None)

    @property
    def confidence_threshold(self) -> float:
        """Get confidence threshold for detections."""
        return self.config.get('model', {}).get('confidence_threshold', 0.5)

    @property
    def device(self) -> str:
        """Get device for model inference (cpu/cuda)."""
        return self.config.get('model', {}).get('device', 'cuda' if self._is_cuda_available() else 'cpu')

    @property
    def text_prompts(self) -> Optional[list]:
        """Get text prompts list for zero-shot models (YOLO-World, Grounding DINO)."""
        return self.config.get('model', {}).get('text_prompts', None)

    @property
    def config_file(self) -> Optional[str]:
        """Get config file path for models like YOLO-World."""
        return self.config.get('model', {}).get('config_file', None)

    @property
    def checkpoint(self) -> Optional[str]:
        """Get checkpoint path for models like YOLO-World."""
        return self.config.get('model', {}).get('checkpoint', None)

    @property
    def max_detections(self) -> int:
        """Get maximum detections for models like YOLO-World."""
        return self.config.get('model', {}).get('max_detections', 100)

    @property
    def nms_threshold(self) -> float:
        """Get NMS threshold for models like YOLO-World."""
        return self.config.get('model', {}).get('nms_threshold', 0.5)

    @property
    def input_path(self) -> str:
        """Get input dataset path."""
        return self.config.get('input', {}).get('path')

    @property
    def input_type(self) -> str:
        """
        Get input type (video or images).
        Auto-detected if not specified.
        """
        input_config = self.config.get('input', {})
        if 'type' in input_config:
            return input_config.get('type')

        path = Path(self.input_path)
        if path.is_file() and path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            return 'video'
        elif path.is_dir():
            return 'images'
        else:
            raise ValueError(f"Cannot determine input type for: {self.input_path}")

    @property
    def frame_sampling_rate(self) -> int:
        """Get frame sampling rate (process every Nth frame)."""
        return self.config.get('input', {}).get('frame_sampling_rate', 1)

    @property
    def ground_truth_path(self) -> Optional[str]:
        """Get ground truth annotations path (optional)."""
        gt_path = self.config.get('ground_truth', {}).get('path', None)
        if gt_path and not os.path.exists(gt_path):
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        return gt_path

    @property
    def ground_truth_format(self) -> str:
        """Get ground truth format (coco, yolo, custom)."""
        return self.config.get('ground_truth', {}).get('format', 'coco')

    @property
    def iou_threshold(self) -> float:
        """Get IoU threshold for matching predictions to ground truth."""
        return self.config.get('ground_truth', {}).get('iou_threshold', 0.5)

    @property
    def has_ground_truth(self) -> bool:
        """Check if ground truth annotations are provided."""
        return self.ground_truth_path is not None

    @property
    def output_directory(self) -> str:
        """Get output directory path."""
        return self.config.get('output', {}).get('directory')

    @property
    def save_annotated_frames(self) -> bool:
        """Check if annotated frames should be saved."""
        return self.config.get('output', {}).get('save_annotated_frames', True)

    @property
    def annotated_frames_dir(self) -> str:
        """Get directory for annotated frames."""
        frames_dir = self.config.get('output', {}).get('annotated_frames_dir', 'annotated_frames')
        if not os.path.isabs(frames_dir):
            frames_dir = os.path.join(self.output_directory, frames_dir)
        return frames_dir

    @property
    def generate_video(self) -> bool:
        """Check if annotated video should be generated (for video inputs)."""
        return self.config.get('output', {}).get('generate_video', False)

    @property
    def results_filename(self) -> str:
        """Get results JSON filename."""
        return self.config.get('output', {}).get('results_filename', 'results.json')

    @property
    def results_path(self) -> str:
        """Get full path to results JSON file."""
        return os.path.join(self.output_directory, self.results_filename)

    @property
    def visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {})

    @property
    def bbox_color_correct(self) -> tuple:
        """Get bounding box color for correct predictions (when ground truth available)."""
        color = self.visualization_config.get('bbox_color_correct', [0, 255, 0])
        return tuple(color)

    @property
    def bbox_color_incorrect(self) -> tuple:
        """Get bounding box color for incorrect predictions."""
        color = self.visualization_config.get('bbox_color_incorrect', [255, 0, 0])
        return tuple(color)

    @property
    def bbox_color_default(self) -> tuple:
        """Get default bounding box color (no ground truth)."""
        color = self.visualization_config.get('bbox_color_default', [0, 255, 255])
        return tuple(color)

    @property
    def bbox_thickness(self) -> int:
        """Get bounding box line thickness."""
        return self.visualization_config.get('bbox_thickness', 2)

    @property
    def font_scale(self) -> float:
        """Get font scale for labels."""
        return self.visualization_config.get('font_scale', 0.5)

    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            return torch.cuda.is_available()
        except ImportError:
            return False

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (f"EvaluationConfig(model={self.model_name}, "
                f"input={self.input_path}, "
                f"output={self.output_directory}, "
                f"has_ground_truth={self.has_ground_truth})")
