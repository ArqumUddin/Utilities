"""
Vision Utilities - A comprehensive library for object detection and computer vision tasks.
"""

__version__ = "0.1.0"

from .data.structures import (
    Detection,
    FrameDetections,
    DataAnnotation,
    Classification,
    SegmentationMask,
    FrameClassifications,
    FrameSegmentations
)
from .utils.bbox import BoundingBox, calculate_iou
from .data.padding import ImagePaddingInfo
from .utils.matching import match_detections_to_ground_truth
from .data.loaders import VideoLoader, ImageDatasetLoader, DataLoader
from .data.parsers import COCOParser
from .models.factory import create_model
from .models.object_detection import ObjectDetectionModel
from .models.zero_shot import ZeroShotDetectionModel
from .models.yolo import YOLOModel
from .models.dinotxt import DINOtxtModel
from .models.inference import ModelInferenceEngine
from .metrics.basic import BasicMetrics
from .metrics.advanced import AdvancedMetrics
from .metrics.timing import ExecutionTimer
from .metrics.gpu_memory import GPUMemoryTracker
from .visualization.annotator import FrameAnnotator
from .visualization.video_writer import AnnotatedVideoWriter
from .visualization.plotter import ResultsPlotter
from .io.config import EvaluationConfig, ServerConfig
from .io.results import ResultsWriter
from .evaluation.evaluator import ObjectDetectionEvaluator
from .evaluation.classification_evaluator import ClassificationEvaluator
from .evaluation.segmentation_evaluator import SegmentationEvaluator
from .video_processing.concatenate import (
    VideoInfo,
    VideoComparison,
    concatenate_videos_horizontal,
    concatenate_videos_vertical,
    concatenate_videos_grid,
    concatenate_videos_sequential,
)
from .server.base import host_model, image_to_str, str_to_image
from .server.server import VisionServer
from .server.client import DetectionClient
from .robotics.structures import Pose3D, Object3D, CameraIntrinsics
from .robotics.rotations import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
from .robotics.transforms import (
    transform_point_world_to_camera,
    project_3d_to_2d,
    project_world_object_to_image,
)
from .image_processing.stable_diffusion_upscale import (
    upscale_image_stable_diffusion,
)
from .image_processing.esrgan_upscale import (
    upscale_image_esrgan,
)
from .ycb.downloader import YCBDownloader
from .ycb.point_cloud_generator import YCBPointCloudGenerator

__all__ = [
    "__version__",
    "Detection",
    "FrameDetections",
    "DataAnnotation",
    "Classification",
    "SegmentationMask",
    "FrameClassifications",
    "FrameSegmentations",
    "BoundingBox",
    "ImagePaddingInfo",
    "calculate_iou",
    "match_detections_to_ground_truth",
    "VideoLoader",
    "ImageDatasetLoader",
    "DataLoader",
    "COCOParser",
    "create_model",
    "ObjectDetectionModel",
    "ZeroShotDetectionModel",
    "YOLOModel",
    "DINOtxtModel",
    "ModelInferenceEngine",
    "BasicMetrics",
    "AdvancedMetrics",
    "ExecutionTimer",
    "GPUMemoryTracker",
    "FrameAnnotator",
    "AnnotatedVideoWriter",
    "ResultsPlotter",
    "EvaluationConfig",
    "ServerConfig",
    "ResultsWriter",
    "ObjectDetectionEvaluator",
    "ClassificationEvaluator",
    "SegmentationEvaluator",
    "VideoInfo",
    "VideoComparison",
    "concatenate_videos_horizontal",
    "concatenate_videos_vertical",
    "concatenate_videos_grid",
    "concatenate_videos_sequential",
    "image_to_str",
    "str_to_image",
    "host_model",
    "VisionServer",
    "DetectionClient",
    "Pose3D",
    "Object3D",
    "CameraIntrinsics",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "transform_point_world_to_camera",
    "project_3d_to_2d",
    "project_world_object_to_image",
    "upscale_image_stable_diffusion",
    "upscale_image_esrgan",
    "YCBDownloader",
    "YCBPointCloudGenerator",
]
