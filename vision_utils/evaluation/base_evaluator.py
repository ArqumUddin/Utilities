"""
Base evaluator class with shared infrastructure for all task types.

This abstract base class provides common functionality for:
- Object detection evaluation
- Image classification evaluation
- Semantic segmentation evaluation
"""
import os
import time
import cv2
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Union
from tqdm import tqdm
from pathlib import Path
from abc import ABC, abstractmethod

from ..io.config import EvaluationConfig
from ..data.loaders import DataLoader
from ..models.factory import create_model
from ..models.inference import ModelInferenceEngine
from ..data.parsers import COCOParser
from ..data.structures import Detection, Classification, SegmentationMask
from ..metrics.basic import BasicMetrics
from ..metrics.timing import ExecutionTimer
from ..metrics.gpu_memory import GPUMemoryTracker
from ..io.results import ResultsWriter
from ..visualization.video_writer import AnnotatedVideoWriter

class BaseEvaluator(ABC):
    """
    Abstract base evaluator for vision tasks.

    Provides shared infrastructure:
    - Model loading and initialization
    - Data loading
    - GPU memory tracking
    - Timing measurements
    - Main evaluation loop structure
    - Result compilation and saving

    Subclasses must implement:
    - _create_metrics(): Create task-specific metrics
    - _create_annotator(): Create task-specific annotator
    - _process_predictions(): Process model predictions for the specific task
    - _compile_results(): Compile task-specific results
    """
    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator with configuration.

        Args:
            config: EvaluationConfig object
        """
        self.config = config
        self.timer = ExecutionTimer()

        self.timer.start('model_loading')
        self.model = create_model(
            model_name=config.model_name,
            confidence_threshold=config.confidence_threshold,
            device=config.device,
            revision=config.model_revision,
            text_prompts=config.text_prompts,
            config_file=config.config_file,
            checkpoint=config.checkpoint,
            max_detections=config.max_detections,
            nms_threshold=config.nms_threshold
        )
        self.timer.stop('model_loading')
        print(f"Model loaded in {self.timer.get_timing('model_loading'):.2f}s")

        self.model_size_mb = self.model.get_model_size_mb()
        print(f"Model size: {self.model_size_mb:.2f} MB")

        self.gpu_memory_tracker = GPUMemoryTracker(device=config.device)
        if self.gpu_memory_tracker.cuda_available:
            torch.cuda.reset_peak_memory_stats()

        self.inference_engine = ModelInferenceEngine(self.model)

        print("Initializing Data Loader")
        self.data_loader = DataLoader(
            input_path=config.input_path,
            input_type=config.input_type,
            frame_sampling_rate=config.frame_sampling_rate
        )
        print(f"Data loader initialized: {self.data_loader}")

        self.basic_metrics = BasicMetrics()
        self.advanced_metrics = None
        self.ground_truth_annotations = None

        if config.has_ground_truth:
            print("Loading Ground Truth Annotations")
            self.timer.start('ground_truth_loading')
            parser = COCOParser(config.ground_truth_path)
            self.ground_truth_annotations = parser.parse()
            self.timer.stop('ground_truth_loading')
            print(f"Loaded {len(self.ground_truth_annotations)} ground truth annotations")
            print(f"Ground truth loaded in {self.timer.get_timing('ground_truth_loading'):.2f}s")

            self.advanced_metrics = self._create_metrics()
        self.annotator = self._create_annotator()

        self.annotated_frames = []
        os.makedirs(config.output_directory, exist_ok=True)
        if config.save_annotated_frames:
            os.makedirs(config.annotated_frames_dir, exist_ok=True)

    @abstractmethod
    def _create_metrics(self):
        """
        Create task-specific advanced metrics.

        Returns:
            Task-specific metrics object (e.g., AdvancedMetrics for detection)
        """
        pass

    @abstractmethod
    def _create_annotator(self):
        """
        Create task-specific annotator.

        Returns:
            Task-specific annotator object (e.g., FrameAnnotator for detection)
        """
        pass

    @abstractmethod
    def _process_predictions(
        self,
        predictions: Union[List[Detection], List[Classification], List[SegmentationMask]],
        frame_id: int,
        image_path: Optional[str],
        padding_info,
        inference_time_ms: float
    ) -> Any:
        """
        Process model predictions for the specific task.

        Args:
            predictions: List of predictions (Detection, Classification, or SegmentationMask)
            frame_id: Frame identifier
            image_path: Path to image file
            padding_info: Image padding information
            inference_time_ms: Inference time in milliseconds

        Returns:
            Task-specific frame result object (e.g., FrameDetections, FrameClassifications, etc.)
        """
        pass

    @abstractmethod
    def _compile_results(self) -> Dict[str, Any]:
        """
        Compile task-specific evaluation results.

        Returns:
            Complete results dictionary with task-specific metrics
        """
        pass

    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete evaluation.

        Returns:
            Dictionary with evaluation results
        """
        print("Running Inference")
        self.timer.start('inference')

        for frame_id, image, image_path in tqdm(
            self.data_loader,
            desc="Processing frames",
            total=len(self.data_loader)
        ):
            start_time = time.perf_counter()
            predictions, padding_info = self.inference_engine.process_frame(
                image,
                frame_id,
                return_padding_info=True
            )

            inference_time_ms = (time.perf_counter() - start_time) * 1000
            self.basic_metrics.add_frame_inference_time(inference_time_ms)
            self.gpu_memory_tracker.record_snapshot()

            frame_result = self._process_predictions(
                predictions,
                frame_id,
                image_path,
                padding_info,
                inference_time_ms
            )

            ground_truth = None
            if self.ground_truth_annotations and image_path:
                filename = os.path.basename(image_path)
                if filename in self.ground_truth_annotations:
                    ground_truth = self.ground_truth_annotations[filename].annotations
                    frame_result.ground_truth = ground_truth

            self.basic_metrics.add_frame_detections(frame_result)
            if self.advanced_metrics and ground_truth is not None:
                self._update_advanced_metrics(predictions, ground_truth)

            if self.config.save_annotated_frames:
                self._save_annotated_frame(frame_result, image)

        self.timer.stop('inference')

        print(f"\nInference completed in {self.timer.get_timing('inference'):.2f}s")
        print(f"Average time per frame: {self.timer.get_timing('inference') / self.basic_metrics.total_frames:.4f}s")

        if self.config.generate_video and self.config.input_type == 'video':
            self._generate_annotated_video()

        results = self._compile_results()

        print("Evaluation Summary")
        ResultsWriter.print_summary(results)

        return results

    def _update_advanced_metrics(self, predictions, ground_truth):
        """
        Update advanced metrics with predictions and ground truth.

        Default implementation for detection. Override if needed.

        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
        """
        if self.advanced_metrics:
            self.advanced_metrics.add_frame_results(predictions, ground_truth)

    def _save_annotated_frame(self, frame_result, image: np.ndarray):
        """
        Save annotated frame.

        Args:
            frame_result: Frame result object (FrameDetections, FrameClassifications, etc.)
            image: Original image (RGB numpy array)
        """
        annotated = self.annotator.annotate_frame(image, frame_result)

        filename = f"frame_{frame_result.frame_id:06d}.jpg"
        output_path = os.path.join(self.config.annotated_frames_dir, filename)

        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_bgr)

        if self.config.generate_video:
            self.annotated_frames.append(output_path)

    def _generate_annotated_video(self):
        """Generate annotated video from saved frames."""
        if len(self.annotated_frames) == 0:
            print("Warning: No annotated frames to generate video from")
            return

        print("Generating Annotated Video")

        fps = 30.0
        if hasattr(self.data_loader.loader, 'fps'):
            fps = self.data_loader.loader.fps

        video_filename = f"{Path(self.config.input_path).stem}_annotated.mp4"
        video_output_path = os.path.join(self.config.output_directory, video_filename)
        AnnotatedVideoWriter.create_video_from_frames(
            self.annotated_frames,
            video_output_path,
            fps=fps
        )

        print(f"Annotated video saved to: {video_output_path}")

    def _save_results(self, results: Dict[str, Any]):
        """
        Save results to file.

        Args:
            results: Results dictionary
        """
        ResultsWriter.write_results(
            output_path=self.config.results_path,
            model_name=self.config.model_name,
            dataset_path=self.config.input_path,
            basic_metrics=results.get('basic_metrics', {}),
            execution_time=self.timer.get_timing('inference'),
            advanced_metrics=results.get('advanced_metrics'),
            config=results.get('config', {}),
            model_size_mb=self.model_size_mb,
            gpu_memory=results.get('gpu_memory', {}),
            display_name=self.config.display_name
        )
