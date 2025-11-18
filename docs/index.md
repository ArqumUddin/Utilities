# Vision Utilities

A comprehensive Python library for computer vision tasks including object detection, image classification, and semantic segmentation. Provides modular utilities for data loading, model inference, metrics calculation, and visualization.

## Features

- **Modular Architecture**: Clean separation between data, models, metrics, and visualization
- **Multiple Vision Tasks**: Object detection, image classification, semantic segmentation
- **Multiple Model Support**:
  - Detection: HuggingFace Transformers (DETR, RT-DETR), YOLO (Ultralytics), Grounding DINO, OWL-ViT
  - Classification & Segmentation: DINOtxt vision-language models
- **Zero-Shot Capabilities**: Text-guided detection and segmentation with Grounding DINO, DINOtxt
- **Flexible Data Loading**: Video and image dataset loaders with automatic format detection
- **Comprehensive Metrics**:
  - Detection: mAP, IoU, precision, recall, F1
  - Classification: Top-k accuracy, confusion matrix
  - Segmentation: Pixel accuracy, mIoU, Dice coefficient
- **Visualization Tools**: Frame annotation, video generation, and multi-model comparison plots
- **REST API Server**: Deploy any model as a REST API with automatic task detection
- **Evaluation Framework**: Unified evaluation pipeline for all vision tasks

## Quick Start

### Installation

Basic installation (data loading and bbox operations only):
```bash
pip install utilities
```

With model inference support:
```bash
# HuggingFace models
pip install utilities[models-hf]

# YOLO models
pip install utilities[models-yolo]

# All models
pip install utilities[all]
```

### Basic Usage

#### Object Detection

```python
from vision_utils import create_model, VideoLoader

# Load a detection model
model = create_model("facebook/detr-resnet-50", confidence_threshold=0.7)

# Process video
loader = VideoLoader("path/to/video.mp4", sample_rate=5)
for frame_id, frame, _ in loader:
    detections = model.predict(frame)
    for det in detections:
        print(f"{det.class_name}: {det.confidence:.2f} at {det.bbox}")
```

#### Zero-Shot Classification (DINOtxt)

```python
from vision_utils import create_model
import cv2

# Load DINOtxt model
model = create_model("dinov3_vitl16_dinotxt", confidence_threshold=0.3)

# Classify with text prompts
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

classifications = model.predict(
    image,
    text_prompts=["cat", "dog", "bird"],
    top_k=3
)

for c in classifications:
    print(f"{c.class_name}: {c.confidence:.3f}")
```

#### Semantic Segmentation (DINOtxt)

```python
# Same model, different method
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["person", "background"],
    output_size=(512, 512)
)

for mask in seg_masks:
    print(f"{mask.class_name}: {mask.mask.shape}, confidence={mask.confidence:.3f}")
```

## Architecture

Vision Utilities provides a comprehensive set of modules:

### Data Structures
- **Detection**: BoundingBox, Detection, FrameDetections
- **Classification**: Classification, FrameClassifications
- **Segmentation**: SegmentationMask, FrameSegmentations

### Model Wrappers
- **Detection Models**: ObjectDetectionModel, YOLOModel, ZeroShotDetectionModel (Grounding DINO, OWL-ViT)
- **Vision-Language Models**: DINOtxtModel (classification & segmentation)
- **Factory**: `create_model()` with automatic model type detection

### Data Loading
- **Loaders**: VideoLoader, ImageDatasetLoader, DataLoader
- **Parsers**: COCOParser for annotation loading

### Evaluation Framework
- **Base Infrastructure**: BaseEvaluator with shared evaluation logic
- **Task-Specific Evaluators**:
  - ObjectDetectionEvaluator (IoU, mAP, precision/recall)
  - ClassificationEvaluator (top-k accuracy, confusion matrix)
  - SegmentationEvaluator (pixel accuracy, mIoU, Dice)
- **Metrics**: BasicMetrics, AdvancedMetrics, GPUMemoryTracker, ExecutionTimer

### Visualization & Output
- **Annotation**: FrameAnnotator for all task types
- **Video Generation**: AnnotatedVideoWriter
- **Plotting**: ResultsPlotter for multi-model comparison
- **I/O**: EvaluationConfig, ResultsWriter

### Server Deployment
- **Unified Server**: VisionServer (handles detection, classification, segmentation)
- **Client**: DetectionClient for REST API requests
- **Auto-detection**: Automatically detects task type from model output

### Additional Utilities
- **Video Processing**: Video concatenation and comparison
- **Robotics**: 3D-to-2D projection for robotics/simulation evaluation
- **Image Processing**: Stable Diffusion upscaling

## Documentation

### Getting Started
- [Getting Started Guide](getting-started.md) - Installation and first steps
- [Basic Usage Guide](guides/basic-usage.md) - Common usage patterns

### Model Guides
- [DINOtxt Guide](guides/dinotxt-guide.md) - Zero-shot classification and segmentation with DINOtxt
- [Object Detection Guide](guides/object-detection.md) - Using detection models (YOLO, DETR, Grounding DINO)

### Deployment & Evaluation
- [REST API Server Guide](guides/rest-api-server.md) - Deploy models as REST APIs
- [Evaluation Guide](guides/evaluation.md) - Evaluate models on datasets
- [Robotics Evaluation Guide](guides/robotics-evaluation.md) - 3D-to-2D projection for robotics/simulation

### Advanced Features
- [Video Processing Guide](guides/video-processing.md) - Video concatenation and comparison
- [Custom Models Guide](guides/custom-models.md) - Integrate your own models

## License

MIT License