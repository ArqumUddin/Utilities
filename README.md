# Vision Utilities

A comprehensive Python library for computer vision tasks including object detection, image classification, and semantic segmentation. Provides modular utilities for data loading, model inference, evaluation, and deployment.

## Features

- **Multiple Vision Tasks**: Object detection, image classification, semantic segmentation
- **Modular Architecture**: Clean separation between data, models, metrics, and visualization
- **Multiple Model Support**:
  - **Detection**: HuggingFace Transformers (DETR, RT-DETR), YOLO, Grounding DINO, OWL-ViT
  - **Classification & Segmentation**: DINOtxt vision-language models
- **Zero-Shot Capabilities**: Text-guided detection and segmentation
- **Flexible Data Loading**: Video and image dataset loaders with automatic format detection
- **Comprehensive Evaluation**:
  - Detection: mAP, IoU, precision, recall, F1-score
  - Classification: Top-k accuracy, confusion matrix
  - Segmentation: Pixel accuracy, mIoU, Dice coefficient
- **REST API Server**: Deploy any model with automatic task detection
- **Visualization Tools**: Frame annotation, video generation, multi-model comparison plots

## Installation

### Complete Installation (Recommended)

Follow these steps to install vision_utils with all dependencies:

#### Step 1: Create Conda Environment

```bash
cd path/to/Utilities
conda env create -f environment.yml
conda activate utilities
```

This installs all core dependencies including PyTorch, CUDA libraries, and the DINOv3 package.

#### Step 2: Install BasicSR

BasicSR provides image processing utilities used by the vision_utils package:

```bash
# Clone BasicSR repository
git clone https://github.com/ArqumUddin/BasicSR.git
cd BasicSR

# Install in development mode
pip install -e .

# Return to utilities directory
cd ../path/to/Utilities
```

#### Step 3: Install vision_utils

```bash
# Install in development mode
pip install -e .
```

Now you're ready to use all vision_utils features including object detection, classification, and segmentation!

### Alternative: Pip Installation (Basic Features Only)

For basic features without DINOtxt support:

```bash
pip install utilities
```

## Quick Start

### Basic Usage

```python
from vision_utils import create_model, VideoLoader, BasicMetrics, FrameDetections

# Load a model (automatically detects model type)
model = create_model("facebook/detr-resnet-50", confidence_threshold=0.7)

# Load video frames
loader = VideoLoader("video.mp4", sample_rate=5)

# Track metrics
metrics = BasicMetrics()

# Process frames
for frame_id, frame in loader:
    detections, _ = model.predict(frame)

    frame_dets = FrameDetections(frame_id=frame_id, frame_path=None, detections=detections)
    metrics.add_frame_detections(frame_dets)

# Get summary
summary = metrics.get_summary()
print(f"Total frames: {summary['total_frames']}")
print(f"Average detections per frame: {summary['average_detections_per_frame']:.2f}")
print(f"Inference FPS: {summary['inference_fps']:.2f}")
```

### Supported Models

#### Object Detection
```python
# HuggingFace Transformers models
model = create_model("facebook/detr-resnet-50")
model = create_model("PekingU/rtdetr_r50vd")

# YOLO models
model = create_model("yolov8n.pt")

# Grounding DINO (zero-shot with text prompts)
model = create_model(
    "IDEA-Research/grounding-dino-base",
    text_prompts=["person", "car", "dog"],
    confidence_threshold=0.35
)
```

#### Image Classification & Segmentation (DINOtxt)
```python
import cv2

# Load DINOtxt model
model = create_model("dinov3_vitl16_dinotxt", confidence_threshold=0.3)

# Zero-shot classification
image = cv2.cvtColor(cv2.imread("image.jpg"), cv2.COLOR_BGR2RGB)
classifications = model.predict(
    image,
    text_prompts=["cat", "dog", "bird"],
    top_k=3
)

# Patch-level segmentation
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["person", "background"],
    output_size=(512, 512)
)
```

### Batch Processing

```python
from vision_utils import create_model, ImageDatasetLoader

model = create_model("yolov8n.pt")
loader = ImageDatasetLoader("path/to/images/")

# Collect batch
batch = []
for i, (frame, filename) in enumerate(loader):
    batch.append(frame)

    if len(batch) == 8:  # Process in batches of 8
        all_detections, _ = model.predict_batch(batch, batch_size=8)
        for filename, detections in zip(filenames, all_detections):
            print(f"{filename}: {len(detections)} objects detected")
        batch = []
```

### Advanced Metrics with Ground Truth

```python
from vision_utils import AdvancedMetrics, COCOParser

# Load ground truth
parser = COCOParser("annotations.json")
metrics = AdvancedMetrics(iou_threshold=0.5)

# Evaluate
for frame_id in range(num_frames):
    frame = load_frame(frame_id)
    predictions, _ = model.predict(frame)
    ground_truth = parser.get_annotations(f"frame_{frame_id}.jpg")

    metrics.add_frame_predictions(predictions, ground_truth)

# Get results
summary = metrics.get_summary()
print(f"mAP: {summary['map']:.3f}")
print(f"Precision: {summary['precision']:.3f}")
print(f"Recall: {summary['recall']:.3f}")
print(f"F1 Score: {summary['f1_score']:.3f}")
```

### Visualization

```python
from vision_utils import FrameAnnotator, AnnotatedVideoWriter

# Annotate frames
annotator = FrameAnnotator()
annotated_frame = annotator.annotate_frame(frame, detections)

# Create annotated video
with AnnotatedVideoWriter("output.mp4", fps=30) as writer:
    for frame_id, frame in loader:
        detections, _ = model.predict(frame)
        annotated = annotator.annotate_frame(frame, detections)
        writer.write_frame(annotated)
```

## Library Architecture

```
vision_utils/
├── data/               # Data loading and structures
│   ├── loaders.py      # VideoLoader, ImageDatasetLoader
│   ├── structures.py   # Detection, FrameDetections, DataAnnotation
│   ├── parsers.py      # COCOParser
│   └── padding.py      # ImagePaddingInfo
├── models/             # Model wrappers
│   ├── factory.py      # create_model()
│   ├── huggingface.py  # ObjectDetectionModel
│   ├── yolo.py         # YOLOModel
│   ├── grounding_dino.py  # GroundingDINOModel
│   └── inference.py    # ModelInferenceEngine
├── metrics/            # Metrics calculation
│   ├── basic.py        # BasicMetrics
│   ├── advanced.py     # AdvancedMetrics (mAP, precision, recall)
│   ├── timing.py       # ExecutionTimer
│   └── gpu_memory.py   # GPUMemoryTracker
├── visualization/      # Visualization tools
│   ├── annotator.py    # FrameAnnotator
│   ├── video_writer.py # AnnotatedVideoWriter
│   └── plotter.py      # ResultsPlotter
├── io/                 # Configuration and results
│   ├── config.py       # EvaluationConfig
│   └── results.py      # ResultsWriter
├── utils/              # Utility functions
│   ├── bbox.py         # BoundingBox, calculate_iou
│   └── matching.py     # match_detections_to_ground_truth
└── evaluation/         # End-to-end evaluation
    └── evaluator.py    # ObjectDetectionEvaluator
```

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Usage Guides](docs/guides/)
  - [Basic Usage](docs/guides/basic-usage.md)
  - [Video Processing](docs/guides/video-processing.md)
  - [REST API Server](docs/guides/rest-api-server.md)
  - [Server Usage](docs/guides/server-usage.md)
  - [Using Example Scripts](docs/guides/using-example-scripts.md)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
