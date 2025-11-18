# Getting Started

This guide will help you get started with Vision Utilities.

## Installation

Vision Utilities uses optional dependency groups to keep installations lightweight. Install only what you need:

### Core Installation

The core package includes data loading, bounding box operations, and basic utilities:

```bash
pip install utilities
```

## Basic Concepts

### Data Structures

Vision Utilities provides clean data structures for working with detections:

```python
from vision_utils import BoundingBox, Detection

# Create a bounding box
bbox = BoundingBox(x_min=100, y_min=150, x_max=200, y_max=250)

# Create a detection
detection = Detection(
    bbox=bbox,
    class_name="person",
    confidence=0.95
)

# Access properties
print(f"Area: {bbox.area}")
print(f"Center: {bbox.center}")
```

### Loading Data

Load images or video frames with automatic format detection:

```python
from vision_utils import VideoLoader, ImageDatasetLoader

# Load video frames
video_loader = VideoLoader("video.mp4", sample_rate=5)
for frame_id, frame in video_loader:
    # frame is a numpy array (H, W, 3) in RGB format
    pass

# Load images from directory
image_loader = ImageDatasetLoader("path/to/images/")
for frame_id, (frame, filename) in enumerate(image_loader):
    # Process images
    pass
```

### Model Inference

Use the factory function to automatically create the right model wrapper:

```python
from vision_utils import create_model

# HuggingFace model
model = create_model("facebook/detr-resnet-50", confidence_threshold=0.5)

# YOLO model
model = create_model("yolov8n.pt", confidence_threshold=0.5)

# Grounding DINO (zero-shot with text prompts)
model = create_model(
    "IDEA-Research/grounding-dino-base",
    text_prompts=["person", "car", "dog"],
    confidence_threshold=0.35
)

# Run inference
detections, _ = model.predict(frame)
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

### Calculating Metrics

Track performance without ground truth:

```python
from vision_utils import BasicMetrics

metrics = BasicMetrics()

# Add detections from each frame
for frame_id, frame in loader:
    detections, _ = model.predict(frame)
    frame_detections = FrameDetections(
        frame_id=frame_id,
        frame_path=None,
        detections=detections
    )
    metrics.add_frame_detections(frame_detections)

# Get summary
summary = metrics.get_summary()
print(f"Total frames: {summary['total_frames']}")
print(f"Total detections: {summary['total_detections']}")
print(f"Average FPS: {summary['inference_fps']:.2f}")
```

Or calculate accuracy metrics with ground truth:

```python
from vision_utils import AdvancedMetrics

metrics = AdvancedMetrics(iou_threshold=0.5)

# Add predictions and ground truth for each frame
for frame_id, frame in loader:
    predictions, _ = model.predict(frame)
    ground_truth = load_annotations(frame_id)  # Your annotation loader

    metrics.add_frame_predictions(predictions, ground_truth)

# Get metrics
summary = metrics.get_summary()
print(f"mAP: {summary['map']:.3f}")
print(f"Precision: {summary['precision']:.3f}")
print(f"Recall: {summary['recall']:.3f}")
```

### Visualization

Annotate frames with detection boxes:

```python
from vision_utils import FrameAnnotator

annotator = FrameAnnotator()

# Annotate frame with detections
annotated = annotator.annotate_frame(
    frame,
    detections,
    ground_truth=None  # Optional
)

# Save or display
cv2.imwrite("annotated.jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
```

## Next Steps

- Check out [Guides](guides/basic-usage.md) for common usage patterns
