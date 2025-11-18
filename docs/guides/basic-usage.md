# Basic Usage Examples

## Simple Video Processing

Process a video and count detections per class:

```python
from vision_utils import create_model, VideoLoader, BasicMetrics, FrameDetections

# Load model
model = create_model("facebook/detr-resnet-50", confidence_threshold=0.7)

# Load video
loader = VideoLoader("traffic.mp4", sample_rate=10)  # Process every 10th frame

# Track metrics
metrics = BasicMetrics()

# Process frames
for frame_id, frame in loader:
    # Run detection
    detections, _ = model.predict(frame)

    # Add to metrics
    frame_dets = FrameDetections(frame_id=frame_id, frame_path=None, detections=detections)
    metrics.add_frame_detections(frame_dets)

# Print summary
summary = metrics.get_summary()
print(f"Processed {summary['total_frames']} frames")
print(f"Total detections: {summary['total_detections']}")
print(f"Detection counts: {summary['detection_counts']}")
```

## Batch Processing

Process multiple images efficiently:

```python
from vision_utils import create_model, ImageDatasetLoader
import numpy as np

model = create_model("yolov8n.pt")
loader = ImageDatasetLoader("path/to/images/")

# Collect batch of frames
batch = []
filenames = []
for i, (frame, filename) in enumerate(loader):
    batch.append(frame)
    filenames.append(filename)
    
    if len(batch) == 8:  # Process in batches of 8
        all_detections, _ = model.predict_batch(batch, batch_size=8)
        
        for filename, detections in zip(filenames, all_detections):
            print(f"{filename}: {len(detections)} objects")
        
        batch = []
        filenames = []
```

## Using Ground Truth

Calculate accuracy metrics with annotations:

```python
from vision_utils import create_model, AdvancedMetrics, COCOParser

model = create_model("facebook/detr-resnet-50")
parser = COCOParser("annotations.json")
metrics = AdvancedMetrics(iou_threshold=0.5)

# Process frames
for frame_id in range(100):
    frame = load_frame(frame_id)  # Your frame loader
    
    # Get predictions
    predictions, _ = model.predict(frame)
    
    # Get ground truth
    ground_truth = parser.get_annotations(f"frame_{frame_id}.jpg")
    
    # Update metrics
    metrics.add_frame_predictions(predictions, ground_truth)

# Get results
summary = metrics.get_summary()
print(f"mAP: {summary['map']:.3f}")
print(f"Per-class AP: {summary['per_class_ap']}")
```
