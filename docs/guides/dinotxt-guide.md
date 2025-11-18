# DINOtxt Vision-Language Model Guide

This guide covers how to use DINOtxt with `vision_utils` for zero-shot image classification and semantic segmentation using text prompts.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Classification](#classification)
- [Segmentation](#segmentation)
- [Server Deployment](#server-deployment)
- [Evaluation](#evaluation)
- [Advanced Usage](#advanced-usage)

## Overview

DINOtxt is a vision-language model built on DINOv3 that enables:
- **Zero-shot image classification**: Classify images using natural language descriptions
- **Patch-level segmentation**: Generate segmentation masks for specific concepts described in text
- **Dynamic text prompts**: Change prompts at inference time without retraining

The model uses contrastive learning to align visual and text representations, allowing it to understand images through text descriptions.

## Installation

DINOtxt requires the `dinov3` package and BasicSR for image processing utilities.

### Step 1: Create Conda Environment

```bash
cd path/to/Utilities
conda env create -f environment.yml
conda activate vision_utils
```

The `dinov3` package is automatically installed from GitHub as part of the environment setup.

### Step 2: Install BasicSR

BasicSR provides image processing utilities used by the vision_utils package:

```bash
# Clone BasicSR repository
git clone https://github.com/ArqumUddin/BasicSR.git
cd BasicSR

# Install in development mode
pip install -e .

# Return to utilities directory
cd ..
```

### Step 3: Install vision_utils

```bash
cd path/to/Utilities
pip install -e .
```

Now you're ready to use DINOtxt and all other vision_utils features!

## Quick Start

### Basic Classification

```python
from vision_utils import create_model
import cv2

# Load model
model = create_model(
    model_name="dinov3_vitl16_dinotxt",
    confidence_threshold=0.3,
    text_prompts=["cat", "dog", "bird", "car"]
)

# Load image
image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run classification
classifications = model.predict(image, text_prompts=["cat", "dog", "car"], top_k=3)

# Print results
for c in classifications:
    print(f"{c.class_name}: {c.confidence:.3f}")
```

### Basic Segmentation

```python
# Same model, different method
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["person", "background"],
    output_size=(512, 512)
)

# Access masks
for mask in seg_masks:
    print(f"{mask.class_name}: {mask.mask.shape}, confidence={mask.confidence:.3f}")
    # mask.mask is a numpy array with shape (H, W) and values in [0, 1]
```

## Classification

### Zero-Shot Image Classification

DINOtxt performs zero-shot classification by computing similarity between image features and text features.

```python
from vision_utils import DINOtxtModel
import numpy as np

# Initialize model
model = DINOtxtModel(
    model_name="dinov3_vitl16_dinotxt",
    confidence_threshold=0.2,
    device="cuda"  # or "cpu"
)

# Define text prompts (can be any descriptions)
prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a landscape photo",
    "a city street"
]

# Classify image
classifications = model.predict(
    image,
    text_prompts=prompts,
    top_k=5  # Return top 5 predictions
)

# Results are sorted by confidence
for c in classifications:
    print(f"Class: {c.class_name}")
    print(f"Confidence: {c.confidence:.4f}")
    print(f"Class ID: {c.class_id}")
    print()
```

### Dynamic Prompts

One of DINOtxt's key features is the ability to change prompts at inference time:

```python
# Initial prompts
results1 = model.predict(image, text_prompts=["cat", "dog"])

# Different prompts on same image
results2 = model.predict(image, text_prompts=["happy", "sad", "neutral"])

# Scene understanding
results3 = model.predict(image, text_prompts=["indoor scene", "outdoor scene"])
```

### Batch Classification

Process multiple images efficiently:

```python
images = [cv2.cvtColor(cv2.imread(f"img_{i}.jpg"), cv2.COLOR_BGR2RGB)
          for i in range(10)]

results = model.predict_batch(
    images,
    text_prompts=["cat", "dog", "bird"],
    batch_size=8,
    top_k=3
)

# results is a list of lists: [[Classification, ...], [Classification, ...], ...]
for i, img_results in enumerate(results):
    print(f"Image {i}: {img_results[0].class_name} ({img_results[0].confidence:.3f})")
```

## Segmentation

### Patch-Level Segmentation

DINOtxt generates segmentation masks by computing patch-level similarity between image patches and text descriptions.

```python
from vision_utils import DINOtxtModel
import matplotlib.pyplot as plt

model = DINOtxtModel(
    model_name="dinov3_vitl16_dinotxt",
    confidence_threshold=0.3
)

# Generate segmentation masks
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["person", "background", "sky"],
    output_size=(512, 512)  # Output resolution
)

# Visualize masks
fig, axes = plt.subplots(1, len(seg_masks), figsize=(15, 5))
for i, mask in enumerate(seg_masks):
    axes[i].imshow(mask.mask, cmap='hot')
    axes[i].set_title(f"{mask.class_name} ({mask.confidence:.3f})")
    axes[i].axis('off')
plt.show()
```

### Convert Masks to Bounding Boxes

```python
# Generate segmentation mask
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["car", "person"],
    output_size=(512, 512)
)

# Convert masks to bounding boxes
for mask in seg_masks:
    bbox = mask.to_bounding_box(threshold=0.5)
    if bbox:
        print(f"{mask.class_name}: [{bbox.x_min}, {bbox.y_min}, {bbox.x_max}, {bbox.y_max}]")
```

### Custom Output Size

Control the resolution of output masks:

```python
# High resolution masks
seg_masks_hires = model.predict_segmentation(
    image,
    text_prompts=["person"],
    output_size=(1024, 1024)
)

# Low resolution (faster)
seg_masks_lowres = model.predict_segmentation(
    image,
    text_prompts=["person"],
    output_size=(256, 256)
)
```

## Server Deployment

Deploy DINOtxt as a REST API server using the unified `VisionServer`.

### Python Server (Flask/FastAPI)

```python
from vision_utils import host_model

# Host DINOtxt model
host_model(
    model_name="dinov3_vitl16_dinotxt",
    confidence_threshold=0.3,
    device="cuda",
    text_prompts=["default", "prompts"]  # Optional defaults
)
```

The server will start on `http://localhost:5000`.

### Making Requests

#### Classification Request

```python
from vision_utils import DetectionClient
import cv2

client = DetectionClient(server_url="http://localhost:5000")

# Load and encode image
image = cv2.imread("image.jpg")

# Classification request
response = client.predict(
    image,
    text_prompts=["cat", "dog", "bird"],
    top_k=5
)

# Response format
{
    "frame_id": 0,
    "frame_path": null,
    "num_classifications": 5,
    "classifications": [
        {"class_name": "cat", "confidence": 0.87, "class_id": 0},
        {"class_name": "dog", "confidence": 0.09, "class_id": 1},
        ...
    ],
    "top_prediction": {"class_name": "cat", "confidence": 0.87, "class_id": 0},
    "model_name": "dinov3_vitl16_dinotxt"
}
```

#### Segmentation Request

```python
# Segmentation request
response = client.predict(
    image,
    text_prompts=["person", "background"],
    output_size=[512, 512]  # Triggers segmentation mode
)

# Response format
{
    "frame_id": 0,
    "frame_path": null,
    "num_masks": 2,
    "segmentation_masks": [
        {
            "mask": [[0.1, 0.2, ...], ...],  # 2D array
            "mask_shape": [512, 512],
            "class_name": "person",
            "confidence": 0.75,
            "class_id": 0,
            "bounding_box": {"x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}
        },
        ...
    ],
    "model_name": "dinov3_vitl16_dinotxt"
}
```

### cURL Examples

```bash
# Classification
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "text_prompts": ["cat", "dog"],
    "top_k": 3
  }'

# Segmentation
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "text_prompts": ["person", "background"],
    "output_size": [512, 512]
  }'
```

## Evaluation

Evaluate DINOtxt on classification or segmentation datasets using the evaluation framework.

### Classification Evaluation

#### 1. Create Configuration File

Create `configs/dinotxt_classification.yaml`:

```yaml
# Model configuration
model_name: "dinov3_vitl16_dinotxt"
confidence_threshold: 0.3
device: "cuda"
text_prompts: ["cat", "dog", "bird", "car", "person"]

# Dataset configuration
input_type: "images"
input_images: "data/classification_dataset/"

# Ground truth (optional)
ground_truth_path: "data/annotations_classification.json"
ground_truth_format: "coco"

# Output configuration
output_directory: "results/dinotxt_classification/"
save_annotated_frames: true
generate_video: false

# Display name for results
display_name: "DINOtxt Classification"
```

#### 2. Run Evaluation

```bash
# Auto-detects classification task
python scripts/evaluate.py --config configs/dinotxt_classification.yaml

# Or explicitly specify
python scripts/evaluate.py --config configs/dinotxt_classification.yaml --task classification
```

#### 3. Results

The evaluation outputs:
- **Top-k accuracy**: Top-1, Top-3, Top-5 accuracy
- **Per-class accuracy**: Accuracy for each class
- **Confusion matrix**: Predicted vs ground truth classes
- **GPU memory usage**: Peak memory consumption
- **Inference timing**: Average time per image

Example output:
```
Classification Evaluation Results:
- Top-1 Accuracy: 85.3%
- Top-3 Accuracy: 96.7%
- Top-5 Accuracy: 98.2%
- Total samples: 1000
- Average inference time: 45ms

Per-class accuracy:
  cat: 87.5% (175/200 correct)
  dog: 83.0% (166/200 correct)
  bird: 86.0% (172/200 correct)
```

### Segmentation Evaluation

#### 1. Create Configuration File

Create `configs/dinotxt_segmentation.yaml`:

```yaml
# Model configuration
model_name: "dinov3_vitl16_dinotxt"
confidence_threshold: 0.3
device: "cuda"
text_prompts: ["person", "car", "background"]
mask_threshold: 0.5  # Threshold for binarizing masks

# Dataset configuration
input_type: "images"
input_images: "data/segmentation_dataset/"

# Ground truth
ground_truth_path: "data/annotations_segmentation.json"
ground_truth_format: "coco"

# Output configuration
output_directory: "results/dinotxt_segmentation/"
save_annotated_frames: true

display_name: "DINOtxt Segmentation"
```

#### 2. Run Evaluation

```bash
python scripts/evaluate.py --config configs/dinotxt_segmentation.yaml --task segmentation
```

#### 3. Results

The evaluation outputs:
- **Pixel accuracy**: Percentage of correctly classified pixels
- **Mean IoU (mIoU)**: Average Intersection over Union across classes
- **Per-class IoU**: IoU for each class
- **Dice coefficient**: Per-class Dice scores

Example output:
```
Segmentation Evaluation Results:
- Pixel Accuracy: 92.3%
- Mean IoU: 67.5%
- Mean Dice: 78.2%
- Total pixels evaluated: 50,000,000

Per-class metrics:
  person: IoU=72.3%, Dice=82.1%
  car: IoU=68.5%, Dice=79.8%
  background: IoU=61.7%, Dice=72.7%
```

## Advanced Usage

### Custom Confidence Thresholds

```python
model = DINOtxtModel(
    model_name="dinov3_vitl16_dinotxt",
    confidence_threshold=0.3  # Initial threshold
)

# Override at inference time
high_conf_results = model.predict(image, text_prompts=["cat", "dog"])
# Uses 0.3 threshold

# Temporarily change threshold
model.confidence_threshold = 0.5
strict_results = model.predict(image, text_prompts=["cat", "dog"])
# Uses 0.5 threshold
```

### Using with ModelInferenceEngine

Process entire datasets efficiently:

```python
from vision_utils import ModelInferenceEngine, DataLoader, DINOtxtModel

# Create model
model = DINOtxtModel(
    model_name="dinov3_vitl16_dinotxt",
    text_prompts=["cat", "dog", "bird"]
)

# Create inference engine
engine = ModelInferenceEngine(model)

# Create data loader
loader = DataLoader(
    input_path="data/images/",
    input_type="images",
    frame_sampling_rate=1
)

# Process all images
for frame_id, image, image_path in loader:
    classifications, padding_info = engine.process_frame(
        image,
        frame_id,
        return_padding_info=True
    )
    print(f"{image_path}: {classifications[0].class_name} ({classifications[0].confidence:.3f})")
```

### Image Padding

DINOtxt automatically pads images to square before processing. Access padding information:

```python
classifications, padding_info = model.predict(
    image,
    text_prompts=["cat", "dog"],
    return_padding_info=True
)

print(f"Original size: {padding_info.original_width}x{padding_info.original_height}")
print(f"Padded size: {padding_info.padded_width}x{padding_info.padded_height}")
print(f"Padding: top={padding_info.pad_top}, bottom={padding_info.pad_bottom}")
```

### Model Information

```python
# Get model size
size_mb = model.get_model_size_mb()
print(f"Model size: {size_mb:.2f} MB")

# Get current text prompts
prompts = model.get_class_names()
print(f"Current prompts: {prompts}")

# Get number of classes
num_classes = model.get_num_classes()
print(f"Number of classes: {num_classes}")

# Model details
print(model)
# Output: DINOtxtModel(model=dinov3_vitl16_dinotxt, device=cuda, threshold=0.3, prompts=['cat', 'dog', 'bird'])
```

## Tips and Best Practices

### Prompt Engineering

1. **Be specific**: Use descriptive prompts like "a photo of a cat" rather than just "cat"
2. **Consider context**: For scene classification, use "indoor scene", "outdoor scene" rather than "indoor", "outdoor"
3. **Use consistent format**: Keep prompt formats consistent across categories

```python
# Good
prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
]

# Less effective
prompts = ["cat", "dog picture", "bird in nature"]
```

### Performance Optimization

1. **Batch processing**: Use `predict_batch()` for multiple images
2. **Output size**: Use smaller output sizes for segmentation when high resolution isn't needed
3. **Device selection**: Use GPU for faster inference

```python
# Faster inference with lower resolution
seg_masks = model.predict_segmentation(
    image,
    text_prompts=["person"],
    output_size=(256, 256)  # Instead of (1024, 1024)
)
```

### Confidence Threshold Tuning

- **Classification**: Start with 0.2-0.3, adjust based on precision/recall needs
- **Segmentation**: Start with 0.5 for mask binarization, adjust for stricter/looser masks

## Troubleshooting

### Model Loading Issues

If you encounter errors loading the model:

```python
# Ensure dinov3 is installed
import dinov3
print(dinov3.__version__)

# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Memory Issues

If you run out of GPU memory:

1. Reduce output size for segmentation
2. Process images individually instead of batches
3. Use CPU instead of GPU (slower but uses system RAM)

```python
# Use CPU
model = DINOtxtModel(
    model_name="dinov3_vitl16_dinotxt",
    device="cpu"
)
```

## See Also

- [REST API Documentation](../rest-api.md)
- [Evaluation Guide](./evaluation.md)
- [Server Deployment Guide](./server-deployment.md)
- [Main Documentation](../index.md)
