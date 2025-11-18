# REST API Server Usage Guide

This guide explains how to use the `server_model.py` script to serve vision_utils models via REST API.

## Quick Start

### Start a Server

```bash
# Serve DETR model
python scripts/server_model.py --model facebook/detr-resnet-50 --port 5000

# Serve RT-DETR
python scripts/server_model.py --model facebook/detr-resnet-101 --port 5001

# Serve YOLO
python scripts/server_model.py --model yolov8n.pt --port 5002

# Serve YOLO-World with prompts
python scripts/server_model.py --model yolov8n-worldv2.pt --prompts "person,car,dog" --port 5003

# Serve Grounding DINO
python scripts/server_model.py --model IDEA-Research/grounding-dino-base --prompts "person,car,bicycle,dog" --port 5004
```

### Use the Client

```bash
# Test with an image
python scripts/client_example.py test_image.jpg --port 5000

# Save annotated output
python scripts/client_example.py test_image.jpg --port 5000 --output annotated.jpg

# Override confidence threshold
python scripts/client_example.py test_image.jpg --port 5000 --confidence 0.7
```

## Command-Line Options

### Server (`server_model.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model name or path (required) | - |
| `--port` | Server port | 5000 |
| `--endpoint` | Endpoint name | detect |
| `--confidence` | Confidence threshold | 0.5 |
| `--device` | Device (cuda/cpu) | auto-detect |
| `--prompts` | Text prompts for zero-shot models | None |
| `--max-detections` | Maximum detections per image | 100 |
| `--nms-threshold` | NMS IoU threshold | 0.5 |

### Client (`client_example.py`)

| Option | Description | Default |
|--------|-------------|---------|
| `image` | Path to input image (required) | - |
| `--port` | Server port | 5000 |
| `--endpoint` | Server endpoint | detect |
| `--confidence` | Confidence threshold override | None |
| `--output` | Path to save annotated image | None |

## API Specification

### Request Format

**Endpoint:** `POST http://localhost:{port}/{endpoint}`

**Headers:**
```
Content-Type: application/json
```

**Body:**
```json
{
  "image": "base64_encoded_image_string",
  "confidence_threshold": 0.5,  // optional
  "caption": "person . car . dog",  // optional: dynamic prompts (string format)
  "text_prompts": ["person", "car", "dog"]  // optional: dynamic prompts (list format)
}
```

**Dynamic Prompts (Grounding DINO & YOLO-World only):**
- Use `caption` field with period-separated string: `"person . car . dog"`
- Or use `text_prompts` field with list: `["person", "car", "dog"]`
- If provided, overrides server initialization prompts for that request
- If not provided, uses prompts from server startup (via `--prompts` flag)
- Regular YOLO/DETR models ignore these fields

### Response Format

```json
{
  "frame_id": 0,
  "frame_path": null,
  "num_detections": 2,
  "detections": [
    {
      "bbox": {
        "x_min": 10.5,
        "y_min": 20.3,
        "x_max": 100.2,
        "y_max": 200.1,
        "width": 89.7,
        "height": 179.8
      },
      "class_name": "person",
      "confidence": 0.95,
      "class_id": 0
    }
  ],
  "model_name": "facebook/detr-resnet-50"
}
```

## Using with curl

```bash
# Encode image to base64
base64 -w 0 test_image.jpg > image.b64

# Send request
curl -X POST http://localhost:5000/detect \\
  -H "Content-Type: application/json" \\
  -d "{\"image\": \"$(cat image.b64)\", \"confidence_threshold\": 0.5}"
```

## Programmatic Usage (Python)

### Basic Usage

```python
from vision_utils import DetectionClient
import cv2

# Connect to server
client = DetectionClient(port=5000)

# Load and convert image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get predictions
detections = client.predict(image_rgb, confidence_threshold=0.7)

# Process results
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
    print(f"  BBox: {det.bbox}")
```

### Dynamic Prompts (Grounding DINO & YOLO-World)

```python
from vision_utils import DetectionClient
import cv2

# Connect to Grounding DINO or YOLO-World server
client = DetectionClient(port=5002)

# Load image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect specific objects with dynamic prompts
detections = client.predict(
    image_rgb,
    text_prompts=["person", "bicycle", "car"],
    confidence_threshold=0.6
)

# Each request can have different prompts
detections_animals = client.predict(
    image_rgb,
    text_prompts=["dog", "cat", "bird"]
)

# Process results
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

## Multi-Server Deployment

You can run multiple servers on different ports for different models:

```bash
# Terminal 1: DETR
python scripts/server_model.py --model facebook/detr-resnet-50 --port 5000

# Terminal 2: YOLO
python scripts/server_model.py --model yolov8n.pt --port 5001

# Terminal 3: Grounding DINO
python scripts/server_model.py --model IDEA-Research/grounding-dino-base --prompts "person,car" --port 5002
```

Then use different clients:

```python
detr_client = DetectionClient(port=5000)
yolo_client = DetectionClient(port=5001)
gdino_client = DetectionClient(port=5002)
```

## Tips

1. **GPU Memory**: Each server loads a model into GPU memory. Monitor with `nvidia-smi`
2. **Ports**: Use different ports for each model server
3. **Static vs Dynamic Prompts**:
   - Set default prompts at server startup with `--prompts` flag
   - Override per-request using `caption` or `text_prompts` in payload
   - Dynamic prompts only work with Grounding DINO and YOLO-World
4. **Quality**: Lower JPEG quality in client (60-80) for faster transfers
5. **Timeouts**: Increase timeout for large images or slow models

## Troubleshooting

**"Connection refused"**
- Ensure server is running
- Check port number matches
- Verify firewall settings

**"Model not found"**
- Check model name/path is correct
- For HuggingFace models, ensure you're online for first download
- For local models (.pt files), verify file exists

**"CUDA out of memory"**
- Use `--device cpu` for CPU inference
- Close other GPU processes
- Use smaller model variant

**Slow inference**
- Use GPU with `--device cuda`
- Reduce image resolution before sending
- Lower `--max-detections` limit
