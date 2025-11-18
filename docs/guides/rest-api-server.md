# REST API Server

The vision_utils library provides REST API server functionality for serving any supported model (YOLO, DETR, RT-DETR, Grounding DINO, etc.) via HTTP endpoints, following the pattern established in YOLO-World and VLFM/VLM codebases.

## Quick Start

### Install Server Dependencies

```bash
pip install -e ".[server]"
```

### Start a Server

```bash
python scripts/server_model.py --model facebook/detr-resnet-50 --port 5000
```

### Use the Client

```python
from vision_utils import DetectionClient
import cv2

client = DetectionClient(port=5000)
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

detections = client.predict(image_rgb)
for det in detections:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

## Supported Models

All models supported by the vision_utils factory can be served:

- **HuggingFace Transformers**: DETR, RT-DETR (`facebook/detr-resnet-50`)
- **YOLO**: YOLOv8, YOLO-World (`yolov8n.pt`, `yolov8n-worldv2.pt`)
- **Grounding DINO**: Zero-shot detection (`IDEA-Research/grounding-dino-base`)

## Usage Examples

### Serve Different Models

```bash
# DETR
python scripts/server_model.py --model facebook/detr-resnet-50 --port 5000

# YOLO-World with prompts
python scripts/server_model.py --model yolov8n-worldv2.pt --prompts "person,car,dog" --port 5001

# Grounding DINO
python scripts/server_model.py --model IDEA-Research/grounding-dino-base --prompts "person,car" --port 5002
```

### Python Client

**Basic Usage:**

```python
from vision_utils import DetectionClient
import cv2

# Connect to server
client = DetectionClient(port=5000, endpoint="detect")

# Load image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get predictions with custom confidence
detections = client.predict(
    image=image_rgb,
    confidence_threshold=0.7
)

# Process results
for det in detections:
    print(f"{det.class_name}: {det.confidence:.3f}")
    print(f"  BBox: {det.bbox.to_dict()}")
```

**Dynamic Prompts (Grounding DINO & YOLO-World):**

```python
from vision_utils import DetectionClient
import cv2

# Connect to zero-shot model server
client = DetectionClient(port=5002, endpoint="detect")

# Load image
image = cv2.imread("test.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect specific objects with dynamic prompts
detections = client.predict(
    image=image_rgb,
    text_prompts=["person", "bicycle", "car"],
    confidence_threshold=0.6
)

# Each request can have different prompts
animals = client.predict(
    image=image_rgb,
    text_prompts=["dog", "cat", "bird"]
)
```

### Command-Line Client

```bash
python scripts/client_model.py test_image.jpg --port 5000 --output annotated.jpg
```

## API Reference

### Request Format

```json
{
  "image": "base64_encoded_image_string",
  "confidence_threshold": 0.5,
  "caption": "person . car . dog",
  "text_prompts": ["person", "car", "dog"]
}
```

**Fields:**
- `image` (required): Base64 encoded image string
- `confidence_threshold` (optional): Override server's confidence threshold
- `caption` (optional): Dynamic prompts as period-separated string (Grounding DINO & YOLO-World only)
- `text_prompts` (optional): Dynamic prompts as list (Grounding DINO & YOLO-World only)

**Dynamic Prompts:**
- For zero-shot models (Grounding DINO, YOLO-World), you can override prompts per-request
- Use either `caption` (string: `"person . car"`) or `text_prompts` (list: `["person", "car"]`)
- If not provided, uses prompts from server initialization
- Regular models (DETR, RT-DETR, standard YOLO) ignore these fields

### Response Format

```json
{
  "frame_id": 0,
  "num_detections": 2,
  "detections": [
    {
      "bbox": {"x_min": 10.5, "y_min": 20.3, "x_max": 100.2, "y_max": 200.1},
      "class_name": "person",
      "confidence": 0.95,
      "class_id": 0
    }
  ],
  "model_name": "facebook/detr-resnet-50"
}
```

## Architecture

The server infrastructure follows this pattern:

1. **DetectionServer**: Wraps any vision_utils model
2. **host_model()**: Hosts the server using Flask
3. **DetectionClient**: Lightweight client for sending requests
4. **Serialization**: `to_dict()`/`from_dict()` for JSON compatibility

This design matches the YOLO-World and VLFM/VLM server patterns for easy integration.

## Advanced Usage

### Creating Custom Servers

```python
from vision_utils import DetectionServer, host_model, str_to_image, create_model

# Create a server with any model
model = create_model("facebook/detr-resnet-50")
server = DetectionServer(model, model_name="custom-detr")

# Host the server
host_model(server, name="custom", port=5000)
```

### Multi-Server Deployment

Run multiple models simultaneously:

```bash
# Terminal 1
python scripts/server_model.py --model facebook/detr-resnet-50 --port 5000

# Terminal 2
python scripts/server_model.py --model yolov8n.pt --port 5001
```

## Performance Tips

1. **GPU Allocation**: Each server loads one model into GPU memory
2. **Image Quality**: Lower JPEG quality (60-80) for faster transfers
3. **Batch Processing**: For multiple images, consider local inference instead
4. **Timeouts**: Increase client timeout for large images or slow models

## See Also

- [Server Usage Guide](server-usage.md) - Comprehensive usage guide
- [YOLO-World Server](https://github.com/ArqumUddin/YOLO-World) - YOLO World Raw Inference Server
