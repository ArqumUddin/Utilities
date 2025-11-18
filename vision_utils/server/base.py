"""
Base server utilities for hosting models via REST API.

This module provides utilities for converting models into REST API servers,
based on the VLFM/VLM server infrastructure pattern.
"""
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from typing import Any, Dict

def str_to_image(img_str: str) -> np.ndarray:
    """
    Convert base64 encoded string to numpy image.

    Args:
        img_str: Base64 encoded image string

    Returns:
        numpy array representing the image (H, W, C) in RGB format
    """
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR, convert to RGB
    return img_np

def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    """
    Convert numpy image to base64 encoded string.

    Args:
        img_np: numpy array representing the image
        quality: JPEG quality (0-100)

    Returns:
        Base64 encoded image string
    """
    if len(img_np.shape) == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    retval, buffer = cv2.imencode(".jpg", img_bgr, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str

def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Host a model as a REST API using Flask.

    The model must have a `process_payload(payload: dict) -> dict` method.

    Args:
        model: Model instance with process_payload method (must implement ServerMixin)
        name: Endpoint name (e.g., 'detect', 'yolo_world', 'gdino')
        port: Port to run the server on (default: 5000)
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        result = model.process_payload(payload)
        return jsonify(result)

    print(f"Server running at: http://localhost:{port}/{name}")

    app.run(host="localhost", port=port)
    