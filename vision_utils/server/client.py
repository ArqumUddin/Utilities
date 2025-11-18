"""
Generic client for communicating with detection servers.

This client can connect to any detection server running the vision_utils server,
similar to YOLOWorldClient but works with all model types.
"""
import numpy as np
import requests
import time
import random
from typing import List, Optional

from .base import image_to_str
from ..data.structures import Detection

class DetectionClient:
    """
    Client for communicating with detection servers.

    This client works with any server created using DetectionServer,
    regardless of the underlying model type.
    """
    def __init__(self, port: int = 5000, endpoint: str = "detect", host: str = "localhost"):
        """
        Initialize detection client.

        Args:
            port: Port where server is running (default: 5000)
            endpoint: Server endpoint name (default: "detect")
            host: Server hostname (default: "localhost")
        """
        self.url = f"http://{host}:{port}/{endpoint}"
        print(f"DetectionClient initialized for {self.url}")

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
        text_prompts: Optional[List[str]] = None,
        quality: float = 90,
        timeout: float = 20.0,
        max_retries: int = 3
    ) -> List[Detection]:
        """
        Send prediction request to server.

        Args:
            image: RGB image as numpy array (H, W, 3)
            confidence_threshold: Optional confidence threshold override
            text_prompts: Optional text prompts for zero-shot models
                         (Note: Currently prompts must be set at server startup)
            quality: JPEG quality for image encoding (0-100, default: 90)
            timeout: Request timeout in seconds (default: 20.0)
            max_retries: Maximum number of retry attempts (default: 3)

        Returns:
            List of Detection objects

        Raises:
            Exception: If request fails after all retries
        """
        payload = {
            "image": image_to_str(image, quality=quality)
        }

        if confidence_threshold is not None:
            payload["confidence_threshold"] = confidence_threshold

        if text_prompts is not None:
            payload["text_prompts"] = text_prompts

        response = self._send_request(payload, timeout=timeout, max_retries=max_retries)

        detections = []
        for det_dict in response["detections"]:
            detection = Detection.from_dict(det_dict)
            detections.append(detection)

        return detections

    def _send_request(
        self,
        payload: dict,
        timeout: float = 20.0,
        max_retries: int = 3
    ) -> dict:
        """
        Send request with retry logic.

        Args:
            payload: Request payload
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts

        Returns:
            Response dictionary

        Raises:
            Exception: If all retry attempts fail
        """
        headers = {"Content-Type": "application/json"}

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if resp.status_code == 200:
                    return resp.json()
                else:
                    error_msg = f"Request failed with status {resp.status_code}"
                    if resp.text:
                        error_msg += f": {resp.text}"
                    raise Exception(error_msg)

            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    raise Exception(f"Request timed out after {timeout}s")
                else:
                    wait_time = 2 + random.random() * 2
                    print(f"Request timed out. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    wait_time = 2 + random.random() * 2
                    print(f"Request failed: {e}. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                else:
                    wait_time = 2 + random.random() * 2
                    print(f"Error: {e}. Retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

        raise Exception("Maximum retries exceeded")
