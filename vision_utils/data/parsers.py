"""
Ground truth annotation parser for COCO dataset
"""
import os
import json
from typing import Dict, List, Optional
from pathlib import Path
from .structures import DataAnnotation, Detection
from ..utils.bbox import BoundingBox

class COCOParser():
    """
    Parser for COCO format annotations.
    """
    def __init__(self, annotations_path: str):
        """
        Initialize parser.

        Args:
            annotations_path: Path to annotations file or directory
        """
        self.annotations_path = annotations_path
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")

    def parse(self) -> Dict[str, DataAnnotation]:
        """Parse COCO format annotations."""
        with open(self.annotations_path, 'r') as f:
            coco_data = json.load(f)

        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        images = {img['id']: img for img in coco_data.get('images', [])}

        annotations_by_image = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        result = {}
        for image_id, image_info in images.items():
            file_name = image_info['file_name']
            frame_id = image_id

            detections = []
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    x, y, w, h = ann['bbox'] # COCO bbox format: [x, y, width, height]
                    bbox = BoundingBox.from_xywh(x, y, w, h)

                    category_id = ann['category_id']
                    class_name = categories.get(category_id, f"class_{category_id}")

                    detection = Detection(
                        bbox=bbox,
                        class_name=class_name,
                        confidence=1.0,  # Ground truth has confidence 1.0
                        class_id=category_id
                    )
                    detections.append(detection)

            gt_annotation = DataAnnotation(
                frame_id=frame_id,
                frame_path=file_name,
                annotations=detections
            )
            result[file_name] = gt_annotation

        return result
