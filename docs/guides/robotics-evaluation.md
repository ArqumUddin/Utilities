# Robotics Evaluation Guide

This guide explains how to use the robotics coordinate transformation utilities to evaluate vision models in robotics and simulation environments.

## Overview

When evaluating vision models on robots or in simulation, you often have:
- **Ground truth**: 3D object positions in world coordinates
- **Robot pose**: Robot position and orientation in world coordinates
- **Camera detections**: 2D bounding boxes from your vision model

To compare detections against ground truth, you need to project the 3D world objects into the camera's 2D image space. The robotics utilities provide functions to do this transformation.

## Core Concepts

### Coordinate Frames

The transformation pipeline involves three coordinate frames:

1. **World Frame**: Global reference frame where ground truth objects and robot pose are defined
2. **Robot Frame**: Local frame attached to the robot
3. **Camera Frame**: Frame attached to the camera (may be offset from robot)

### Transformation Pipeline

```
3D Object (World) → Robot Frame → Camera Frame → 2D Image (pixels)
```

## Data Structures

### Pose3D

Represents a pose (position + orientation) in 3D space:

```python
from vision_utils import Pose3D
import numpy as np

# Using quaternion [w, x, y, z]
robot_pose = Pose3D(
    position=np.array([1.0, 2.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
)

# Or using 3x3 rotation matrix
R = np.eye(3)  # Identity rotation
robot_pose = Pose3D(
    position=np.array([1.0, 2.0, 0.0]),
    orientation=R
)
```

### Object3D

Represents a 3D object in world space:

```python
from vision_utils import Object3D
import numpy as np

obj = Object3D(
    position=np.array([5.0, 2.0, 0.5]),      # Center position [x, y, z]
    dimensions=np.array([1.0, 0.5, 0.3]),    # [length, width, height]
    class_name="bottle",
    object_id=0  # Optional
)

# Get 8 corners of the bounding box
corners = obj.get_corners()  # Shape: (8, 3)
```

### CameraIntrinsics

Camera intrinsic parameters for projection:

```python
from vision_utils import CameraIntrinsics
import numpy as np

# Method 1: Direct parameters
camera = CameraIntrinsics(
    fx=500.0,           # Focal length x
    fy=500.0,           # Focal length y
    cx=320.0,           # Principal point x
    cy=240.0,           # Principal point y
    image_width=640,
    image_height=480
)

# Method 2: From intrinsic matrix K
K = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0]
])
camera = CameraIntrinsics.from_matrix(K, image_width=640, image_height=480)
```

## Basic Usage

### Project World Object to 2D Bounding Box

The main utility function projects a 3D object to a 2D bounding box:

```python
from vision_utils import (
    project_world_object_to_image,
    Pose3D, Object3D, CameraIntrinsics
)
import numpy as np

# Define 3D object in world
obj = Object3D(
    position=np.array([5.0, 2.0, 0.5]),
    dimensions=np.array([1.0, 0.5, 0.3]),
    class_name="bottle"
)

# Define robot pose in world
robot_pose = Pose3D(
    position=np.array([0.0, 0.0, 0.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0])
)

# Camera transform relative to robot (position + orientation)
camera_transform = (
    np.array([0.5, 0.0, 0.3]),          # Camera position in robot frame
    np.array([1.0, 0.0, 0.0, 0.0])      # Camera orientation (quaternion)
)

# Camera parameters
camera = CameraIntrinsics(
    fx=500.0, fy=500.0,
    cx=320.0, cy=240.0,
    image_width=640, image_height=480
)

# Project to 2D
projected_bbox = project_world_object_to_image(
    obj, robot_pose, camera_transform, camera
)

if projected_bbox is not None:
    print(f"Projected bbox: {projected_bbox}")
    print(f"  x: {projected_bbox.x_min:.1f} - {projected_bbox.x_max:.1f}")
    print(f"  y: {projected_bbox.y_min:.1f} - {projected_bbox.y_max:.1f}")
else:
    print("Object not visible (behind camera or out of frame)")
```

## Evaluation Workflow

### Complete Example: Evaluate Detections Against Ground Truth

```python
from vision_utils import (
    project_world_object_to_image,
    Pose3D, Object3D, CameraIntrinsics,
    Detection, AdvancedMetrics,
    create_model
)
import numpy as np
import cv2

# 1. Load your vision model
model = create_model("facebook/detr-resnet-50", confidence_threshold=0.5)

# 2. Set up camera parameters
camera = CameraIntrinsics(
    fx=500.0, fy=500.0,
    cx=320.0, cy=240.0,
    image_width=640, image_height=480
)

# 3. Define camera-to-robot transform (fixed mount)
camera_transform = (
    np.array([0.5, 0.0, 0.3]),
    np.array([1.0, 0.0, 0.0, 0.0])
)

# 4. Set up metrics
metrics = AdvancedMetrics(iou_threshold=0.5)

# 5. Process each frame
for frame_id in range(num_frames):
    # Get image
    image = cv2.imread(f"frame_{frame_id}.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get robot pose for this frame (from your robot/simulation)
    robot_pose = Pose3D(
        position=robot_positions[frame_id],
        orientation=robot_orientations[frame_id]
    )

    # Get ground truth objects in world (from your simulation/annotations)
    world_objects = [
        Object3D(
            position=np.array([5.0, 2.0, 0.5]),
            dimensions=np.array([1.0, 0.5, 0.3]),
            class_name="bottle"
        ),
        Object3D(
            position=np.array([3.0, -1.0, 0.4]),
            dimensions=np.array([0.8, 0.8, 0.6]),
            class_name="cup"
        )
    ]

    # Project ground truth to 2D
    ground_truth_detections = []
    for obj in world_objects:
        bbox = project_world_object_to_image(
            obj, robot_pose, camera_transform, camera
        )
        if bbox is not None:
            ground_truth_detections.append(Detection(
                bbox=bbox,
                class_name=obj.class_name,
                confidence=1.0,
                class_id=None
            ))

    # Get model predictions
    predicted_detections, _ = model.predict(image_rgb)

    # Add to metrics
    metrics.add_frame_results(
        predictions=predicted_detections,
        ground_truth=ground_truth_detections
    )

# 6. Calculate final metrics
results = metrics.calculate_metrics()
print(f"mAP: {results['mAP']:.3f}")
print(f"Precision: {results['overall_precision']:.3f}")
print(f"Recall: {results['overall_recall']:.3f}")

for class_name, class_metrics in results['per_class_metrics'].items():
    print(f"\n{class_name}:")
    print(f"  AP: {class_metrics['average_precision']:.3f}")
    print(f"  Precision: {class_metrics['precision']:.3f}")
    print(f"  Recall: {class_metrics['recall']:.3f}")
```

## Advanced Usage

### Transform Individual Points

```python
from vision_utils import transform_point_world_to_camera, project_3d_to_2d
import numpy as np

# Transform a point from world to camera frame
point_world = np.array([5.0, 2.0, 0.5])
point_camera = transform_point_world_to_camera(
    point_world,
    robot_pose,
    camera_transform
)

# Project to 2D image coordinates
point_2d, is_valid = project_3d_to_2d(point_camera, camera)
if is_valid:
    print(f"2D pixel coordinates: ({point_2d[0]:.1f}, {point_2d[1]:.1f})")
```

### Batch Processing Multiple Points

```python
import numpy as np

# Transform multiple points at once
points_world = np.array([
    [5.0, 2.0, 0.5],
    [3.0, -1.0, 0.4],
    [7.0, 0.0, 1.0]
])  # Shape: (3, 3)

points_camera = transform_point_world_to_camera(
    points_world,
    robot_pose,
    camera_transform
)

points_2d, valid_mask = project_3d_to_2d(points_camera, camera)
# points_2d shape: (3, 2)
# valid_mask shape: (3,) - boolean array
```

### Using 4x4 Transformation Matrix

```python
import numpy as np

# Define camera transform as 4x4 matrix
T_robot_camera = np.array([
    [1, 0, 0, 0.5],  # Last column is translation
    [0, 1, 0, 0.0],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1.0]
])

projected_bbox = project_world_object_to_image(
    obj,
    robot_pose,
    T_robot_camera,  # Pass 4x4 matrix directly
    camera
)
```

### Quaternion and Rotation Matrix Utilities

```python
from vision_utils import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion
import numpy as np

# Convert quaternion to rotation matrix
quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
R = quaternion_to_rotation_matrix(quat)

# Convert rotation matrix to quaternion
quat_back = rotation_matrix_to_quaternion(R)
```

## Handling Edge Cases

### Objects Behind Camera

Objects behind the camera (z ≤ 0 in camera frame) return `None`:

```python
bbox = project_world_object_to_image(obj, robot_pose, camera_transform, camera)
if bbox is None:
    print("Object not visible")
```

### Objects Partially Out of Frame

Bounding boxes are automatically clipped to image boundaries:

```python
# Even if object extends beyond image, bbox is clipped to valid pixel range
bbox = project_world_object_to_image(obj, robot_pose, camera_transform, camera)
# bbox.x_min >= 0, bbox.x_max <= image_width
# bbox.y_min >= 0, bbox.y_max <= image_height
```

### Invalid Bounding Boxes

If clipping results in zero area, `None` is returned:

```python
# Object completely outside image bounds
bbox = project_world_object_to_image(obj, robot_pose, camera_transform, camera)
if bbox is None:
    print("Object not in field of view")
```

## Tips and Best Practices

1. **Coordinate Frame Conventions**: Make sure your world, robot, and camera frames follow consistent conventions (e.g., ROS: x-forward, y-left, z-up)

2. **Quaternion Convention**: Uses [w, x, y, z] convention where w is the scalar part

3. **Camera Calibration**: Use accurate camera intrinsic parameters from calibration for best results

4. **Filtering Visibility**: Always check if `project_world_object_to_image` returns `None` before using the result

5. **Performance**: For batch processing many objects, consider vectorizing transformations using the batch point processing functions

6. **Validation**: Visualize projected bounding boxes on images to verify coordinate transformations are correct

## Integration with Existing Metrics

All projected bounding boxes are standard `BoundingBox` objects, so they work seamlessly with existing evaluation tools:

```python
from vision_utils import AdvancedMetrics

# Use projected ground truth with any existing metrics
metrics = AdvancedMetrics(iou_threshold=0.5)
metrics.add_frame_results(predictions, ground_truth_projected)
results = metrics.calculate_metrics()
```

## See Also

- [Basic Usage Guide](basic-usage.md) - General object detection usage
- [Metrics Documentation](../getting-started.md#metrics) - Understanding mAP and other metrics
