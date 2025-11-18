"""
Coordinate transformation utilities for robotics applications.

Provides functions to transform 3D objects from world coordinates to camera
coordinates and project them to 2D image coordinates.
"""
from typing import Optional, Tuple, Union
import numpy as np

from ..utils.bbox import BoundingBox
from .structures import Pose3D, Object3D, CameraIntrinsics
from .rotations import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

def create_transform_matrix(
    position: np.ndarray,
    orientation: np.ndarray
) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix from position and orientation.

    Args:
        position: 3D position as [x, y, z]
        orientation: Quaternion [w, x, y, z] or 3x3 rotation matrix

    Returns:
        4x4 homogeneous transformation matrix
    """
    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)

    if position.shape != (3,):
        raise ValueError(f"Position must be shape (3,), got {position.shape}")

    if orientation.shape == (4,):
        R = quaternion_to_rotation_matrix(orientation)
    elif orientation.shape == (3, 3):
        R = orientation
    else:
        raise ValueError(
            f"Orientation must be quaternion (4,) or rotation matrix (3,3), "
            f"got {orientation.shape}"
        )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = position

    return T

def transform_point_world_to_camera(
    point_world: np.ndarray,
    robot_pose: Pose3D,
    camera_to_robot_transform: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    Transform a 3D point from world coordinates to camera coordinates.

    Args:
        point_world: 3D point in world frame, shape (3,) or (N, 3)
        robot_pose: Robot pose in world frame (position + orientation)
        camera_to_robot_transform: Either:
            - 4x4 transformation matrix from camera to robot
            - Tuple of (position, orientation) for camera in robot frame

    Returns:
        3D point in camera frame, same shape as input
    """
    point_world = np.asarray(point_world, dtype=np.float64)
    is_single_point = (point_world.ndim == 1)

    if is_single_point:
        if point_world.shape != (3,):
            raise ValueError(f"Point must be shape (3,), got {point_world.shape}")
        point_world = point_world.reshape(1, 3)
    elif point_world.ndim == 2:
        if point_world.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {point_world.shape}")
    else:
        raise ValueError(f"Point must be 1D or 2D array, got shape {point_world.shape}")

    T_world_robot = create_transform_matrix(robot_pose.position, robot_pose.orientation)
    if isinstance(camera_to_robot_transform, np.ndarray):
        if camera_to_robot_transform.shape == (4, 4):
            T_robot_camera = camera_to_robot_transform
        else:
            raise ValueError(
                f"If providing matrix, must be 4x4, got {camera_to_robot_transform.shape}"
            )
    elif isinstance(camera_to_robot_transform, (tuple, list)):
        if len(camera_to_robot_transform) != 2:
            raise ValueError("Tuple must be (position, orientation)")
        cam_pos, cam_orient = camera_to_robot_transform
        T_robot_camera = create_transform_matrix(cam_pos, cam_orient)
    else:
        raise ValueError(
            "camera_to_robot_transform must be 4x4 matrix or (position, orientation) tuple"
        )

    T_world_camera = T_world_robot @ T_robot_camera
    T_camera_world = np.linalg.inv(T_world_camera)
    points_homog = np.hstack([point_world, np.ones((point_world.shape[0], 1))])
    points_camera_homog = (T_camera_world @ points_homog.T).T
    points_camera = points_camera_homog[:, :3]

    if is_single_point:
        return points_camera[0]
    return points_camera

def project_3d_to_2d(
    points_camera: np.ndarray,
    camera_intrinsics: CameraIntrinsics
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D points in camera coordinates to 2D image coordinates.

    Args:
        points_camera: 3D points in camera frame, shape (3,) or (N, 3)
        camera_intrinsics: Camera intrinsic parameters

    Returns:
        Tuple of:
            - 2D points in image coordinates, shape (2,) or (N, 2)
            - Boolean mask indicating which points are valid (in front of camera)
    """
    points_camera = np.asarray(points_camera, dtype=np.float64)
    is_single_point = (points_camera.ndim == 1)

    if is_single_point:
        if points_camera.shape != (3,):
            raise ValueError(f"Point must be shape (3,), got {points_camera.shape}")
        points_camera = points_camera.reshape(1, 3)
    elif points_camera.ndim == 2:
        if points_camera.shape[1] != 3:
            raise ValueError(f"Points must be shape (N, 3), got {points_camera.shape}")
    else:
        raise ValueError(f"Points must be 1D or 2D array, got shape {points_camera.shape}")

    valid_mask = points_camera[:, 2] > 0
    points_2d = np.zeros((points_camera.shape[0], 2), dtype=np.float64)
    if np.any(valid_mask):
        valid_points = points_camera[valid_mask]
        z = valid_points[:, 2]
        points_2d[valid_mask, 0] = camera_intrinsics.fx * valid_points[:, 0] / z + camera_intrinsics.cx
        points_2d[valid_mask, 1] = camera_intrinsics.fy * valid_points[:, 1] / z + camera_intrinsics.cy

    if is_single_point:
        return points_2d[0], valid_mask[0]
    return points_2d, valid_mask

def project_world_object_to_image(
    world_object: Object3D,
    robot_pose: Pose3D,
    camera_to_robot_transform: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    camera_intrinsics: CameraIntrinsics
) -> Optional[BoundingBox]:
    """
    Project a 3D object from world coordinates to a 2D bounding box in image space.

    This is the main utility function for robotics evaluation. It transforms a 3D
    object from world coordinates through robot and camera frames, then projects
    to a 2D bounding box that can be compared against vision model detections.

    Args:
        world_object: 3D object in world coordinates
        robot_pose: Robot pose in world frame
        camera_to_robot_transform: Camera transform relative to robot, either:
            - 4x4 transformation matrix
            - Tuple of (position, orientation)
        camera_intrinsics: Camera intrinsic parameters

    Returns:
        BoundingBox in image pixel coordinates, or None if object is not visible
        (behind camera or completely outside image bounds)
    """
    corners_world = world_object.get_corners()  # Shape: (8, 3)
    corners_camera = transform_point_world_to_camera(
        corners_world,
        robot_pose,
        camera_to_robot_transform
    )

    corners_2d, valid_mask = project_3d_to_2d(corners_camera, camera_intrinsics)
    if not np.any(valid_mask):
        return None  # Object is behind camera
    valid_corners_2d = corners_2d[valid_mask]

    x_min = np.min(valid_corners_2d[:, 0])
    y_min = np.min(valid_corners_2d[:, 1])
    x_max = np.max(valid_corners_2d[:, 0])
    y_max = np.max(valid_corners_2d[:, 1])

    x_min = np.clip(x_min, 0, camera_intrinsics.image_width)
    y_min = np.clip(y_min, 0, camera_intrinsics.image_height)
    x_max = np.clip(x_max, 0, camera_intrinsics.image_width)
    y_max = np.clip(y_max, 0, camera_intrinsics.image_height)

    if x_max <= x_min or y_max <= y_min:
        return None  # Object is outside image bounds

    return BoundingBox(
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max)
    )
