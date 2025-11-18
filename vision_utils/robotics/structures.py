"""
Data structures for robotics coordinate transformations.
"""
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np
from .rotations import quaternion_to_rotation_matrix

@dataclass
class Pose3D:
    """
    Represents a 3D pose (position + orientation) in space.

    Attributes:
        position: 3D position as [x, y, z] numpy array
        orientation: Orientation as quaternion [w, x, y, z] or 3x3 rotation matrix
        frame_id: Optional identifier for the reference frame
    """
    position: np.ndarray  # Shape: (3,)
    orientation: np.ndarray  # Shape: (4,) for quaternion or (3,3) for rotation matrix
    frame_id: Optional[int] = None

    def __post_init__(self):
        """Validate inputs."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.orientation = np.asarray(self.orientation, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")

        if self.orientation.shape not in [(4,), (3, 3)]:
            raise ValueError(
                f"Orientation must be quaternion (4,) or rotation matrix (3,3), "
                f"got {self.orientation.shape}"
            )

    @property
    def is_quaternion(self) -> bool:
        """Check if orientation is stored as quaternion."""
        return self.orientation.shape == (4,)

    @property
    def is_rotation_matrix(self) -> bool:
        """Check if orientation is stored as rotation matrix."""
        return self.orientation.shape == (3, 3)

@dataclass
class Object3D:
    """
    Represents a 3D object in world space.

    Attributes:
        position: 3D position (center) as [x, y, z] numpy array
        dimensions: Object dimensions as [length, width, height] numpy array
        class_name: Object class/category name
        object_id: Optional unique identifier for the object
        orientation: Optional orientation (defaults to axis-aligned)
    """
    position: np.ndarray  # Shape: (3,)
    dimensions: np.ndarray  # Shape: (3,) - [length, width, height]
    class_name: str
    object_id: Optional[int] = None
    orientation: Optional[np.ndarray] = None  # Quaternion or rotation matrix

    def __post_init__(self):
        """Validate inputs."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.dimensions = np.asarray(self.dimensions, dtype=np.float64)

        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")

        if self.dimensions.shape != (3,):
            raise ValueError(f"Dimensions must be shape (3,), got {self.dimensions.shape}")

        if self.orientation is not None:
            self.orientation = np.asarray(self.orientation, dtype=np.float64)
            if self.orientation.shape not in [(4,), (3, 3)]:
                raise ValueError(
                    f"Orientation must be quaternion (4,) or rotation matrix (3,3), "
                    f"got {self.orientation.shape}"
                )

    def get_corners(self) -> np.ndarray:
        """
        Get the 8 corners of the 3D bounding box.

        Returns:
            np.ndarray: Shape (8, 3) array of corner positions
        """
        # Half dimensions
        l, w, h = self.dimensions / 2

        # 8 corners of axis-aligned box (before rotation)
        corners = np.array([
            [-l, -w, -h],  # 0: back-left-bottom
            [+l, -w, -h],  # 1: back-right-bottom
            [+l, +w, -h],  # 2: front-right-bottom
            [-l, +w, -h],  # 3: front-left-bottom
            [-l, -w, +h],  # 4: back-left-top
            [+l, -w, +h],  # 5: back-right-top
            [+l, +w, +h],  # 6: front-right-top
            [-l, +w, +h],  # 7: front-left-top
        ])

        if self.orientation is not None:
            if self.orientation.shape == (4,):
                R = quaternion_to_rotation_matrix(self.orientation)
            else:
                R = self.orientation

            corners = corners @ R.T

        corners += self.position

        return corners

@dataclass
class CameraIntrinsics:
    """
    Camera intrinsic parameters for 3D-to-2D projection.

    Attributes:
        fx: Focal length in x direction (pixels)
        fy: Focal length in y direction (pixels)
        cx: Principal point x coordinate (pixels)
        cy: Principal point y coordinate (pixels)
        image_width: Image width in pixels
        image_height: Image height in pixels
        distortion_coeffs: Optional distortion coefficients [k1, k2, p1, p2, k3, ...]
    """
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    distortion_coeffs: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate inputs."""
        if self.distortion_coeffs is not None:
            self.distortion_coeffs = np.asarray(self.distortion_coeffs, dtype=np.float64)

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        image_width: int,
        image_height: int,
        distortion_coeffs: Optional[np.ndarray] = None
    ) -> "CameraIntrinsics":
        """
        Create CameraIntrinsics from 3x3 intrinsic matrix K.

        Args:
            K: 3x3 intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            image_width: Image width in pixels
            image_height: Image height in pixels
            distortion_coeffs: Optional distortion coefficients

        Returns:
            CameraIntrinsics instance
        """
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must be 3x3, got {K.shape}")

        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            image_width=image_width,
            image_height=image_height,
            distortion_coeffs=distortion_coeffs
        )

    def to_matrix(self) -> np.ndarray:
        """
        Convert to 3x3 intrinsic matrix K.

        Returns:
            3x3 numpy array [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
