"""
Rotation conversion utilities.

Helper functions for converting between different rotation representations.
"""
import numpy as np

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to 3x3 rotation matrix.

    Args:
        q: Quaternion as [w, x, y, z] numpy array

    Returns:
        3x3 rotation matrix

    Note:
        Uses the convention q = [w, x, y, z] where w is the scalar part.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Quaternion must be shape (4,), got {q.shape}")

    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x**2 + z**2),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)

    return R

def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion.

    Args:
        R: 3x3 rotation matrix

    Returns:
        Quaternion as [w, x, y, z] numpy array
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {R.shape}")

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z], dtype=np.float64)
