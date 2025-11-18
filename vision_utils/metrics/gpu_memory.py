"""
GPU memory tracking utilities.
"""
import torch
from typing import Dict, List, Any, Optional

class GPUMemoryTracker:
    """
    Track GPU memory usage during model inference.
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initialize GPU memory tracker.

        Args:
            device: Device string ('cuda', 'cpu', etc.)
        """
        self.device = device
        self.cuda_available = torch.cuda.is_available() and (device is None or 'cuda' in device)
        self.memory_snapshots = []  # List of (allocated_mb, reserved_mb) tuples

        self.device_name = None
        self.total_memory_mb = None
        if self.cuda_available:
            try:
                self.device_name = torch.cuda.get_device_name(0)
                self.total_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            except:
                pass

    def record_snapshot(self):
        """
        Record current GPU memory usage snapshot.
        If CUDA is not available, records zeros.
        """
        if self.cuda_available:
            try:
                allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                self.memory_snapshots.append((allocated_mb, reserved_mb))
            except:
                self.memory_snapshots.append((0.0, 0.0))
        else:
            self.memory_snapshots.append((0.0, 0.0))

    def get_peak_memory_mb(self) -> float:
        """
        Get peak allocated memory in MB.

        Returns:
            Peak memory usage in MB
        """
        if not self.memory_snapshots:
            return 0.0
        return max(snapshot[0] for snapshot in self.memory_snapshots)

    def get_average_memory_mb(self) -> float:
        """
        Get average allocated memory in MB.

        Returns:
            Average memory usage in MB
        """
        if not self.memory_snapshots:
            return 0.0
        total = sum(snapshot[0] for snapshot in self.memory_snapshots)
        return total / len(self.memory_snapshots)

    def get_min_memory_mb(self) -> float:
        """
        Get minimum allocated memory in MB.

        Returns:
            Minimum memory usage in MB
        """
        if not self.memory_snapshots:
            return 0.0
        return min(snapshot[0] for snapshot in self.memory_snapshots)

    def get_per_frame_memory(self) -> List[float]:
        """
        Get list of allocated memory per frame in MB.

        Returns:
            List of memory values
        """
        return [snapshot[0] for snapshot in self.memory_snapshots]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get complete GPU memory summary.

        Returns:
            Dictionary with memory statistics
        """
        if not self.cuda_available or not self.memory_snapshots:
            return {
                'device_name': self.device_name or 'CPU',
                'cuda_available': False,
                'peak_memory_mb': 0.0,
                'average_memory_mb': 0.0,
                'min_memory_mb': 0.0
            }

        return {
            'device_name': self.device_name or 'Unknown GPU',
            'total_memory_mb': self.total_memory_mb,
            'cuda_available': True,
            'peak_memory_mb': self.get_peak_memory_mb(),
            'average_memory_mb': self.get_average_memory_mb(),
            'min_memory_mb': self.get_min_memory_mb(),
            'num_snapshots': len(self.memory_snapshots)
        }

    def reset(self):
        """Reset all memory tracking."""
        self.memory_snapshots.clear()

    def __repr__(self) -> str:
        return (f"GPUMemoryTracker(device={self.device_name}, "
                f"peak={self.get_peak_memory_mb():.1f}MB, "
                f"avg={self.get_average_memory_mb():.1f}MB)")
