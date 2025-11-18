"""
Execution timing utilities for tracking evaluation performance.
"""
import time
from typing import Dict

class ExecutionTimer:
    """
    Track execution time for different stages of evaluation.
    """
    def __init__(self):
        """Initialize execution timer."""
        self.timings = {}
        self._start_times = {}

    def start(self, stage: str):
        """
        Start timing a stage.

        Args:
            stage: Name of the stage
        """
        self._start_times[stage] = time.time()

    def stop(self, stage: str):
        """
        Stop timing a stage and record the duration.

        Args:
            stage: Name of the stage
        """
        if stage in self._start_times:
            duration = time.time() - self._start_times[stage]
            self.timings[stage] = duration
            del self._start_times[stage]

    def get_timing(self, stage: str) -> float:
        """
        Get timing for a specific stage.

        Args:
            stage: Name of the stage

        Returns:
            Duration in seconds
        """
        return self.timings.get(stage, 0.0)

    def get_total_time(self) -> float:
        """
        Get total execution time across all stages.

        Returns:
            Total time in seconds
        """
        return sum(self.timings.values())

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all timings.

        Returns:
            Dictionary with timing information
        """
        summary = dict(self.timings)
        summary['total_time'] = self.get_total_time()
        return summary

    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self._start_times.clear()

    def __repr__(self) -> str:
        return f"ExecutionTimer(total_time={self.get_total_time():.2f}s)"
