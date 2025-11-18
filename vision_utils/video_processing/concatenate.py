"""
Video concatenation and comparison utilities.

This module provides functionality for creating side-by-side video comparisons
and sequential video concatenations using FFmpeg.
"""
import os
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import math
import ffmpeg

logger = logging.getLogger(__name__)

class VideoInfo:
    """Container for video metadata."""
    def __init__(self, filepath: str, probe_data: Dict):
        self.filepath = filepath
        self.probe_data = probe_data

        video_streams = [s for s in probe_data['streams'] if s['codec_type'] == 'video']
        if not video_streams:
            raise ValueError(f"No video stream found in {filepath}")

        self.video_stream = video_streams[0]
        self.width = int(self.video_stream['width'])
        self.height = int(self.video_stream['height'])
        self.codec = self.video_stream['codec_name']

        fps_str = self.video_stream.get('r_frame_rate', '30/1')
        num, denom = map(int, fps_str.split('/'))
        self.fps = num / denom if denom != 0 else 30.0

        audio_streams = [s for s in probe_data['streams'] if s['codec_type'] == 'audio']
        self.has_audio = len(audio_streams) > 0
        self.audio_codec = audio_streams[0]['codec_name'] if audio_streams else None

        self.duration = float(probe_data['format'].get('duration', 0))

    def __repr__(self) -> str:
        return (f"VideoInfo(resolution={self.width}x{self.height}, "
                f"fps={self.fps:.2f}, codec={self.codec}, "
                f"audio={self.audio_codec or 'none'})")

class VideoComparison:
    """Handles side-by-side video comparison and sequential concatenation with automatic property normalization."""

    def __init__(self, output_path: str, mode: str = 'side-by-side', layout: str = 'horizontal', target_fps: Optional[float] = None):
        """
        Initialize the video comparison.

        Args:
            output_path: Path to output video file
            mode: Comparison mode - 'side-by-side' or 'sequential'
            layout: Layout type - 'horizontal' for side-by-side, 'grid' for auto grid, or 'vertical' for stacked
            target_fps: Target frame rate (overrides automatic detection from first video)
        """
        self.output_path = output_path
        self.mode = mode
        self.layout = layout
        self.target_fps = target_fps
        self.video_infos: List[VideoInfo] = []

    def probe_video(self, filepath: str) -> VideoInfo:
        """
        Probe video file to extract metadata.

        Args:
            filepath: Path to video file

        Returns:
            VideoInfo object with metadata

        Raises:
            RuntimeError: If ffprobe fails
        """
        try:
            probe = ffmpeg.probe(filepath)
            return VideoInfo(filepath, probe)
        except ffmpeg.Error as e:
            logger.error(f"Failed to probe {filepath}: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to probe video: {filepath}")

    def validate_inputs(self, video_paths: List[str]) -> List[VideoInfo]:
        """
        Validate all input videos and extract metadata.

        Args:
            video_paths: List of video file paths

        Returns:
            List of VideoInfo objects

        Raises:
            FileNotFoundError: If any video file doesn't exist
        """
        logger.info(f"Validating {len(video_paths)} input videos...")

        video_infos = []
        for path in video_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Video file not found: {path}")

            logger.info(f"Probing: {path}")
            info = self.probe_video(path)
            logger.info(f"  {info}")
            video_infos.append(info)

        return video_infos

    def _calculate_grid_layout(self, num_videos: int) -> Tuple[int, int]:
        """
        Calculate optimal grid layout (rows, cols) for number of videos.

        Args:
            num_videos: Number of videos to arrange

        Returns:
            Tuple of (rows, cols)
        """
        if num_videos <= 2:
            return (1, num_videos)
        elif num_videos <= 4:
            return (2, 2)
        elif num_videos <= 6:
            return (2, 3)
        elif num_videos <= 9:
            return (3, 3)
        else:
            cols = math.ceil(math.sqrt(num_videos))
            rows = math.ceil(num_videos / cols)
            return (rows, cols)

    def create_sequential_concatenation(self, video_infos: List[VideoInfo]) -> None:
        """
        Create sequential video concatenation (videos played one after another).
        Uses simple file list concatenation with optional FPS adjustment.

        Args:
            video_infos: List of VideoInfo objects
        """
        if not video_infos:
            raise ValueError("No videos to concatenate")

        num_videos = len(video_infos)
        logger.info(f"Creating sequential concatenation with {num_videos} videos...")

        ref = video_infos[0]
        target_fps = self.target_fps if self.target_fps is not None else ref.fps
        logger.info(f"Original FPS: {ref.fps:.2f}, Target FPS: {target_fps:.2f}")

        concat_list_fd, concat_list_path = tempfile.mkstemp(suffix='.txt', text=True)

        try:
            with os.fdopen(concat_list_fd, 'w') as f:
                for info in video_infos:
                    escaped_path = info.filepath.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            logger.info(f"Created file list: {concat_list_path}")

            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
            ]

            if self.target_fps is not None:
                speed_factor = target_fps / ref.fps
                pts_factor = 1.0 / speed_factor

                logger.info(f"Changing playback speed by {speed_factor:.2f}x and setting FPS to {target_fps:.2f}...")
                cmd.extend([
                    '-filter:v', f'setpts={pts_factor}*PTS',
                    '-r', str(target_fps),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23'
                ])
            else:
                logger.info("No FPS adjustment, using stream copy (fast)...")
                cmd.extend(['-c', 'copy'])

            cmd.extend(['-y', self.output_path])

            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                logger.info(f"Successfully created: {self.output_path}")

            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr}")
                raise RuntimeError("Sequential concatenation failed")

        finally:
            if os.path.exists(concat_list_path):
                os.unlink(concat_list_path)

    def create_side_by_side_comparison(self, video_infos: List[VideoInfo]) -> None:
        """
        Create side-by-side video comparison.

        Args:
            video_infos: List of VideoInfo objects
        """
        if not video_infos:
            raise ValueError("No videos to compare")

        num_videos = len(video_infos)
        logger.info(f"Creating {self.layout} comparison with {num_videos} videos...")

        if self.layout == 'horizontal':
            rows, cols = 1, num_videos
        elif self.layout == 'vertical':
            rows, cols = num_videos, 1
        elif self.layout == 'grid':
            rows, cols = self._calculate_grid_layout(num_videos)
        else:
            raise ValueError(f"Unknown layout: {self.layout}")

        logger.info(f"Layout: {rows} row(s) x {cols} column(s)")

        ref = video_infos[0]
        target_fps = self.target_fps if self.target_fps is not None else ref.fps
        logger.info(f"Target FPS: {target_fps:.2f}")

        cell_width = 1920 // cols 
        cell_height = 1080 // rows

        logger.info(f"Cell size: {cell_width}x{cell_height}")
        logger.info(f"Output size: {cell_width * cols}x{cell_height * rows}")

        scaled_videos = []
        audio_inputs = []

        for i, info in enumerate(video_infos):
            input_stream = ffmpeg.input(info.filepath)

            video = input_stream.video
            video = video.filter('scale',
                                 cell_width, cell_height,
                                 force_original_aspect_ratio='decrease')
            video = video.filter('pad',
                                 cell_width, cell_height,
                                 '(ow-iw)/2', '(oh-ih)/2',
                                 color='black')

            if abs(info.fps - target_fps) > 0.1:
                video = video.filter('fps', fps=target_fps)

            video = video.filter('format', 'yuv420p')
            scaled_videos.append(video)

            if i == 0 and info.has_audio:
                audio = input_stream.audio
                audio = audio.filter('aresample', 48000)
                audio_inputs.append(audio)

        if rows == 1:
            stacked = ffmpeg.filter(scaled_videos, 'hstack', inputs=num_videos)
        elif cols == 1:
            stacked = ffmpeg.filter(scaled_videos, 'vstack', inputs=num_videos)
        else:
            row_stacks = []
            for row in range(rows):
                start_idx = row * cols
                end_idx = min(start_idx + cols, num_videos)
                row_videos = scaled_videos[start_idx:end_idx]

                while len(row_videos) < cols:
                    black = ffmpeg.input(
                        f'color=c=black:s={cell_width}x{cell_height}:d={ref.duration}',
                        f='lavfi'
                    )
                    row_videos.append(black)

                if len(row_videos) == 1:
                    row_stack = row_videos[0]
                else:
                    row_stack = ffmpeg.filter(row_videos, 'hstack', inputs=len(row_videos))
                row_stacks.append(row_stack)

            if len(row_stacks) == 1:
                stacked = row_stacks[0]
            else:
                stacked = ffmpeg.filter(row_stacks, 'vstack', inputs=len(row_stacks))

        try:
            if audio_inputs:
                output = ffmpeg.output(
                    stacked, audio_inputs[0],
                    self.output_path,
                    vcodec='libx264',
                    preset='medium',
                    crf=23,
                    acodec='aac',
                    audio_bitrate='128k',
                    **{'movflags': '+faststart'}
                )
            else:
                logger.warning("No audio found in first video - creating video without audio")
                output = ffmpeg.output(
                    stacked,
                    self.output_path,
                    vcodec='libx264',
                    preset='medium',
                    crf=23,
                    **{'movflags': '+faststart'}
                )

            output = output.overwrite_output()

            logger.info("Running FFmpeg (this may take a while)...")
            output.run(capture_stdout=True, capture_stderr=True)
            logger.info(f"Successfully created: {self.output_path}")

        except Exception as e:
            logger.error(f"FFmpeg error: {str(e)}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(e.stderr.decode() if isinstance(e.stderr, bytes) else str(e.stderr))
            raise RuntimeError("Video comparison failed")

    def compare(self, video_paths: List[str]) -> None:
        """
        Process videos based on selected mode (side-by-side comparison or sequential concatenation).

        Args:
            video_paths: List of video file paths
        """
        video_infos = self.validate_inputs(video_paths)

        if not video_infos:
            raise ValueError("No valid videos to process")

        if self.mode == 'sequential':
            self.create_sequential_concatenation(video_infos)
        elif self.mode == 'side-by-side':
            self.create_side_by_side_comparison(video_infos)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if os.path.exists(self.output_path):
            output_size = os.path.getsize(self.output_path) / (1024 * 1024)  # MB
            logger.info(f"Output file size: {output_size:.2f} MB")
        else:
            raise RuntimeError("Output file was not created")

def concatenate_videos_horizontal(
    video_paths: List[str],
    output_path: str,
    target_fps: Optional[float] = None
) -> None:
    """
    Create a horizontal side-by-side video comparison.

    Args:
        video_paths: List of video file paths to concatenate horizontally
        output_path: Path to output video file
        target_fps: Optional target frame rate (default: use first video's FPS)

    Example:
        >>> from vision_utils import concatenate_videos_horizontal
        >>> concatenate_videos_horizontal(
        ...     ["video1.mp4", "video2.mp4"],
        ...     "output_horizontal.mp4"
        ... )
    """
    comparison = VideoComparison(
        output_path=output_path,
        mode='side-by-side',
        layout='horizontal',
        target_fps=target_fps
    )
    comparison.compare(video_paths)

def concatenate_videos_vertical(
    video_paths: List[str],
    output_path: str,
    target_fps: Optional[float] = None
) -> None:
    """
    Create a vertical stacked video comparison.

    Args:
        video_paths: List of video file paths to stack vertically
        output_path: Path to output video file
        target_fps: Optional target frame rate (default: use first video's FPS)

    Example:
        >>> from vision_utils import concatenate_videos_vertical
        >>> concatenate_videos_vertical(
        ...     ["video1.mp4", "video2.mp4"],
        ...     "output_vertical.mp4"
        ... )
    """
    comparison = VideoComparison(
        output_path=output_path,
        mode='side-by-side',
        layout='vertical',
        target_fps=target_fps
    )
    comparison.compare(video_paths)

def concatenate_videos_grid(
    video_paths: List[str],
    output_path: str,
    target_fps: Optional[float] = None
) -> None:
    """
    Create a grid layout video comparison (automatically calculates optimal grid).

    Args:
        video_paths: List of video file paths to arrange in a grid
        output_path: Path to output video file
        target_fps: Optional target frame rate (default: use first video's FPS)

    Example:
        >>> from vision_utils import concatenate_videos_grid
        >>> concatenate_videos_grid(
        ...     ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"],
        ...     "output_grid.mp4"
        ... )
    """
    comparison = VideoComparison(
        output_path=output_path,
        mode='side-by-side',
        layout='grid',
        target_fps=target_fps
    )
    comparison.compare(video_paths)

def concatenate_videos_sequential(
    video_paths: List[str],
    output_path: str,
    target_fps: Optional[float] = None
) -> None:
    """
    Create a sequential video concatenation (videos played one after another).

    Args:
        video_paths: List of video file paths to concatenate sequentially
        output_path: Path to output video file
        target_fps: Optional target frame rate (default: use first video's FPS)

    Example:
        >>> from vision_utils import concatenate_videos_sequential
        >>> concatenate_videos_sequential(
        ...     ["video1.mp4", "video2.mp4", "video3.mp4"],
        ...     "output_sequential.mp4",
        ...     target_fps=30
        ... )
    """
    comparison = VideoComparison(
        output_path=output_path,
        mode='sequential',
        layout='horizontal',  # layout is ignored in sequential mode
        target_fps=target_fps
    )
    comparison.compare(video_paths)
