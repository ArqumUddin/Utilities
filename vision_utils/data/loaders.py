"""
Data loader for video and image datasets.
"""
import os
import cv2
from pathlib import Path
from typing import Iterator, Tuple, Optional, List
import numpy as np
from PIL import Image

class VideoLoader:
    """
    Load and process video files frame by frame.
    """
    def __init__(self, video_path: str, frame_sampling_rate: int = 1):
        """
        Initialize video loader.

        Args:
            video_path: Path to video file
            frame_sampling_rate: Process every Nth frame (1 = all frames)
        """
        self.video_path = video_path
        self.frame_sampling_rate = frame_sampling_rate

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.processed_frames = 0

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Iterate over video frames.

        Yields:
            Tuple of (frame_id, frame_array) where frame is RGB numpy array
        """
        frame_id = 0
        actual_frame_number = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if actual_frame_number % self.frame_sampling_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_id, frame_rgb
                frame_id += 1
                self.processed_frames += 1

            actual_frame_number += 1

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get a specific frame by number.

        Args:
            frame_number: Frame number to retrieve

        Returns:
            RGB frame as numpy array, or None if frame doesn't exist
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.processed_frames = 0

    def release(self):
        """Release video capture object."""
        self.cap.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

    def __len__(self) -> int:
        """Get number of frames that will be processed."""
        return (self.total_frames + self.frame_sampling_rate - 1) // self.frame_sampling_rate

    def __repr__(self) -> str:
        return (f"VideoLoader(path={self.video_path}, "
                f"total_frames={self.total_frames}, "
                f"fps={self.fps:.2f}, "
                f"resolution={self.width}x{self.height}, "
                f"sampling_rate={self.frame_sampling_rate})")

class ImageDatasetLoader:
    """
    Load and process image datasets from a directory.
    """
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, dataset_path: str, frame_sampling_rate: int = 1):
        """
        Initialize image dataset loader.

        Args:
            dataset_path: Path to directory containing images
            frame_sampling_rate: Process every Nth image (1 = all images)
        """
        self.dataset_path = dataset_path
        self.frame_sampling_rate = frame_sampling_rate

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        if not os.path.isdir(dataset_path):
            raise ValueError(f"Path is not a directory: {dataset_path}")

        self.image_paths = self._get_image_paths()

        if len(self.image_paths) == 0:
            raise ValueError(f"No supported image files found in: {dataset_path}")

        self.processed_images = 0

    def _get_image_paths(self) -> List[str]:
        """Get all image file paths from dataset directory."""
        image_paths = []
        for root, dirs, files in os.walk(self.dataset_path):
            for file in sorted(files):  # Sort for consistent ordering
                if Path(file).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, str]]:
        """
        Iterate over images.

        Yields:
            Tuple of (image_id, image_array, image_path) where image is RGB numpy array
        """
        image_id = 0
        for idx, image_path in enumerate(self.image_paths):
            if idx % self.frame_sampling_rate == 0:
                image = self.load_image(image_path)
                if image is not None:
                    yield image_id, image, image_path
                    image_id += 1
                    self.processed_images += 1

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load a single image.

        Args:
            image_path: Path to image file

        Returns:
            RGB image as numpy array, or None if loading fails
        """
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            return None

    def get_image_by_index(self, index: int) -> Optional[Tuple[np.ndarray, str]]:
        """
        Get image by index.

        Args:
            index: Image index

        Returns:
            Tuple of (image_array, image_path) or None if index invalid
        """
        if 0 <= index < len(self.image_paths):
            image_path = self.image_paths[index]
            image = self.load_image(image_path)
            if image is not None:
                return image, image_path
        return None

    def __len__(self) -> int:
        """Get number of images that will be processed."""
        return (len(self.image_paths) + self.frame_sampling_rate - 1) // self.frame_sampling_rate

    def __repr__(self) -> str:
        return (f"ImageDatasetLoader(path={self.dataset_path}, "
                f"total_images={len(self.image_paths)}, "
                f"sampling_rate={self.frame_sampling_rate})")

class DataLoader:
    """
    Unified data loader that handles both videos and image datasets.
    """
    def __init__(self, input_path: str, input_type: str = 'auto', frame_sampling_rate: int = 1):
        """
        Initialize data loader.

        Args:
            input_path: Path to video file or image directory
            input_type: Type of input - 'video', 'images', or 'auto'
            frame_sampling_rate: Process every Nth frame/image
        """
        self.input_path = input_path
        self.frame_sampling_rate = frame_sampling_rate

        if input_type == 'auto':
            input_type = self._detect_input_type(input_path)

        self.input_type = input_type
        if self.input_type == 'video':
            self.loader = VideoLoader(input_path, frame_sampling_rate)
        elif self.input_type == 'images':
            self.loader = ImageDatasetLoader(input_path, frame_sampling_rate)
        else:
            raise ValueError(f"Invalid input type: {input_type}. Must be 'video', 'images', or 'auto'")

    @staticmethod
    def _detect_input_type(path: str) -> str:
        """
        Auto-detect input type from path.

        Args:
            path: Input path

        Returns:
            'video' or 'images'
        """
        path_obj = Path(path)

        if path_obj.is_file():
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
            if path_obj.suffix.lower() in video_extensions:
                return 'video'
            raise ValueError(f"File format not supported: {path_obj.suffix}")
        elif path_obj.is_dir():
            return 'images'
        else:
            raise ValueError(f"Path does not exist: {path}")

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray, Optional[str]]]:
        """
        Iterate over frames/images.

        Yields:
            Tuple of (frame_id, image_array, image_path)
            For videos, image_path is None
        """
        if self.input_type == 'video':
            for frame_id, frame in self.loader:
                yield frame_id, frame, None
        else:
            for image_id, image, image_path in self.loader:
                yield image_id, image, image_path

    def __len__(self) -> int:
        """Get number of frames/images that will be processed."""
        return len(self.loader)

    def __enter__(self):
        """Context manager entry."""
        if hasattr(self.loader, '__enter__'):
            self.loader.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if hasattr(self.loader, '__exit__'):
            self.loader.__exit__(exc_type, exc_val, exc_tb)

    def __repr__(self) -> str:
        return f"DataLoader(type={self.input_type}, loader={self.loader})"
