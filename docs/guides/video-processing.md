# Video Processing Examples

The video processing module provides utilities for concatenating and comparing videos using FFmpeg.

## Basic Usage

### Horizontal Side-by-Side Comparison

Create a horizontal side-by-side comparison of two or more videos:

```python
from vision_utils import concatenate_videos_horizontal

# Simple horizontal comparison
concatenate_videos_horizontal(
    video_paths=["model1_output.mp4", "model2_output.mp4"],
    output_path="comparison.mp4"
)

# With custom frame rate
concatenate_videos_horizontal(
    video_paths=["video1.mp4", "video2.mp4", "video3.mp4"],
    output_path="horizontal_30fps.mp4",
    target_fps=30
)
```

### Vertical Stacking

Stack videos vertically:

```python
from vision_utils import concatenate_videos_vertical

concatenate_videos_vertical(
    video_paths=["top_video.mp4", "bottom_video.mp4"],
    output_path="vertical_stack.mp4"
)
```

### Grid Layout

Automatically arrange videos in a grid:

```python
from vision_utils import concatenate_videos_grid

# Creates a 2x2 grid for 4 videos
concatenate_videos_grid(
    video_paths=["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"],
    output_path="grid_comparison.mp4"
)
```

### Sequential Concatenation

Concatenate videos end-to-end:

```python
from vision_utils import concatenate_videos_sequential

# Play videos one after another
concatenate_videos_sequential(
    video_paths=["part1.mp4", "part2.mp4", "part3.mp4"],
    output_path="full_video.mp4"
)

# With FPS adjustment
concatenate_videos_sequential(
    video_paths=["clip1.mp4", "clip2.mp4"],
    output_path="fast_compilation.mp4",
    target_fps=60  # Speed up to 60 fps
)
```

## Advanced Usage

### Using VideoComparison Class

For more control, use the `VideoComparison` class directly:

```python
from vision_utils import VideoComparison

# Create a custom comparison
comparison = VideoComparison(
    output_path="custom_comparison.mp4",
    mode="side-by-side",
    layout="grid",
    target_fps=30
)

# Process the videos
comparison.compare([
    "detection_model1.mp4",
    "detection_model2.mp4",
    "detection_model3.mp4",
    "detection_model4.mp4"
])
```

### Inspecting Video Metadata

```python
from vision_utils import VideoComparison

comparison = VideoComparison("output.mp4")

# Probe a single video
video_info = comparison.probe_video("input_video.mp4")

print(f"Resolution: {video_info.width}x{video_info.height}")
print(f"FPS: {video_info.fps}")
print(f"Codec: {video_info.codec}")
print(f"Duration: {video_info.duration}s")
print(f"Has audio: {video_info.has_audio}")
```

## Model Comparison Workflow

Compare outputs from different object detection models:

```python
from vision_utils import (
    ObjectDetectionEvaluator,
    EvaluationConfig,
    concatenate_videos_grid
)

# Step 1: Run evaluations with video output
models = ["facebook/detr-resnet-50", "facebook/detr-resnet-101"]
output_videos = []

for model in models:
    config = EvaluationConfig({
        "model_name": model,
        "input_video": "test_video.mp4",
        "save_annotated_video": True,
        "output_dir": f"results/{model.replace('/', '_')}"
    })

    evaluator = ObjectDetectionEvaluator(config)
    results = evaluator.evaluate()
    output_videos.append(results["annotated_video_path"])

# Step 2: Create side-by-side comparison
concatenate_videos_grid(
    video_paths=output_videos,
    output_path="model_comparison.mp4",
    target_fps=30
)
```

## Command Line Usage

The module also provides a CLI script for quick video operations:

```bash
# Horizontal comparison
python scripts/concatenate_videos.py video1.mp4 video2.mp4 -o comparison.mp4

# Vertical stack
python scripts/concatenate_videos.py video1.mp4 video2.mp4 \\
    --layout vertical -o vertical.mp4

# Grid layout
python scripts/concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 \\
    --layout grid -o grid.mp4

# Sequential concatenation
python scripts/concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 \\
    --mode sequential -o sequential.mp4

# With custom FPS
python scripts/concatenate_videos.py video1.mp4 video2.mp4 \\
    --mode sequential --fps 60 -o fast.mp4

# From file list
echo "video1.mp4" > videos.txt
echo "video2.mp4" >> videos.txt
python scripts/concatenate_videos.py -i videos.txt -o output.mp4
```

## Tips

1. **Frame Rate Matching**: When comparing videos with different frame rates, specify a `target_fps` to ensure smooth playback.

2. **Resolution Handling**: Videos are automatically scaled and padded to fit in grid cells while maintaining aspect ratio.

3. **Audio**: Only the first video's audio is included in side-by-side comparisons. Sequential concatenation preserves audio from all videos.

4. **Output Location**: The CLI script automatically saves outputs to the `results/` directory.

5. **Performance**: Grid layouts and sequential concatenation can be slow for long videos. Use `--verbose` for progress tracking.
