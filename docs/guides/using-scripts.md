# Using Scripts

The `scripts/` directory contains ready-to-use scripts that demonstrate how to use the vision_utils library. These scripts serve as both learning references and practical command-line tools.

## Available Scripts

### 1. Object Detection Evaluation (`evaluate.py`)

A comprehensive example showing how to evaluate object detection models on videos or image datasets.

**Location**: [scripts/evaluate.py](../../scripts/evaluate.py)

#### Features

- Run evaluation on single models
- Generate multi-model comparison plots
- Support for YAML configuration files
- Automatic results saving

#### Usage

**Run a single model evaluation:**

```bash
python scripts/evaluate.py --config configs/rtdetr_r50vd_evaluation.yaml
```

**Compare multiple models:**

```bash
# First, run evaluations for each model
python scripts/evaluate.py --config configs/model1.yaml
python scripts/evaluate.py --config configs/model2.yaml

# Then generate comparison plots
python scripts/evaluate.py --visualize \\
    results/model1/results.json \\
    results/model2/results.json \\
    --output comparison_plots/
```

**Compare 3+ models:**

```bash
python scripts/evaluate.py --visualize \\
    results/rtdetr_r50/results.json \\
    results/rtdetr_r101/results.json \\
    results/yolov8/results.json \\
    --output multi_model_comparison/
```

#### Configuration File Format

The script expects a YAML configuration file with the following structure:

```yaml
# Model configuration
model_name: "facebook/detr-resnet-50"  # HuggingFace model or YOLO variant

# Input data (choose one)
input_video: "path/to/video.mp4"
# OR
input_images: "path/to/image/directory"

# Optional: Ground truth annotations (COCO format)
ground_truth_path: "path/to/annotations.json"

# Output settings
output_dir: "results/rtdetr_r50"
save_annotated_video: true

# Model-specific parameters (optional)
confidence_threshold: 0.5
max_detections: 100
```

See [configs/examples/](../../configs/examples/) for complete examples.

#### Code Reference

The example demonstrates:

- Using `EvaluationConfig` to load configurations
- Creating an `ObjectDetectionEvaluator` instance
- Running evaluations with `evaluator.evaluate()`
- Generating comparison plots with `ResultsPlotter`

**Key code snippet:**

```python
from vision_utils import EvaluationConfig, ObjectDetectionEvaluator

# Load configuration
config = EvaluationConfig("configs/model.yaml")

# Create evaluator
evaluator = ObjectDetectionEvaluator(config)

# Run evaluation
results = evaluator.evaluate()
```

---

### 2. Video Concatenation (`concatenate_videos.py`)

Demonstrates how to use the video processing utilities for creating side-by-side comparisons and sequential concatenations.

**Location**: [scripts/concatenate_videos.py](../../scripts/concatenate_videos.py)

#### Features

- Horizontal side-by-side comparison
- Vertical stacking
- Automatic grid layout
- Sequential concatenation
- Frame rate control
- Audio preservation

#### Usage

**Horizontal side-by-side comparison:**

```bash
python scripts/concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 \\
    -o comparison.mp4
```

**Vertical stacking:**

```bash
python scripts/concatenate_videos.py video1.mp4 video2.mp4 \\
    --layout vertical -o vertical_stack.mp4
```

**Grid layout (automatic):**

```bash
python scripts/concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 \\
    --layout grid -o grid_comparison.mp4
```

**Sequential concatenation:**

```bash
python scripts/concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 \\
    --mode sequential -o full_video.mp4
```

**With custom frame rate:**

```bash
python scripts/concatenate_videos.py video1.mp4 video2.mp4 \\
    --mode sequential --fps 60 -o fast_compilation.mp4
```

**From a file list:**

```bash
# Create a list of videos
echo "video1.mp4" > videos.txt
echo "video2.mp4" >> videos.txt
echo "video3.mp4" >> videos.txt

# Process the list
python scripts/concatenate_videos.py -i videos.txt \\
    --mode sequential -o output.mp4
```

#### Command-Line Options

| Option | Description |
|--------|-------------|
| `videos` | Video files to process (positional arguments) |
| `-i, --input-list` | Text file with video paths (one per line) |
| `-o, --output` | Output video filename (saved in `results/`) |
| `--mode` | Processing mode: `side-by-side` or `sequential` |
| `--layout` | Layout for side-by-side: `horizontal`, `vertical`, or `grid` |
| `--fps` | Target frame rate (overrides auto-detection) |
| `-v, --verbose` | Enable verbose logging |

#### Code Reference

The example shows how to use the video processing API programmatically:

```python
from vision_utils import concatenate_videos_horizontal

# Simple horizontal comparison
concatenate_videos_horizontal(
    video_paths=["model1_output.mp4", "model2_output.mp4"],
    output_path="comparison.mp4"
)
```

See the [Video Processing Examples](video-processing.md) for more programmatic usage.

---

## Common Workflows

### Workflow 1: Model Comparison Pipeline

Complete pipeline for comparing multiple object detection models:

```bash
# Step 1: Prepare configurations
cp configs/examples/rtdetr_r50vd_evaluation.yaml configs/my_rtdetr_r50.yaml
cp configs/examples/rtdetr_r101vd_evaluation.yaml configs/my_rtdetr_r101.yaml

# Edit configs to point to your data
# ...

# Step 2: Run evaluations
python scripts/evaluate.py --config configs/my_rtdetr_r50.yaml
python scripts/evaluate.py --config configs/my_rtdetr_r101.yaml

# Step 3: Generate comparison plots
python scripts/evaluate.py --visualize \\
    results/rtdetr_r50/results.json \\
    results/rtdetr_r101/results.json \\
    --output model_comparison/

# Step 4: Create video comparison
python scripts/concatenate_videos.py \\
    results/rtdetr_r50/annotated_output.mp4 \\
    results/rtdetr_r101/annotated_output.mp4 \\
    --layout horizontal \\
    -o rtdetr_comparison.mp4
```

### Workflow 2: Batch Video Processing

Process multiple videos with the same model:

```bash
# Create a script to automate batch processing
for video in videos/*.mp4; do
    python scripts/evaluate.py \\
        --config configs/base_config.yaml \\
        --input-video "$video"
done

# Combine all outputs
ls results/*/annotated_output.mp4 > videos.txt
python scripts/concatenate_videos.py -i videos.txt \\
    --mode sequential -o full_batch_output.mp4
```

---

## Using as Python Modules

While these scripts are designed for command-line use, you can also import and use them in your own Python code:

```python
from scripts.evaluate import run_evaluation, visualize_results

# Run evaluation programmatically
run_evaluation("configs/model.yaml", plot=False)

# Generate visualizations
visualize_results([
    "results/model1/results.json",
    "results/model2/results.json"
], output_dir="results/plots")
```

However, it's recommended to use the vision_utils library directly for programmatic use:

```python
from vision_utils import (
    EvaluationConfig,
    ObjectDetectionEvaluator,
    ResultsPlotter,
    concatenate_videos_horizontal
)

# Your custom workflow here...
```

---

## Tips and Best Practices

1. **Configuration Management**: Keep your config files in `configs/` and use descriptive names
2. **Result Organization**: Use the `output_dir` setting to organize results by model/experiment
3. **Video Output**: Always check that `save_annotated_video: true` is set if you want video output
4. **Comparison Requirements**: Visualization requires at least 2 models for meaningful comparisons
5. **Performance**: Grid layouts can be slow for long videos - consider trimming videos first
6. **Audio**: Only the first video's audio is preserved in side-by-side comparisons

---

## Troubleshooting

### Common Issues

**"Configuration file not found"**
- Ensure the path to your config file is correct
- Use relative paths from the project root

**"Result file not found"**
- Make sure you've run the evaluation before trying to visualize
- Check that the paths in your visualize command match the actual result file locations

**"Visualization requires at least 2 models"**
- You need results from at least 2 different model evaluations to create comparison plots
- Run evaluation on multiple models or configs first

**FFmpeg errors in video concatenation**
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check that all input videos exist and are valid MP4 files
- Try running with `-v` flag for verbose output

---

## Next Steps

- Check out [Basic Usage Examples](basic-usage.md) for library usage
- Explore [Video Processing Examples](video-processing.md) for programmatic video operations
