#!/usr/bin/env python3
"""
Vision Model Evaluation Script

This script demonstrates how to use the vision_utils library for evaluating
vision models on video and image datasets. Supports:
- Object detection models (YOLO, DETR, RT-DETR, Grounding DINO, OWL-ViT, etc.)
- Image classification models (DINOtxt, etc.)
- Semantic segmentation models (DINOtxt, etc.)

Usage:
    # Run evaluation with a config file (auto-detects task type)
    python scripts/evaluate.py --config configs/rtdetr_r50vd_evaluation.yaml

    # Run evaluation with explicit task type
    python scripts/evaluate.py --config configs/dinotxt_classification.yaml --task classification

    # Generate comparison plots from existing results (requires 2+ models)
    python scripts/evaluate.py --visualize results/model1.json results/model2.json --output plots/
"""
import argparse
import sys
import os
from pathlib import Path

from vision_utils import (
    EvaluationConfig,
    ObjectDetectionEvaluator,
    ClassificationEvaluator,
    SegmentationEvaluator,
    ResultsPlotter,
    create_model
)

def detect_task_type(config: EvaluationConfig) -> str:
    """
    Auto-detect task type from model name.

    Args:
        config: EvaluationConfig object

    Returns:
        Task type: 'detection', 'classification', or 'segmentation'
    """
    model_name_lower = config.model_name.lower()

    if 'dinotxt' in model_name_lower or 'dinov3' in model_name_lower:
        if hasattr(config, 'task_type') and config.task_type:
            return config.task_type
        return 'classification'

    return 'detection'

def run_evaluation(config_path: str, plot: bool = False, task_type: str = None):
    """
    Run vision model evaluation.

    Args:
        config_path: Path to configuration YAML file
        plot: Whether to generate plots after evaluation (requires 2+ models)
        task_type: Explicit task type ('detection', 'classification', 'segmentation').
                  If None, auto-detects from model name.
    """
    print(f"Loading configuration from: {config_path}")
    config = EvaluationConfig(config_path)

    if task_type is None:
        task_type = detect_task_type(config)

    print(f"Starting {task_type.capitalize()} Evaluation")
    print(f"Model: {config.model_name}")
    print(f"Input: {config.input_path}")

    if task_type == 'detection':
        evaluator = ObjectDetectionEvaluator(config)
    elif task_type == 'classification':
        evaluator = ClassificationEvaluator(config)
    elif task_type == 'segmentation':
        evaluator = SegmentationEvaluator(config)
    else:
        raise ValueError(f"Unknown task type: {task_type}. Must be 'detection', 'classification', or 'segmentation'.")

    evaluator.evaluate()

    print("Evaluation Complete!")
    print(f"Results saved to: {config.results_path}")

    if config.generate_video and config.input_type == 'video':
        video_filename = f"{os.path.splitext(os.path.basename(config.input_path))[0]}_annotated.mp4"
        video_path = os.path.join(config.output_directory, video_filename)
        if os.path.exists(video_path):
            print(f"Annotated video: {video_path}")

    if plot:
        print("\nNote: The --plot flag is for multi-model comparisons.")
        print("To visualize results, run evaluation for at least 2 models, then use:")
        print(f"  python scripts/evaluate.py --visualize <results1.json> <results2.json>")

def visualize_results(result_paths: list, output_dir: str = "results/plots"):
    """
    Generate visualization plots from evaluation results.
    Requires at least 2 models for comparison.

    Args:
        result_paths: List of paths to result JSON files (minimum 2)
        output_dir: Output directory for plots
    """
    if len(result_paths) < 2:
        raise ValueError(
            f"Visualization requires at least 2 models for comparison.\n"
            f"Provided: {len(result_paths)} result file(s)\n"
            f"Please provide 2 or more result JSON files."
        )

    if not output_dir.startswith("results/"):
        output_dir = os.path.join("results", output_dir)

    print(f"\nLoading results from {len(result_paths)} models for comparison...")

    for path in result_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result file not found: {path}")

    plotter = ResultsPlotter()

    print("Generating multi-model comparison plots...")
    plotter.compare_models(result_paths, output_dir=output_dir)

    print(f"\nComparison plots saved to: {output_dir}/")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vision Model Evaluation Script (Detection, Classification, Segmentation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
                # Run detection evaluation (auto-detected)
                python scripts/evaluate.py --config configs/rtdetr_r101vd_evaluation.yaml

                # Run classification evaluation (auto-detected for DINOtxt)
                python scripts/evaluate.py --config configs/dinotxt_classification.yaml

                # Run segmentation evaluation (explicit task type)
                python scripts/evaluate.py --config configs/dinotxt_segmentation.yaml --task segmentation

                # Compare 2 models side-by-side
                python scripts/evaluate.py --visualize \\
                    results/rtdetr_r50/results.json \\
                    results/rtdetr_r101/results.json \\
                    --output comparison_plots/

                # Compare 3+ models
                python scripts/evaluate.py --visualize \\
                    results/rtdetr/results.json \\
                    results/yolov8/results.json \\
                    results/dinotxt/results.json \\
                    --output multi_model_comparison/

            Note:
                - Task type is auto-detected from model name
                - DINOtxt defaults to classification; use --task segmentation for segmentation
                - Visualization requires at least 2 models for comparison
                - All result files must exist before running --visualize
                - Outputs are automatically saved under the results/ directory
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file for evaluation'
    )
    group.add_argument(
        '--visualize',
        nargs='+',
        metavar='RESULT_FILE',
        help='Path(s) to result JSON file(s) for comparison (minimum 2 required)'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['detection', 'classification', 'segmentation'],
        help='Explicit task type (default: auto-detect from model name)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Show info about generating comparison plots (use --visualize for actual plotting)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/plots',
        help='Output directory for plots (default: results/plots/)'
    )

    args = parser.parse_args()

    try:
        if args.config:
            if not os.path.exists(args.config):
                print(f"Error: Configuration file not found: {args.config}", file=sys.stderr)
                sys.exit(1)

            run_evaluation(args.config, plot=args.plot, task_type=args.task)

        elif args.visualize:
            visualize_results(args.visualize, output_dir=args.output)

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
