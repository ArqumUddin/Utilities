"""
Results writer for saving evaluation results to JSON format.
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class ResultsWriter:
    """
    Write evaluation results to JSON format.
    """
    @staticmethod
    def write_results(
        output_path: str,
        model_name: str,
        dataset_path: str,
        basic_metrics: Dict[str, Any],
        execution_time: float,
        advanced_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        model_size_mb: Optional[float] = None,
        gpu_memory: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None
    ) -> None:
        """
        Write evaluation results to JSON file.

        Args:
            output_path: Path to output JSON file
            model_name: Name of the model evaluated
            dataset_path: Path to input dataset
            basic_metrics: Basic metrics dictionary
            execution_time: Total execution time in seconds
            advanced_metrics: Advanced metrics dictionary (optional)
            config: Configuration dictionary (optional)
            model_size_mb: Model size in MB (optional)
            gpu_memory: GPU memory statistics (optional)
            display_name: Display name for the model (optional)
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        results = {
            'metadata': {
                'model_name': model_name,
                'display_name': display_name if display_name else model_name,
                'dataset_path': dataset_path,
                'timestamp': datetime.now().isoformat(),
                'evaluation_type': 'advanced' if advanced_metrics else 'basic'
            },
            'execution': {
                'total_time_seconds': round(execution_time, 2),
                'time_per_frame_seconds': round(
                    execution_time / basic_metrics['total_frames'], 4
                ) if basic_metrics['total_frames'] > 0 else 0
            },
            'basic_metrics': basic_metrics
        }

        if model_size_mb is not None:
            results['metadata']['model_size_mb'] = round(model_size_mb, 2)

        if gpu_memory:
            results['gpu_memory'] = gpu_memory

        if config:
            results['configuration'] = config

        if advanced_metrics:
            results['advanced_metrics'] = advanced_metrics

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_path}")

    @staticmethod
    def load_results(results_path: str) -> Dict[str, Any]:
        """
        Load results from JSON file.

        Args:
            results_path: Path to results JSON file

        Returns:
            Results dictionary
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")

        with open(results_path, 'r') as f:
            results = json.load(f)

        return results

    @staticmethod
    def merge_results(result_paths: list) -> Dict[str, list]:
        """
        Merge multiple result files for comparison.

        Args:
            result_paths: List of paths to result JSON files

        Returns:
            Dictionary with merged results
        """
        merged = {
            'models': [],
            'results': []
        }

        for path in result_paths:
            results = ResultsWriter.load_results(path)
            model_name = results['metadata']['model_name']

            merged['models'].append(model_name)
            merged['results'].append(results)

        return merged

    @staticmethod
    def format_summary(results: Dict[str, Any]) -> str:
        """
        Format results as a human-readable summary string.

        Args:
            results: Results dictionary

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("Evaluation results summary \n")

        metadata = results['metadata']
        lines.append(f"\nModel: {metadata['model_name']}")
        lines.append(f"Dataset: {metadata['dataset_path']}")
        lines.append(f"Timestamp: {metadata['timestamp']}")
        lines.append(f"Evaluation Type: {metadata['evaluation_type'].upper()}")

        execution = results['execution']
        lines.append(f"\nExecution Time: {execution['total_time_seconds']:.2f} seconds")
        lines.append(f"Time per Frame: {execution['time_per_frame_seconds']:.4f} seconds")

        basic = results['basic_metrics']
        lines.append(f"\nTotal Frames: {basic['total_frames']}")
        lines.append(f"Frames with Detections: {basic['frames_with_detections']}")
        lines.append(f"Detection Rate: {basic['detection_rate']:.2%}")
        lines.append(f"Average Detections per Frame: {basic['average_detections_per_frame']:.2f}")
        lines.append(f"Total Detections: {basic['total_detections']}")

        lines.append("\nDetections by Class:")
        for class_name, count in sorted(basic['detection_counts'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {class_name}: {count}")

        if 'advanced_metrics' in results:
            advanced = results['advanced_metrics']
            lines.append("\nAdvanced Metrics:")
            lines.append(f"Overall Accuracy: {advanced.get('overall_accuracy', 0):.2%}")
            lines.append(f"Mean Average Precision (mAP): {advanced.get('mean_average_precision', 0):.4f}")

            if 'per_class_metrics' in advanced:
                lines.append("\nPer-Class Performance:")
                for class_name, metrics in advanced['per_class_metrics'].items():
                    lines.append(f"\n  {class_name}:")
                    lines.append(f"    Precision: {metrics.get('precision', 0):.4f}")
                    lines.append(f"    Recall: {metrics.get('recall', 0):.4f}")
                    lines.append(f"    F1-Score: {metrics.get('f1_score', 0):.4f}")
                    lines.append(f"    True Positives: {metrics.get('true_positives', 0)}")
                    lines.append(f"    False Positives: {metrics.get('false_positives', 0)}")
                    lines.append(f"    False Negatives: {metrics.get('false_negatives', 0)}")

        return "\n".join(lines)

    @staticmethod
    def print_summary(results: Dict[str, Any]):
        """
        Print formatted results summary to console.

        Args:
            results: Results dictionary
        """
        print(ResultsWriter.format_summary(results))
