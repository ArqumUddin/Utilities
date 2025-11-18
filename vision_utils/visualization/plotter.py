"""
Results plotter for visualizing and comparing model evaluation results.
"""
import os
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

from ..io.results import ResultsWriter

class ResultsPlotter:
    """
    Create visualization plots from evaluation results using Plotly.
    Supports multi-model comparison plots only.
    """
    MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up',
                      'triangle-down', 'star', 'hexagon', 'pentagon']

    COLOR_PALETTE = px.colors.qualitative.Set2

    def __init__(self, template: str = 'plotly_white'):
        """
        Initialize results plotter.

        Args:
            template: Plotly template to use for styling
        """
        self.template = template

    @staticmethod
    def _get_class_synonyms() -> Dict[str, str]:
        """
        Define class synonym mappings for combining similar classes.

        Returns:
            Dictionary mapping class names to their canonical form

        Examples:
            "sofa" -> "couch/sofa"
            "couch" -> "couch/sofa"
            "tv" -> "tv/monitor"
        """
        return {
            'sofa': 'couch/sofa',
            'couch': 'couch/sofa',
            'tv': 'tv/monitor',
            'television': 'tv/monitor',
            'monitor': 'tv/monitor',
            'tvmonitor': 'tv/monitor',
        }

    @staticmethod
    def _normalize_class_name(cls_name: str) -> str:
        """
        Normalize class names to handle inconsistent naming.
        Removes all spaces, underscores, and hyphens to handle compound words.

        Args:
            cls_name: Original class name

        Returns:
            Normalized class name (lowercase, no spaces/underscores)

        Examples:
            "Potted Plant" -> "pottedplant"
            "potted_plant" -> "pottedplant"
            "pottedplant" -> "pottedplant"
            "potted-plant" -> "pottedplant"
        """
        normalized = cls_name.strip().lower()
        normalized = re.sub(r'[\s_-]+', '', normalized)

        return normalized

    def compare_models(self, result_paths: List[str], output_dir: str = None):
        """
        Generate comparison plots for multiple models.

        Args:
            result_paths: List of paths to result JSON files
            output_dir: Directory to save plots (default: results/comparison_plots)
        """
        results_list = [ResultsWriter.load_results(path) for path in result_paths]
        model_names = [r['metadata'].get('display_name', r['metadata']['model_name']) for r in results_list]
        if output_dir is None:
            output_dir = "results/comparison_plots"

        os.makedirs(output_dir, exist_ok=True)

        print(f"\nGenerating comparison plots for {len(model_names)} models...")
        self._plot_detection_counts_comparison(
            results_list,
            model_names,
            os.path.join(output_dir, "detection_counts_comparison.png")
        )

        self._plot_memory_consumption_comparison(
            results_list,
            model_names,
            os.path.join(output_dir, "memory_consumption_comparison.png")
        )

        self._plot_inference_fps_comparison(
            results_list,
            model_names,
            os.path.join(output_dir, "inference_fps_comparison.png")
        )

        self._plot_execution_time_comparison(
            results_list,
            model_names,
            os.path.join(output_dir, "execution_time_comparison.png")
        )

        if all(r['metadata']['evaluation_type'] == 'advanced' for r in results_list):
            self._plot_map_comparison(
                results_list,
                model_names,
                os.path.join(output_dir, "map_comparison.png")
            )

            self._plot_precision_recall_comparison(
                results_list,
                model_names,
                os.path.join(output_dir, "precision_recall_comparison.png")
            )

        print(f"Comparison plots saved to: {output_dir}")

    def _plot_detection_counts_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot detection count comparison as a dot plot with different shapes and colors per model."""
        all_normalized_classes = {}

        for results in results_list:
            detection_counts = results['basic_metrics']['detection_counts']
            for original_cls in detection_counts.keys():
                normalized = self._normalize_class_name(original_cls)
                if normalized not in all_normalized_classes:
                    all_normalized_classes[normalized] = original_cls.strip().lower()

        all_classes = sorted(list(all_normalized_classes.keys()))
        display_names = [all_normalized_classes[cls] for cls in all_classes]

        if len(all_classes) == 0:
            print("Skipping detection counts comparison (no detections)")
            return

        fig = go.Figure()

        for model_idx, (results, model_name) in enumerate(zip(results_list, model_names)):
            detection_counts = results['basic_metrics']['detection_counts']
            normalized_counts = {}
            for original_cls, count in detection_counts.items():
                normalized = self._normalize_class_name(original_cls)
                normalized_counts[normalized] = normalized_counts.get(normalized, 0) + count

            counts = [normalized_counts.get(cls, 0) for cls in all_classes]
            x_positions = list(range(len(all_classes)))
            marker_symbol = self.MARKER_SYMBOLS[model_idx % len(self.MARKER_SYMBOLS)]
            color = self.COLOR_PALETTE[model_idx % len(self.COLOR_PALETTE)]

            fig.add_trace(go.Scatter(
                x=x_positions,
                y=counts,
                mode='markers+lines',
                name=model_name,
                marker=dict(
                    symbol=marker_symbol,
                    size=14,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                line=dict(
                    color=color,
                    width=1.5,
                    shape='spline',
                    smoothing=1.0 
                ),
                hovertemplate='<b>%{customdata}</b><br>Count: %{y}<br>Model: ' + model_name + '<extra></extra>',
                customdata=display_names
            ))

        fig.update_layout(
            title=dict(
                text='Detection Counts Comparison by Class',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Detected Class',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickmode='array',
                tickvals=list(range(len(all_classes))),
                ticktext=display_names,
                tickangle=-45,
                tickfont=dict(size=12),
                dtick=1
            ),
            yaxis=dict(
                title='Quantity Detected',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=11)
            ),
            template=self.template,
            hovermode='closest',
            legend=dict(
                title=dict(text='Model', font=dict(size=12)),
                font=dict(size=11),
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02
            ),
            width=1400,
            height=600,
            margin=dict(l=80, r=150, t=80, b=140)
        )

        fig.write_image(output_path, width=1400, height=600, scale=2)
        print(f"  ✓ Detection counts dot plot saved")

    def _plot_memory_consumption_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot memory consumption comparison using grouped bar chart (GPU memory + model size)."""
        gpu_memory = []
        model_sizes = []

        for results in results_list:
            if 'gpu_memory' in results and results['gpu_memory'].get('cuda_available', False):
                gpu_memory.append(results['gpu_memory']['average_memory_mb'])
            else:
                gpu_memory.append(0.0)
            model_sizes.append(results['metadata'].get('model_size_mb', 0.0))

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='GPU Memory',
            x=model_names,
            y=gpu_memory,
            marker_color='steelblue',
            marker_line=dict(color='black', width=1),
            text=[f'{m:.1f}' for m in gpu_memory],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>GPU Memory: %{y:.1f} MB<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Model Size',
            x=model_names,
            y=model_sizes,
            marker_color='darkorange',
            marker_line=dict(color='black', width=1),
            text=[f'{s:.1f}' for s in model_sizes],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Model Size: %{y:.1f} MB<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text='Memory Consumption Comparison: GPU Memory & Model Size',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Model',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title='Memory Consumption (MB)',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=11),
                gridcolor='lightgray'
            ),
            template=self.template,
            barmode='group',
            legend=dict(
                font=dict(size=11),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=120)
        )

        fig.write_image(output_path, width=1200, height=600, scale=2)
        print(f"  ✓ Memory consumption comparison saved")

    def _plot_inference_fps_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot inference FPS comparison with color coding."""
        fps_values = [results['basic_metrics'].get('inference_fps', 0.0) for results in results_list]
        colors = []
        for fps in fps_values:
            if fps >= 30:
                colors.append('green')
            elif fps >= 15:
                colors.append('orange')
            else:
                colors.append('red')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=model_names,
            y=fps_values,
            marker_color=colors,
            marker_line=dict(color='black', width=1.5),
            text=[f'{fps:.1f}' for fps in fps_values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>FPS: %{y:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text='Model Inference FPS Comparison',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Model',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title='Frames Per Second (FPS)',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=11),
                gridcolor='lightgray'
            ),
            template=self.template,
            showlegend=False,
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=80, b=120)
        )

        fig.write_image(output_path, width=1200, height=600, scale=2)
        print(f"  ✓ Inference FPS comparison saved")

    def _plot_execution_time_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot execution time comparison with grouped bars and dual Y-axes."""
        times = [r['execution']['total_time_seconds'] for r in results_list]
        times_per_frame_seconds = [r['execution']['time_per_frame_seconds'] for r in results_list]

        times_per_frame_ms = [t * 1000 for t in times_per_frame_seconds]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                name='Total Execution Time',
                x=model_names,
                y=times,
                marker_color='steelblue',
                marker_line=dict(color='black', width=1),
                text=[f'{t:.2f}s' for t in times],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Total Time: %{y:.2f}s<extra></extra>'
            ),
            secondary_y=False
        )

        fig.add_trace(
            go.Scatter(
                name='Time per Frame',
                x=model_names,
                y=times_per_frame_ms,
                mode='lines+markers',
                line=dict(color='coral', width=3),
                marker=dict(size=10, color='coral', line=dict(color='white', width=2)),
                text=[f'{t:.1f}ms' for t in times_per_frame_ms],
                textposition='top center',
                hovertemplate='<b>%{x}</b><br>Time per Frame: %{y:.1f}ms<extra></extra>'
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text='Execution Time Comparison',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Model',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            template=self.template,
            legend=dict(
                font=dict(size=11),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=120)
        )

        fig.update_yaxes(
            title_text="Total Execution Time (seconds)",
            title_font=dict(size=14, color='steelblue'),
            tickfont=dict(color='steelblue'),
            gridcolor='lightgray',
            secondary_y=False
        )
        fig.update_yaxes(
            title_text="Time per Frame (milliseconds)",
            title_font=dict(size=14, color='coral'),
            tickfont=dict(color='coral'),
            secondary_y=True
        )

        fig.write_image(output_path, width=1200, height=600, scale=2)
        print(f"  ✓ Execution time comparison saved")

    def _plot_map_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot mAP comparison."""
        maps = [r['advanced_metrics']['mean_average_precision'] for r in results_list]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=model_names,
            y=maps,
            marker_color='mediumseagreen',
            marker_line=dict(color='black', width=1.5),
            text=[f'{m:.4f}' for m in maps],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>mAP: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text='Mean Average Precision (mAP) Comparison',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Model',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title='mAP',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=11),
                range=[0, 1.1],
                gridcolor='lightgray'
            ),
            template=self.template,
            showlegend=False,
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=80, b=120)
        )

        fig.write_image(output_path, width=1200, height=600, scale=2)
        print(f"  ✓ mAP comparison saved")

    def _plot_precision_recall_comparison(self, results_list: List[Dict], model_names: List[str], output_path: str):
        """Plot precision and recall comparison."""
        avg_precision = []
        avg_recall = []

        for results in results_list:
            per_class = results['advanced_metrics']['per_class_metrics']
            if len(per_class) > 0:
                avg_p = np.mean([m['precision'] for m in per_class.values()])
                avg_r = np.mean([m['recall'] for m in per_class.values()])
            else:
                avg_p = avg_r = 0.0
            avg_precision.append(avg_p)
            avg_recall.append(avg_r)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Precision',
            x=model_names,
            y=avg_precision,
            marker_color='royalblue',
            marker_line=dict(color='black', width=1),
            text=[f'{p:.3f}' for p in avg_precision],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Precision: %{y:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Bar(
            name='Recall',
            x=model_names,
            y=avg_recall,
            marker_color='lightcoral',
            marker_line=dict(color='black', width=1),
            text=[f'{r:.3f}' for r in avg_recall],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Recall: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text='Average Precision and Recall Comparison',
                font=dict(size=16, family='Arial, sans-serif', color='#333')
            ),
            xaxis=dict(
                title='Model',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title='Score',
                title_font=dict(size=14, family='Arial, sans-serif'),
                tickfont=dict(size=11),
                range=[0, 1.1],
                gridcolor='lightgray'
            ),
            template=self.template,
            barmode='group',
            legend=dict(
                font=dict(size=11),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            width=1200,
            height=600,
            margin=dict(l=80, r=80, t=100, b=120)
        )

        fig.write_image(output_path, width=1200, height=600, scale=2)
        print(f"  ✓ Precision and recall comparison saved")
