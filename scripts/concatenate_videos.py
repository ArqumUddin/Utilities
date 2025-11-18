#!/usr/bin/env python3
"""
Video Comparison and Concatenation Script

Creates side-by-side video comparisons or sequential concatenations from multiple MP4 videos using FFmpeg.
Supports frame rate control and automatic resolution/FPS normalization.
Output videos are automatically saved under the results/ directory.

Usage:
    # Create horizontal side-by-side comparison (saves to results/comparison.mp4)
    python concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 -o comparison.mp4

    # Create sequential concatenation (saves to results/concatenated.mp4)
    python concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 --mode sequential -o concatenated.mp4

    # Sequential concatenation with custom frame rate (saves to results/output_60fps.mp4)
    python concatenate_videos.py video1.mp4 video2.mp4 --mode sequential --fps 60 -o output_60fps.mp4

    # Create 2x2 grid layout (saves to results/grid_comparison.mp4)
    python concatenate_videos.py video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o grid_comparison.mp4 --layout grid

    # Using a file list (saves to results/output.mp4)
    python concatenate_videos.py -i videos.txt --mode sequential -o output.mp4
"""
import argparse
import logging
import os
import sys
from typing import List
from vision_utils import VideoComparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_file_list(filepath: str) -> List[str]:
    """
    Parse a text file containing video paths (one per line).

    Args:
        filepath: Path to file list

    Returns:
        List of video file paths
    """
    with open(filepath, 'r') as f:
        paths = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return paths

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Create side-by-side video comparisons or sequential concatenations from multiple MP4 videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Horizontal side-by-side comparison (saves to results/comparison.mp4)
            %(prog)s video1.mp4 video2.mp4 video3.mp4 -o comparison.mp4

            # Sequential concatenation (saves to results/concatenated.mp4)
            %(prog)s video1.mp4 video2.mp4 video3.mp4 --mode sequential -o concatenated.mp4

            # Sequential with custom FPS (saves to results/output_60fps.mp4)
            %(prog)s video1.mp4 video2.mp4 --mode sequential --fps 60 -o output_60fps.mp4

            # Vertical stack (saves to results/vertical.mp4)
            %(prog)s video1.mp4 video2.mp4 -o vertical.mp4 --layout vertical

            # Grid layout (saves to results/grid.mp4)
            %(prog)s video1.mp4 video2.mp4 video3.mp4 video4.mp4 -o grid.mp4 --layout grid

            Note: Output files are automatically saved under the results/ directory.
        """)

    parser.add_argument(
        'videos',
        nargs='*',
        help='Video files to concatenate'
    )
    parser.add_argument(
        '-i', '--input-list',
        help='Text file containing list of video paths (one per line)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output video file path (automatically saved under results/)'
    )
    parser.add_argument(
        '--mode',
        choices=['side-by-side', 'sequential'],
        default='side-by-side',
        help='Comparison mode: side-by-side (default) or sequential (concatenate end-to-end)'
    )
    parser.add_argument(
        '--layout',
        choices=['horizontal', 'vertical', 'grid'],
        default='horizontal',
        help='Layout type for side-by-side mode: horizontal (default), vertical, or grid'
    )
    parser.add_argument(
        '--fps',
        type=float,
        help='Output frame rate (overrides automatic detection from first video)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_path = args.output
    if not output_path.startswith("results/"):
        output_path = os.path.join("results", output_path)
        logger.info(f"Output path adjusted to: {output_path}")

    if args.input_list:
        if not os.path.exists(args.input_list):
            logger.error(f"Input list file not found: {args.input_list}")
            return 1
        video_paths = parse_file_list(args.input_list)
    elif args.videos:
        video_paths = args.videos
    else:
        logger.error("No video files specified. Use positional arguments or -i/--input-list")
        parser.print_help()
        return 1

    if not video_paths:
        logger.error("No video files specified (empty list)")
        return 1

    logger.info(f"Input videos: {len(video_paths)}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Mode: {args.mode}")
    if args.mode == 'side-by-side':
        logger.info(f"Layout: {args.layout}")
    if args.fps:
        logger.info(f"Target FPS: {args.fps}")

    try:
        comparison = VideoComparison(
            output_path=output_path,
            mode=args.mode,
            layout=args.layout,
            target_fps=args.fps
        )
        comparison.compare(video_paths)

        if args.mode == 'sequential':
            logger.info("Sequential video concatenation completed successfully!")
        else:
            logger.info("Video comparison completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Video comparison failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
