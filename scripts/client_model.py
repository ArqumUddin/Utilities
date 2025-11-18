#!/usr/bin/env python3
"""
Example client demonstrating how to use the DetectionClient
"""
import argparse
import cv2
import sys
from pathlib import Path

from vision_utils import DetectionClient, FrameAnnotator

def main():
    parser = argparse.ArgumentParser(description="Test detection server with an image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--endpoint", type=str, default="detect", help="Server endpoint (default: detect)")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold override")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated text prompts for zero-shot models (e.g., 'person,car,dog')")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated image")

    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to server at http://localhost:{args.port}/{args.endpoint}")
    client = DetectionClient(port=args.port, endpoint=args.endpoint)

    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image: {args.image}", file=sys.stderr)
        sys.exit(1)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Sending prediction request...")

    text_prompts = None
    if args.prompts:
        text_prompts = [p.strip() for p in args.prompts.split(',')]
        print(f"Using text prompts: {text_prompts}")

    try:
        detections = client.predict(
            image=image_rgb,
            confidence_threshold=args.confidence,
            text_prompts=text_prompts
        )

        print(f"\nReceived {len(detections)} detections:")
        for i, det in enumerate(detections, 1):
            print(f"{i}. {det.class_name}: {det.confidence:.3f}")
            print(f"   BBox: ({det.bbox.x_min:.1f}, {det.bbox.y_min:.1f}, "
                  f"{det.bbox.x_max:.1f}, {det.bbox.y_max:.1f})")

        if args.output:
            annotator = FrameAnnotator()
            annotated = annotator.annotate_frame(image_rgb, detections)
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            cv2.imwrite(args.output, annotated_bgr)
            print(f"\nAnnotated image saved to: {args.output}")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
