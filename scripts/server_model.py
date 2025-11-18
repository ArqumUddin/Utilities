"""
Model Server - Serve any vision_utils model via REST API
"""
import argparse
import sys
from pathlib import Path

from vision_utils import DetectionServer, host_model

def main():
    parser = argparse.ArgumentParser(description="Serve vision_utils models via REST API")

    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    parser.add_argument("--endpoint", type=str, default="detect", help="Endpoint name (default: detect)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto-detect)")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated text prompts for zero-shot models")
    parser.add_argument("--max-detections", type=int, default=100, help="Maximum detections per image (default: 100)")
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold (default: 0.5)")

    args = parser.parse_args()

    text_prompts = None
    if args.prompts:
        text_prompts = [p.strip() for p in args.prompts.split(",")]

    print("Vision Utils Model Server")
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Endpoint: /{args.endpoint}")
    print(f"Device: {args.device or 'auto-detect'}")
    print(f"Confidence threshold: {args.confidence}")
    if text_prompts:
        print(f"Text prompts: {text_prompts}")
    print("\nLoading model...")

    try:
        server = DetectionServer(
            model_name=args.model,
            confidence_threshold=args.confidence,
            device=args.device,
            text_prompts=text_prompts,
            max_detections=args.max_detections,
            nms_threshold=args.nms_threshold
        )

        print(f"\nServer ready at: http://localhost:{args.port}/{args.endpoint}")
        print("Starting server...\n")

        host_model(server, name=args.endpoint, port=args.port)

    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
