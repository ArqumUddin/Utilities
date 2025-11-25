"""
Model Server - Serve any vision_utils model via REST API
"""
import argparse
import sys
from pathlib import Path

from vision_utils import VisionServer, host_model
from vision_utils.io.config import EvaluationConfig

def main():
    parser = argparse.ArgumentParser(description="Serve vision_utils models via REST API")

    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model", type=str, default=None, help="Model name or path")
    parser.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    parser.add_argument("--endpoint", type=str, default="detect", help="Endpoint name (default: detect)")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, default: auto-detect)")
    parser.add_argument("--prompts", type=str, default=None, help="Comma-separated text prompts for zero-shot models")
    parser.add_argument("--max-detections", type=int, default=None, help="Maximum detections per image (default: 100)")
    parser.add_argument("--nms-threshold", type=float, default=None, help="NMS IoU threshold (default: 0.5)")

    args = parser.parse_args()

    # Load from config or use CLI args
    if args.config:
        config = EvaluationConfig(args.config)
        model_name = config.model_name
        confidence = args.confidence if args.confidence is not None else config.confidence_threshold
        device = args.device if args.device is not None else config.device
        text_prompts = config.text_prompts
        max_detections = args.max_detections if args.max_detections is not None else config.max_detections
        nms_threshold = args.nms_threshold if args.nms_threshold is not None else config.nms_threshold
    else:
        if not args.model:
            parser.error("Either --config or --model must be provided")
        model_name = args.model
        confidence = args.confidence if args.confidence is not None else 0.5
        device = args.device
        text_prompts = None
        if args.prompts:
            text_prompts = [p.strip() for p in args.prompts.split(",")]
        max_detections = args.max_detections if args.max_detections is not None else 100
        nms_threshold = args.nms_threshold if args.nms_threshold is not None else 0.5

    print("Vision Utils Model Server")
    print(f"Model: {model_name}")
    print(f"Port: {args.port}")
    print(f"Endpoint: /{args.endpoint}")
    print(f"Device: {device or 'auto-detect'}")
    print(f"Confidence threshold: {confidence}")
    if text_prompts:
        print(f"Text prompts: {text_prompts}")
    print("\nLoading model...")

    try:
        server = VisionServer(
            model_name=model_name,
            confidence_threshold=confidence,
            device=device,
            text_prompts=text_prompts,
            max_detections=max_detections,
            nms_threshold=nms_threshold
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
