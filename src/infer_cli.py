from __future__ import annotations

import argparse
import json

from src.inference import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(checkpoint_path=args.checkpoint, image_size=args.image_size, top_k=args.top_k)
    output = engine.predict(args.image_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
