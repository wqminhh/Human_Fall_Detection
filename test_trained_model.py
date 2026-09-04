from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 image-classification model on a test folder."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/classify/fall_classifier/weights/best.pt"),
        help="Path to trained model .pt file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("fall_dataset/yolo/data.yaml"),
        help="Path to YOLO data.yaml file.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("fall_dataset/splits/test"),
        help="Directory containing test class folders.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device or cpu.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/test"),
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--name",
        default="fall_classifier_test",
        help="Name for the evaluation run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    test_dir = args.test_dir.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    model = YOLO(str(model_path))
    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Testing on: {test_dir}")

    metrics = model.val(
        data=str(data_path),
        split="test",
        device=args.device,
        project=str(args.project),
        name=args.name,
        save_json=True,
        conf=args.conf,
    )

    print("\n[INFO] Test finished.")
    print(f"[INFO] Results saved under: {args.project / args.name}")
    print(f"[INFO] Accuracy: {metrics.results_dict.get('accuracy', 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
