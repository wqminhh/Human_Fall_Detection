from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 image-classification model on the fall/not-fall dataset split."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("fall_dataset/yolo/data.yaml"),
        help="Path to the YOLO data.yaml file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-cls.pt",
        help="Base model. Default: yolov8n-cls.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device, e.g. 0 or cpu. Use '0' for first GPU or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/classify"),
        help="Training output directory.",
    )
    parser.add_argument(
        "--name",
        default="fall_classifier",
        help="Run name.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_file = args.data.resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Data config file not found: {data_file}")

    model = YOLO(args.model)

    print(f"[INFO] Training model: {args.model}")
    print(f"[INFO] Data YAML: {data_file}")
    print(f"[INFO] Output dir: {args.project / args.name}")

    results = model.train(
        data=str(data_file),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        device=args.device,
        project=str(args.project),
        name=args.name,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        verbose=True,
    )

    print("\n[INFO] Training finished.")
    print(f"[INFO] Best weights saved in: {args.project / args.name / 'weights' }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
