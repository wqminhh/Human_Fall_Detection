"""
Evaluates the YOLOv8-Pose fall detection pipeline on a labeled test folder
with optimized thresholds and temporal smoothing.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fall_core import YOLOv8FallDetector

LABELS = ["Fall", "Lying", "Normal"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpeg", ".mpg"}
LABEL_ALIASES = {
    "fall": "Fall",
    "falls": "Fall",
    "fallen": "Fall",
    "lying": "Lying",
    "lie": "Lying",
    "laying": "Lying",
    "normal": "Normal",
    "not_fall": "Normal",
    "not-fall": "Normal",
    "nonfall": "Normal",
    "no_fall": "Normal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Human Fall Detection on labeled images/videos."
    )
    parser.add_argument(
        "test_dir",
        type=Path,
        help="Root folder with class subfolders: Fall, Lying, Normal.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLOv8 pose weight path. Default: yolov8n-pose.pt",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda:0, etc.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--fps", type=int, default=30, help="Tracker FPS setting.")
    parser.add_argument(
        "--window-size", type=int, default=15, help="Temporal tracker window size."
    )
    # Tối ưu lại các ngưỡng mặc định giúp nâng cao Recall & Precision
    parser.add_argument(
        "--v-thresh", type=float, default=35.0, help="Fall velocity threshold (Optimized)."
    )
    parser.add_argument(
        "--dy-thresh", type=float, default=12.0, help="Fall vertical drop threshold (Optimized)."
    )
    parser.add_argument(
        "--ar-thresh", type=float, default=0.25, help="Fall aspect-ratio delta threshold."
    )
    parser.add_argument(
        "--lying-ar-thresh",
        type=float,
        default=1.1,
        help="Aspect ratio threshold to classify posture as Lying.",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1, help="Evaluate every Nth frame."
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Max frames per video. 0 = no limit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for confusion_matrix.png and predictions.csv.",
    )
    return parser.parse_args()


def canonical_label(name: str) -> str | None:
    return LABEL_ALIASES.get(name.strip().lower())


def iter_labeled_items(test_dir: Path) -> Iterable[tuple[Path, str]]:
    for label_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = canonical_label(label_dir.name)
        if label is None:
            continue

        for path in sorted(label_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                yield path, label


def reset_detector_state(detector: YOLOv8FallDetector) -> None:
    detector.trackers.clear()
    detector.next_tracker_id = 1
    detector.total_fall_incidents = 0


def current_posture_label(
    detector: YOLOv8FallDetector, lying_ar_thresh: float
) -> str:
    max_ar = 0.0
    has_person = False

    for tracker in detector.trackers.values():
        if not tracker.history:
            continue
        has_person = True
        max_ar = max(max_ar, float(tracker.history[-1].get("ar", 0.0)))

    if has_person and max_ar >= lying_ar_thresh:
        return "Lying"
    return "Normal"


def predict_frame(
    detector: YOLOv8FallDetector,
    frame: np.ndarray,
    lying_ar_thresh: float,
) -> tuple[str, float]:
    start = time.perf_counter()
    _, stats = detector.process_frame(frame)
    latency_ms = (time.perf_counter() - start) * 1000.0

    if stats.get("has_fall", False):
        return "Fall", latency_ms
    return current_posture_label(detector, lying_ar_thresh), latency_ms


def predict_video(
    detector: YOLOv8FallDetector,
    path: Path,
    lying_ar_thresh: float,
    frame_stride: int,
    max_frames: int,
) -> tuple[str, int, list[float]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    frame_idx = 0
    evaluated = 0
    latencies_ms: list[float] = []
    frame_preds: list[str] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride == 0:
            pred, latency_ms = predict_frame(detector, frame, lying_ar_thresh)
            frame_preds.append(pred)
            latencies_ms.append(latency_ms)
            evaluated += 1

            if max_frames > 0 and evaluated >= max_frames:
                break

        frame_idx += 1

    cap.release()

    if evaluated == 0:
        raise ValueError(f"No frames evaluated for video: {path}")

    return aggregate_video_prediction(frame_preds), evaluated, latencies_ms


def aggregate_video_prediction(frame_preds: list[str]) -> str:
    """
    Nâng cấp thuật toán Gom kết quả Video:
    Nếu có hành vi Fall kéo dài từ 2-3 frames liên tiếp -> Báo 'Fall' (Giảm nhiễu do 1 frame lỗi).
    """
    fall_count = 0
    for pred in frame_preds:
        if pred == "Fall":
            fall_count += 1
            if fall_count >= 2:  # Bắt buộc xuất hiện ít nhất 2 frames ngã
                return "Fall"
        else:
            fall_count = 0

    counts = Counter(frame_preds)
    if counts["Lying"] > counts["Normal"]:
        return "Lying"
    return "Normal"


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[true], label_to_idx[pred]] += 1
    return matrix


def compute_metrics(matrix: np.ndarray) -> dict[str, object]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for idx, label in enumerate(LABELS):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    weights = np.array(supports, dtype=float)
    weighted = lambda values: float(np.average(values, weights=weights)) if total else 0.0

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "weighted_precision": weighted(precisions),
        "weighted_recall": weighted(recalls),
        "weighted_f1": weighted(f1s),
    }


def save_confusion_plot(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cbar=False,
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.title("Confusion Matrix - Fall Detection Evaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> int:
    args = parse_args()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_labeled_items(test_dir))
    if not items:
        print(f"[ERROR] No labeled video/images found in {test_dir}")
        return 1

    detector = YOLOv8FallDetector(
        model_path=args.model,
        window_size=args.window_size,
        fps=args.fps,
        v_thresh=args.v_thresh,
        ar_thresh=args.ar_thresh,
        dy_thresh=args.dy_thresh,
        conf_thresh=args.conf,
        device=args.device,
    )

    y_true, y_pred, all_latencies_ms = [], [], []

    print(f"[INFO] Running evaluation on {len(items)} samples...")
    for idx, (path, true_label) in enumerate(items, start=1):
        reset_detector_state(detector)
        pred_label, frame_count, latencies_ms = predict_video(
            detector, path, args.lying_ar_thresh, max(1, args.frame_stride), args.max_frames
        )
        y_true.append(true_label)
        y_pred.append(pred_label)
        all_latencies_ms.extend(latencies_ms)

        print(f"[{idx:>3}/{len(items)}] True: {true_label:<6} | Pred: {pred_label:<6} | File: {path.name}")

    matrix = confusion_matrix(y_true, y_pred)
    metrics = compute_metrics(matrix)
    save_confusion_plot(matrix, output_dir / "confusion_matrix.png")

    print("\n" + "="*40)
    print(f" ACCURACY:          {metrics['accuracy']*100:.2f}%")
    print(f" WEIGHTED F1-SCORE: {metrics['weighted_f1']*100:.2f}%")
    print("="*40)
    print(f"[SUCCESS] Saved confusion_matrix.png to {output_dir}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())