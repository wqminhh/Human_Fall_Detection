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

DEFAULT_LABELS = ["Fall", "Lying", "Normal"]
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
        "--static-fall-ar-thresh",
        type=float,
        default=0.95,
        help="Image-only width/height threshold used by the static fall heuristic.",
    )
    parser.add_argument(
        "--static-fall-score",
        type=float,
        default=2.0,
        help="Minimum static posture score to classify an image as Fall.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["static", "fsm"],
        default="static",
        help="static = classify image posture directly; fsm = use temporal FSM even for images.",
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
    parser.add_argument(
        "--label-set",
        choices=["auto", "binary", "three-class"],
        default="auto",
        help="auto = use labels found in dataset; binary = Fall/Normal; three-class = Fall/Lying/Normal.",
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


def predict_image(
    detector: YOLOv8FallDetector,
    path: Path,
    lying_ar_thresh: float,
    image_mode: str,
    static_fall_ar_thresh: float,
    static_fall_score: float,
) -> tuple[str, int, list[float]]:
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")

    if image_mode == "static":
        pred, latency_ms = predict_static_image(
            detector, frame, lying_ar_thresh, static_fall_ar_thresh, static_fall_score
        )
        return pred, 1, [latency_ms]

    pred, latency_ms = predict_frame(detector, frame, lying_ar_thresh)
    return pred, 1, [latency_ms]


def predict_static_image(
    detector: YOLOv8FallDetector,
    frame: np.ndarray,
    lying_ar_thresh: float,
    static_fall_ar_thresh: float,
    static_fall_score: float,
) -> tuple[str, float]:
    start = time.perf_counter()
    results = detector.model(
        frame,
        conf=detector.conf_thresh,
        device=detector.device,
        verbose=False,
    )[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    if results.boxes is None or len(results.boxes) == 0:
        return "Normal", latency_ms

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    keypoints = None
    if results.keypoints is not None:
        keypoints = results.keypoints.data.cpu().numpy()

    best_label = "Normal"
    best_score = -1.0
    for idx, bbox in enumerate(boxes):
        kpts = keypoints[idx] if keypoints is not None and idx < len(keypoints) else None
        score, bbox_ar = static_fall_posture_score(bbox, kpts, static_fall_ar_thresh)
        score += min(float(confs[idx]), 1.0) * 0.25

        if score > best_score:
            best_score = score
            if score >= static_fall_score:
                best_label = "Fall"
            elif bbox_ar >= lying_ar_thresh:
                best_label = "Lying"
            else:
                best_label = "Normal"

    return best_label, latency_ms


def static_fall_posture_score(
    bbox: np.ndarray,
    keypoints: np.ndarray | None,
    static_fall_ar_thresh: float,
) -> tuple[float, float]:
    x1, y1, x2, y2 = map(float, bbox)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    bbox_ar = width / height
    score = 0.0

    if bbox_ar >= static_fall_ar_thresh:
        score += 2.0
    elif bbox_ar >= static_fall_ar_thresh * 0.8:
        score += 1.0

    if keypoints is None or len(keypoints) < 17:
        return score, bbox_ar

    torso_angle = torso_angle_from_horizontal(keypoints)
    if torso_angle is not None:
        if torso_angle <= 35.0:
            score += 1.5
        elif torso_angle <= 50.0:
            score += 0.75

    visible_width, visible_height = visible_keypoint_span(keypoints)
    if visible_width > 0 and visible_height > 0:
        kpt_ar = visible_width / max(1.0, visible_height)
        if kpt_ar >= static_fall_ar_thresh:
            score += 1.0

    return score, bbox_ar


def visible_point(keypoints: np.ndarray, idx: int) -> np.ndarray | None:
    if idx >= len(keypoints):
        return None
    point = keypoints[idx]
    conf = point[2] if len(point) >= 3 else 1.0
    if conf <= 0.2 or point[0] <= 0 or point[1] <= 0:
        return None
    return point[:2].astype(float)


def average_visible_points(keypoints: np.ndarray, indices: list[int]) -> np.ndarray | None:
    points = [visible_point(keypoints, idx) for idx in indices]
    points = [point for point in points if point is not None]
    if not points:
        return None
    return np.mean(points, axis=0)


def torso_angle_from_horizontal(keypoints: np.ndarray) -> float | None:
    shoulder_center = average_visible_points(keypoints, [5, 6])
    hip_center = average_visible_points(keypoints, [11, 12])
    if shoulder_center is None or hip_center is None:
        return None

    dx = float(hip_center[0] - shoulder_center[0])
    dy = float(hip_center[1] - shoulder_center[1])
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    return min(angle, 180.0 - angle)


def visible_keypoint_span(keypoints: np.ndarray) -> tuple[float, float]:
    visible = []
    for point in keypoints:
        conf = point[2] if len(point) >= 3 else 1.0
        if conf > 0.2 and point[0] > 0 and point[1] > 0:
            visible.append(point[:2])
    if len(visible) < 3:
        return 0.0, 0.0

    visible_array = np.asarray(visible, dtype=float)
    width = float(np.max(visible_array[:, 0]) - np.min(visible_array[:, 0]))
    height = float(np.max(visible_array[:, 1]) - np.min(visible_array[:, 1]))
    return width, height


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


def normalize_prediction_for_labels(pred_label: str, labels: list[str]) -> str:
    if pred_label == "Lying" and "Lying" not in labels:
        return "Normal"
    return pred_label


def select_labels(items: list[tuple[Path, str]], label_set: str) -> list[str]:
    if label_set == "three-class":
        return DEFAULT_LABELS
    if label_set == "binary":
        return ["Fall", "Normal"]

    found = {label for _, label in items}
    if found <= {"Fall", "Normal"}:
        return ["Fall", "Normal"]
    return [label for label in DEFAULT_LABELS if label in found or label == "Normal"]


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[label_to_idx[true], label_to_idx[pred]] += 1
    return matrix


def compute_metrics(matrix: np.ndarray, labels: list[str]) -> dict[str, object]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for idx, label in enumerate(labels):
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


def save_confusion_plot(matrix: np.ndarray, output_path: Path, labels: list[str]) -> None:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
    )
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.title("Confusion Matrix - Fall Detection Evaluation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_predictions_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "ground_truth",
                "prediction",
                "frames",
                "avg_latency_ms",
                "fps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_results(
    matrix: np.ndarray,
    labels: list[str],
    metrics: dict[str, object],
    avg_latency_ms: float,
) -> None:
    avg_fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    print("\nConfusion Matrix")
    print("Rows = Ground Truth, Columns = Predicted")
    print("          " + "  ".join(f"{label:>7}" for label in labels))
    for label, row in zip(labels, matrix):
        print(f"{label:>8}  " + "  ".join(f"{value:7d}" for value in row))

    print("\nOverall Metrics")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall:    {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1-Score:  {metrics['weighted_f1']:.4f}")

    print("\nPer-Class Metrics")
    for label in labels:
        values = metrics["per_class"][label]
        print(
            f"{label:>8}: "
            f"precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, "
            f"support={values['support']}"
        )

    print("\nInference Speed")
    print(f"Average FPS:        {avg_fps:.2f}")
    print(f"Average Latency:    {avg_latency_ms:.2f} ms/frame")


def main() -> int:
    args = parse_args()
    test_dir = args.test_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = list(iter_labeled_items(test_dir))
    if not items:
        print(f"[ERROR] No labeled video/images found in {test_dir}")
        return 1
    labels = select_labels(items, args.label_set)

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
    prediction_rows = []

    print(f"[INFO] Running evaluation on {len(items)} samples...")
    print(f"[INFO] Evaluation labels: {', '.join(labels)}")
    for idx, (path, true_label) in enumerate(items, start=1):
        reset_detector_state(detector)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTS:
            pred_label, frame_count, latencies_ms = predict_image(
                detector,
                path,
                args.lying_ar_thresh,
                args.image_mode,
                args.static_fall_ar_thresh,
                args.static_fall_score,
            )
        elif suffix in VIDEO_EXTS:
            pred_label, frame_count, latencies_ms = predict_video(
                detector,
                path,
                args.lying_ar_thresh,
                max(1, args.frame_stride),
                args.max_frames,
            )
        else:
            continue

        pred_label = normalize_prediction_for_labels(pred_label, labels)
        y_true.append(true_label)
        y_pred.append(pred_label)
        all_latencies_ms.extend(latencies_ms)
        avg_item_latency = float(np.mean(latencies_ms))
        item_fps = 1000.0 / avg_item_latency if avg_item_latency > 0 else 0.0
        prediction_rows.append(
            {
                "path": str(path),
                "ground_truth": true_label,
                "prediction": pred_label,
                "frames": frame_count,
                "avg_latency_ms": f"{avg_item_latency:.2f}",
                "fps": f"{item_fps:.2f}",
            }
        )

        print(f"[{idx:>3}/{len(items)}] True: {true_label:<6} | Pred: {pred_label:<6} | File: {path.name}")

    matrix = confusion_matrix(y_true, y_pred, labels)
    metrics = compute_metrics(matrix, labels)
    confusion_path = output_dir / "confusion_matrix.png"
    predictions_path = output_dir / "predictions.csv"
    avg_latency_ms = float(np.mean(all_latencies_ms)) if all_latencies_ms else 0.0

    save_confusion_plot(matrix, confusion_path, labels)
    save_predictions_csv(prediction_rows, predictions_path)

    print("\n" + "="*40)
    print(f" ACCURACY:          {metrics['accuracy']*100:.2f}%")
    print(f" WEIGHTED F1-SCORE: {metrics['weighted_f1']*100:.2f}%")
    print("="*40)
    print_results(matrix, labels, metrics, avg_latency_ms)
    print(f"[SUCCESS] Saved confusion matrix: {confusion_path}")
    print(f"[SUCCESS] Saved predictions CSV:  {predictions_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
