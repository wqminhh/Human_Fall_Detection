import cv2
import os
import math
import time
import numpy as np
from collections import deque
import torch
from utils.resource import get_resource_path, get_model_path, check_weights_exist

# Safeguard against PyInstaller / Headless OpenCV missing GUI attributes required by Ultralytics
for _attr, _default_fn in [
    ("imshow", lambda *args, **kwargs: None),
    ("waitKey", lambda *args, **kwargs: 0),
    ("destroyAllWindows", lambda *args, **kwargs: None),
    ("namedWindow", lambda *args, **kwargs: None),
    ("resizeWindow", lambda *args, **kwargs: None),
]:
    if not hasattr(cv2, _attr):
        setattr(cv2, _attr, _default_fn)

# ==============================================================================
# 1. Tracker for YOLOv8 (17 COCO Keypoints)
# ==============================================================================
# COCO Keypoint Map:
# 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
# 5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow,
# 9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip,
# 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle

SKELETON_PAIRS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Face
    (5, 6),  # Shoulders
    (5, 7),
    (7, 9),  # Left Arm
    (6, 8),
    (8, 10),  # Right Arm
    (5, 11),
    (6, 12),  # Torso Sides
    (11, 12),  # Hips
    (11, 13),
    (13, 15),  # Left Leg
    (12, 14),
    (14, 16),  # Right Leg
]


class YOLOv8PersonTracker:
    """Tracks movement and posture over a temporal window to identify fall events."""

    def __init__(
        self,
        tracker_id: int,
        window_size: int = 15,
        fps: int = 30,
        v_thresh: float = 60.0,
        ar_thresh: float = 0.35,
        dy_thresh: float = 20.0,
    ):
        self.tracker_id = tracker_id
        self.window_size = max(3, window_size)
        self.fps = max(1, fps)
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh

        # History queues: store (keypoints, bbox, center_of_mass, aspect_ratio, timestamp)
        self.history = deque(maxlen=self.window_size)
        self.last_update = time.time()
        self.is_falling = False
        self.fall_reason = ""
        self.fall_confirmed_time = 0.0

    def add_frame_data(
        self, keypoints: np.ndarray, bbox: np.ndarray, conf: float = 1.0
    ):
        """
        Add a new detection frame to the tracker history.
        keypoints: shape (17, 3) or (17, 2) [x, y, conf]
        bbox: [x1, y1, x2, y2]
        """
        self.last_update = time.time()
        com = self.compute_center_of_mass(keypoints, bbox)
        ar = self.compute_aspect_ratio(keypoints, bbox)
        posture_y = self.compute_hip_y_with_fallback(keypoints, bbox)
        body_angle = self.compute_body_angle(keypoints)
        bbox_height = self.compute_bbox_height(bbox)
        self.history.append(
            {
                "keypoints": keypoints,
                "bbox": bbox,
                "com": com,
                "ar": ar,
                "posture_y": posture_y,
                "body_angle": body_angle,
                "bbox_height": bbox_height,
                "time": self.last_update,
                "conf": conf,
            }
        )

    def is_ready(self) -> bool:
        return len(self.history) >= min(5, self.window_size)

    def compute_center_of_mass(
        self, keypoints: np.ndarray, bbox: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Center of Mass (CoM) primarily from shoulders and hips.
        Falls back to torso or bounding box center if keypoints are obscured.
        """
        torso_indices = [5, 6, 11, 12]  # L/R Shoulder, L/R Hip
        valid_points = []

        if keypoints is not None and len(keypoints) >= 17:
            for idx in torso_indices:
                pt = keypoints[idx]
                # If keypoint contains confidence
                if len(pt) >= 3:
                    if pt[2] > 0.2 and pt[0] > 0 and pt[1] > 0:
                        valid_points.append([pt[0], pt[1]])
                elif pt[0] > 0 and pt[1] > 0:
                    valid_points.append([pt[0], pt[1]])

        if len(valid_points) >= 2:
            return np.mean(valid_points, axis=0)

        # Fallback to bbox center
        if bbox is not None and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return np.array([cx, cy])

        return np.array([0.0, 0.0])

    def compute_aspect_ratio(
        self, keypoints: np.ndarray, bbox: np.ndarray
    ) -> float:
        """
        Compute width/height aspect ratio.
        A standing person has AR < 0.6; a fallen person has AR > 1.0.
        """
        if bbox is not None and len(bbox) == 4:
            w = max(1.0, float(bbox[2] - bbox[0]))
            h = max(1.0, float(bbox[3] - bbox[1]))
            return w / h

        if keypoints is not None and len(keypoints) > 0:
            xs = keypoints[:, 0]
            ys = keypoints[:, 1]
            valid_xs = xs[xs > 0]
            valid_ys = ys[ys > 0]
            if len(valid_xs) >= 4 and len(valid_ys) >= 4:
                w = float(np.max(valid_xs) - np.min(valid_xs))
                h = float(np.max(valid_ys) - np.min(valid_ys))
                return w / h if h > 1.0 else 0.0

        return 0.0

    def compute_bbox_height(self, bbox: np.ndarray) -> float:
        if bbox is not None and len(bbox) == 4:
            return max(1.0, float(bbox[3] - bbox[1]))
        return 1.0

    def get_valid_keypoint(
        self, keypoints: np.ndarray, idx: int, conf_thresh: float = 0.3
    ):
        if keypoints is None or len(keypoints) <= idx:
            return None

        pt = keypoints[idx]
        conf = pt[2] if len(pt) >= 3 else 1.0
        if conf < conf_thresh or pt[0] <= 0 or pt[1] <= 0:
            return None
        return np.array([float(pt[0]), float(pt[1])])

    def average_keypoints(
        self, keypoints: np.ndarray, indices: list, conf_thresh: float = 0.3
    ):
        points = [
            self.get_valid_keypoint(keypoints, idx, conf_thresh)
            for idx in indices
        ]
        points = [pt for pt in points if pt is not None]
        if not points:
            return None
        return np.mean(points, axis=0)

    def compute_hip_y_with_fallback(
        self, keypoints: np.ndarray, bbox: np.ndarray
    ) -> float:
        """
        Use mid-hip Y as the vertical position. If hip confidence is weak,
        fall back to mid-shoulder Y, then bbox center Y.
        """
        left_hip = self.get_valid_keypoint(keypoints, 11, conf_thresh=0.3)
        right_hip = self.get_valid_keypoint(keypoints, 12, conf_thresh=0.3)
        if left_hip is not None and right_hip is not None:
            return float(np.mean([left_hip[1], right_hip[1]]))

        shoulders = self.average_keypoints(keypoints, [5, 6], conf_thresh=0.3)
        if shoulders is not None:
            return float(shoulders[1])

        if bbox is not None and len(bbox) == 4:
            return float((bbox[1] + bbox[3]) / 2.0)

        return 0.0

    def compute_body_angle(self, keypoints: np.ndarray):
        """
        Angle between mid-shoulders and mid-hips relative to horizontal.
        0 degrees is horizontal/lying; 90 degrees is upright/vertical.
        """
        shoulders = self.average_keypoints(keypoints, [5, 6], conf_thresh=0.3)
        hips = self.average_keypoints(keypoints, [11, 12], conf_thresh=0.3)
        if shoulders is None or hips is None:
            return None

        dx = float(hips[0] - shoulders[0])
        dy = float(hips[1] - shoulders[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        angle = abs(math.degrees(math.atan2(dy, dx)))
        return min(angle, 180.0 - angle)

    def normalize_ratio_threshold(self, value: float, default_ratio: float) -> float:
        """
        Keep backward compatibility with existing controls:
        0.25 means 25%, and 25/30/55 mean 25%/30%/55% body height.
        """
        if value is None:
            return default_ratio
        value = float(value)
        if value <= 0:
            return default_ratio
        return value if value <= 1.0 else value / 100.0

    def check_fall(self):
        """
        Evaluate temporal cues for fall detection:
        - Normalized hip/shoulder drop relative to body height
        - Change in aspect ratio from upright to horizontal
        - Body angle relative to the horizontal axis
        """
        if not self.is_ready():
            return False, None, "Buffering...", ""

        first = self.history[0]
        last = self.history[-1]

        c1, c2 = first["com"], last["com"]
        dx = float(c2[0] - c1[0])
        dy = float(c2[1] - c1[1])  # Positive dy is moving downward in image space
        dist = math.sqrt(dx**2 + dy**2)

        dt = max(0.01, float(last["time"] - first["time"]))
        v = min(dist / dt, 500.0)

        ar_start = first["ar"]
        ar_end = last["ar"]
        ar_delta = ar_end - ar_start
        bbox_height = max(
            1.0,
            float(first.get("bbox_height", 1.0)),
            float(last.get("bbox_height", 1.0)),
        )
        hip_drop = float(
            last.get("posture_y", c2[1]) - first.get("posture_y", c1[1])
        )
        norm_v = hip_drop / bbox_height
        norm_v_thresh = self.normalize_ratio_threshold(self.v_thresh, 0.25)
        norm_dy_thresh = self.normalize_ratio_threshold(self.dy_thresh, 0.15)
        body_angle = last.get("body_angle")
        is_horizontal = body_angle is not None and body_angle < 30.0
        is_upright_stable = (
            body_angle is not None
            and body_angle > 60.0
            and norm_v < max(0.08, norm_v_thresh * 0.5)
        )

        tags = []
        if is_upright_stable:
            angle_text = f"{body_angle:.1f}"
            debug_info = (
                f"norm_v={norm_v:.2f}/{norm_v_thresh:.2f}, "
                f"norm_dy={norm_v:.2f}/{norm_dy_thresh:.2f}, "
                f"angle={angle_text}, AR={ar_end:.2f} (d={ar_delta:+.2f})"
            )
            self.is_falling = False
            self.fall_reason = ""
            return False, last["bbox"], debug_info, ""

        # Rule 1: Resolution-independent downward hip drop.
        if norm_v > norm_v_thresh and ar_end > 0.3:
            tags.append("NormHipDrop")

        # Rule 2: Sudden change from upright to horizontal with downward descent.
        if norm_v > norm_dy_thresh and ar_delta > self.ar_thresh:
            tags.append("DownFlat")

        # Rule 3: Body is nearly horizontal, confirming lying/fall posture.
        if is_horizontal and (norm_v > norm_dy_thresh or ar_end > 0.8):
            tags.append("BodyHorizontal")

        debug_info = (
            f"v={v:.1f}/{self.v_thresh:.1f}, "
            f"dy={dy:.1f}/{self.dy_thresh:.1f}, "
            f"AR={ar_end:.2f} (Δ={ar_delta:+.2f})"
        )

        angle_text = "n/a" if body_angle is None else f"{body_angle:.1f}"
        debug_info = (
            f"norm_v={norm_v:.2f}/{norm_v_thresh:.2f}, "
            f"norm_dy={norm_v:.2f}/{norm_dy_thresh:.2f}, "
            f"angle={angle_text}, AR={ar_end:.2f} (d={ar_delta:+.2f})"
        )

        if len(tags) > 0:
            self.is_falling = True
            self.fall_reason = " + ".join(tags)
            self.fall_confirmed_time = time.time()
            return True, last["bbox"], debug_info, self.fall_reason

        # Reset falling status after 1.5 seconds if no conditions met
        if time.time() - self.fall_confirmed_time > 1.5:
            self.is_falling = False
            self.fall_reason = ""

        return False, last["bbox"], debug_info, ""


# ==============================================================================
# 2. Main YOLOv8 Fall Detector Engine
# ==============================================================================
class YOLOv8FallDetector:
    """
    High-level Inference & Tracking pipeline using YOLOv8-pose.
    Handles frame preprocessing, inference, skeleton drawing, and fall classification.
    """

    def __init__(
        self,
        model_path: str = "weights/yolov8n-pose.pt",
        window_size: int = 15,
        fps: int = 30,
        v_thresh: float = 55.0,
        ar_thresh: float = 0.35,
        dy_thresh: float = 18.0,
        conf_thresh: float = 0.35,
        device: str = None,
    ):
        self.window_size = window_size
        self.fps = fps
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh
        self.conf_thresh = conf_thresh

        # Resolve weight path dynamically for PyInstaller / Dev
        resolved_path = get_model_path(model_path)
        check_weights_exist(resolved_path, raise_error=False)

        print(f"[INFO] Initializing YOLOv8FallDetector with model: {resolved_path}")
        from ultralytics import YOLO

        self.model = YOLO(resolved_path)

        # Device selection
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.trackers = {}
        self.next_tracker_id = 1
        self.total_fall_incidents = 0

    def set_parameters(
        self,
        window_size: int = None,
        fps: int = None,
        v_thresh: float = None,
        ar_thresh: float = None,
        dy_thresh: float = None,
        conf_thresh: float = None,
    ):
        """Update detection thresholds dynamically."""
        if window_size is not None:
            self.window_size = window_size
        if fps is not None:
            self.fps = fps
        if v_thresh is not None:
            self.v_thresh = v_thresh
        if ar_thresh is not None:
            self.ar_thresh = ar_thresh
        if dy_thresh is not None:
            self.dy_thresh = dy_thresh
        if conf_thresh is not None:
            self.conf_thresh = conf_thresh

        # Update existing active trackers
        for tracker in self.trackers.values():
            if window_size is not None:
                tracker.window_size = window_size
            if fps is not None:
                tracker.fps = fps
            if v_thresh is not None:
                tracker.v_thresh = v_thresh
            if ar_thresh is not None:
                tracker.ar_thresh = ar_thresh
            if dy_thresh is not None:
                tracker.dy_thresh = dy_thresh

    def match_detections_to_trackers(
        self, detections: list, dist_thresh: float = 120.0, timeout: float = 1.2
    ):
        """
        Associate detections to existing trackers based on spatial distance.
        """
        now = time.time()
        # Clean up stale trackers
        stale_ids = [
            tid
            for tid, t in self.trackers.items()
            if (now - t.last_update) > timeout
        ]
        for tid in stale_ids:
            del self.trackers[tid]

        assigned_trackers = set()
        matched_pairs = []

        for det in detections:
            bbox = det["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            best_tid = None
            best_dist = float("inf")

            for tid, tracker in self.trackers.items():
                if tid in assigned_trackers:
                    continue
                if len(tracker.history) == 0:
                    continue

                last_com = tracker.history[-1]["com"]
                dist = math.hypot(cx - last_com[0], cy - last_com[1])

                if dist < dist_thresh and dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                assigned_trackers.add(best_tid)
                matched_pairs.append((best_tid, det))
            else:
                # Create new tracker
                tid = self.next_tracker_id
                self.next_tracker_id += 1
                new_tracker = YOLOv8PersonTracker(
                    tracker_id=tid,
                    window_size=self.window_size,
                    fps=self.fps,
                    v_thresh=self.v_thresh,
                    ar_thresh=self.ar_thresh,
                    dy_thresh=self.dy_thresh,
                )
                self.trackers[tid] = new_tracker
                assigned_trackers.add(tid)
                matched_pairs.append((tid, det))

        return matched_pairs

    def draw_skeleton(
        self, image: np.ndarray, keypoints: np.ndarray, is_fall: bool = False
    ):
        """Render COCO 17-keypoint skeleton with glowing colors."""
        if keypoints is None or len(keypoints) < 17:
            return image

        joint_color = (0, 0, 255) if is_fall else (0, 255, 255)
        limb_color = (50, 50, 255) if is_fall else (0, 220, 100)

        # Draw bones
        for p1, p2 in SKELETON_PAIRS:
            kpt1 = keypoints[p1]
            kpt2 = keypoints[p2]
            conf1 = kpt1[2] if len(kpt1) >= 3 else 1.0
            conf2 = kpt2[2] if len(kpt2) >= 3 else 1.0

            if conf1 > 0.25 and conf2 > 0.25:
                x1, y1 = int(kpt1[0]), int(kpt1[1])
                x2, y2 = int(kpt2[0]), int(kpt2[1])
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(image, (x1, y1), (x2, y2), limb_color, 2, cv2.LINE_AA)

        # Draw joints
        for kpt in keypoints:
            conf = kpt[2] if len(kpt) >= 3 else 1.0
            if conf > 0.25:
                x, y = int(kpt[0]), int(kpt[1])
                if x > 0 and y > 0:
                    cv2.circle(image, (x, y), 4, joint_color, -1, cv2.LINE_AA)
                    cv2.circle(image, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return image

    def draw_person_overlay(
        self,
        image: np.ndarray,
        tracker_id: int,
        bbox: np.ndarray,
        is_fall: bool,
        fall_reason: str,
        debug_info: str,
    ):
        """Draw bounding box, tracker ID tag, metrics, and alert badge."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if is_fall:
            # Red bounding box with thick highlight
            box_color = (0, 0, 255)  # Bright Red
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)

            # Prominent Fall Alert Tag
            label = f"🚨 FALL DETECTED (ID: #{tracker_id})"
            sublabel = (
                f"[{fall_reason}]" if fall_reason else "[Posture Anomaly]"
            )

            # Background rectangle for text
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)[0]
            tag_y = max(y1 - 10, t_size[1] + 10)
            cv2.rectangle(
                image,
                (x1, tag_y - t_size[1] - 8),
                (x1 + t_size[0] + 16, tag_y + 6),
                (0, 0, 200),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 8, tag_y - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Sub-reason label
            cv2.putText(
                image,
                sublabel,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # Normal state - Emerald Green / Cyan
            box_color = (0, 200, 100)
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            label = f"Person #{tracker_id} [NORMAL]"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            tag_y = max(y1 - 6, t_size[1] + 6)
            cv2.rectangle(
                image,
                (x1, tag_y - t_size[1] - 4),
                (x1 + t_size[0] + 8, tag_y + 4),
                (0, 150, 50),
                -1,
            )
            cv2.putText(
                image,
                label,
                (x1 + 4, tag_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Draw metrics below person
        if debug_info:
            cv2.putText(
                image,
                debug_info,
                (x1, min(h - 10, y2 + 40)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return image

    def process_frame(self, frame: np.ndarray, prev_time: float = None):
        """
        Process a single image/video frame through YOLOv8 pose estimation and fall tracker.

        Returns:
            processed_image: Annotated BGR frame
            stats: Dictionary containing frame statistics:
                   {
                       'has_fall': bool,
                       'fall_count': int,
                       'active_persons': int,
                       'fall_events': list of dicts,
                       'fps': float,
                       'new_time': float
                   }
        """
        start_time = time.time()
        annotated_frame = frame.copy()

        # Run YOLOv8 Pose Inference
        results = self.model(
            frame,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if (
            results.boxes is not None
            and results.keypoints is not None
            and len(results.boxes) > 0
        ):
            boxes_data = results.boxes.xyxy.cpu().numpy()
            conf_data = results.boxes.conf.cpu().numpy()
            kpts_data = (
                results.keypoints.data.cpu().numpy()
            )  # shape: (N, 17, 3)

            for i in range(len(boxes_data)):
                detections.append(
                    {
                        "bbox": boxes_data[i],
                        "conf": float(conf_data[i]),
                        "keypoints": kpts_data[i],
                    }
                )

        # Match detections to trackers
        matched = self.match_detections_to_trackers(detections)

        has_fall_this_frame = False
        fall_events = []

        for tid, det in matched:
            tracker = self.trackers[tid]
            tracker.add_frame_data(
                det["keypoints"], det["bbox"], conf=det["conf"]
            )

            is_fall, bbox, debug_info, tag = tracker.check_fall()

            if is_fall:
                has_fall_this_frame = True
                self.total_fall_incidents += 1
                fall_events.append(
                    {
                        "tracker_id": tid,
                        "tag": tag,
                        "debug": debug_info,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "bbox": bbox.tolist() if bbox is not None else [],
                    }
                )

            # Draw skeleton and overlay box
            annotated_frame = self.draw_skeleton(
                annotated_frame, det["keypoints"], is_fall=is_fall
            )
            annotated_frame = self.draw_person_overlay(
                annotated_frame,
                tracker_id=tid,
                bbox=det["bbox"],
                is_fall=is_fall,
                fall_reason=tag,
                debug_info=debug_info,
            )

        # Calculate FPS
        curr_time = time.time()
        fps = (
            1.0 / max(0.001, (curr_time - prev_time))
            if prev_time is not None
            else (1.0 / max(0.001, (curr_time - start_time)))
        )

        stats = {
            "has_fall": has_fall_this_frame,
            "fall_count": len(fall_events),
            "active_persons": len(matched),
            "fall_events": fall_events,
            "fps": fps,
            "new_time": curr_time,
        }

        return annotated_frame, stats

    def handle_frame(self, frame: np.ndarray, prev_time: float = None):
        """Helper method compatible with legacy calling convention."""
        annotated_frame, stats = self.process_frame(frame, prev_time)
        return annotated_frame, stats["new_time"]


# ==============================================================================
# 3. Legacy YOLOv7 Pose Fall Detector (Preserved for compatibility)
# ==============================================================================
class PersonFallTracker:
    def __init__(self, window_size, fps, v_thresh, ar_thresh, dy_thresh):
        self.pose_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.fps = fps
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh

    def add_pose(self, pose):
        if self.is_pose_complete(pose):
            self.pose_window.append(pose)

    def is_ready(self):
        return len(self.pose_window) == self.window_size

    def compute_center_of_mass(self, pose):
        return np.mean(
            [
                [pose[10], pose[11]],  # left shoulder
                [pose[13], pose[14]],  # right shoulder
                [pose[22], pose[23]],  # left hip
                [pose[25], pose[26]],  # right hip
            ],
            axis=0,
        )

    def compute_velocity(self, p1, p2):
        c1 = self.compute_center_of_mass(p1)
        c2 = self.compute_center_of_mass(p2)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        dist = math.sqrt(dx**2 + dy**2)
        t = (self.window_size - 1) / self.fps
        return min(dist / t, 300.0), dy

    def compute_ar_delta(self, p1, p2):
        def ar(p):
            length = len(p) - (len(p) % 3)
            x = [p[i] for i in range(0, length, 3)]
            y = [p[i + 1] for i in range(0, length, 3)]
            w, h = max(x) - min(x), max(y) - min(y)
            return w / h if h else 0

        return ar(p2) - ar(p1)

    def check_fall(self):
        if not self.is_ready():
            return False, None, None, ""

        p1, p2 = self.pose_window[0], self.pose_window[-1]
        v, dy = self.compute_velocity(p1, p2)
        ar_delta = self.compute_ar_delta(p1, p2)

        ar_end = self._safe_aspect_ratio(p2)

        tag = []
        if v > self.v_thresh and dy > self.dy_thresh and ar_end > 0.1:
            tag.append("SpeedDrop")
        if dy > self.dy_thresh and ar_delta > self.ar_thresh:
            tag.append("DownFlat")

        debug = (
            f"v={v:.1f}/{self.v_thresh:.1f}, "
            f"dy={dy:.1f}/{self.dy_thresh:.1f}, "
            f"ar={ar_delta:.2f}/{self.ar_thresh:.2f}"
        )

        if tag:
            cx, cy, w, h = p2[2], p2[3], p2[4], p2[5]
            return (
                True,
                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                debug,
                " ".join(tag),
            )

        return False, None, debug, ""

    def _safe_aspect_ratio(self, p):
        length = len(p) - (len(p) % 3)
        x = [p[i] for i in range(0, length, 3)]
        y = [p[i + 1] for i in range(0, length, 3)]
        w, h = max(x) - min(x), max(y) - min(y)
        return w / h if h else 0

    def is_pose_complete(self, pose, required_joints=(11, 14, 23, 26)):
        try:
            complete = True
            visible_joints = 0
            length = len(pose) - (len(pose) % 3)

            for i in range(0, length, 3):
                conf = pose[i + 2]
                if conf > 0.2:
                    visible_joints += 1

            for idx in required_joints:
                if pose[idx] == 0 or pose[idx + 1] == 0:
                    complete = False

            return complete and visible_joints >= 10
        except IndexError:
            return False


class FallDetectorMulti:
    """Legacy detector for YOLOv7 pose weights."""

    def __init__(
        self,
        model_path="yolov7-w6-pose.pt",
        window_size=10,
        fps=30,
        v_thresh=60.0,
        ar_thresh=0.35,
        dy_thresh=20.0,
    ):
        resolved_path = get_resource_path(model_path)
        if os.path.exists(resolved_path):
            self.model, self.device = self.load_model(resolved_path)
        else:
            self.model, self.device = None, "cpu"

        self.trackers = {}
        self.window_size = window_size
        self.fps = fps
        self.v_thresh = v_thresh
        self.ar_thresh = ar_thresh
        self.dy_thresh = dy_thresh
        self.next_id = 1

    def load_model(self, path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = torch.load(path, map_location=device, weights_only=False)
        model = weights["model"].float().eval()
        return (model.half().to(device) if torch.cuda.is_available() else model), device

    def handle_frame(self, frame, prev_time=None, writer=None):
        return frame, time.time()
