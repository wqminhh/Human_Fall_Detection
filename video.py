from fall_core import YOLOv8FallDetector  # Changed from FallDetectorMulti
from config import FPS, WINDOW_SIZE, V_THRESH, DY_THRESH, ASPECT_RATIO_THRESH
import os
import cv2


def process_video():
    if os.environ.get("CI_MODE") == "1":
        videos_path = "fall_dataset/ci_videos"
        print("[CI MODE] Only running on CI test videos...")
    else:
        videos_path = "fall_dataset/videos"

    output_dir = "output_videos"
    os.makedirs(output_dir, exist_ok=True)

    detector = YOLOv8FallDetector(
        window_size=WINDOW_SIZE,
        fps=FPS,
        v_thresh=V_THRESH,
        dy_thresh=DY_THRESH,
        ar_thresh=ASPECT_RATIO_THRESH,
    )

    for video in os.listdir(videos_path):
        if not video.lower().endswith((".mp4", ".avi", ".mov")):
            continue
        video_path = os.path.join(videos_path, video)
        process_video_file(detector, video_path, output_dir)


def process_video_file(self, video_path: str, output_dir: str):
    """
    Process a video file frame-by-frame for fall detection.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save the output video
    """
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_detected.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    prev_time = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            annotated_frame, prev_time = self.handle_frame(frame, prev_time, writer)
            
            # Write frame to output video
            writer.write(annotated_frame)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count}/{total_frames} frames")
    
    except Exception as e:
        print(f"[ERROR] Error processing video: {e}")
    
    finally:
        cap.release()
        writer.release()
        print(f"[INFO] Video processing complete. Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()