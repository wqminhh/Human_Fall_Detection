"""
Human Fall Detection Desktop GUI Application
Powered by YOLOv8-pose and CustomTkinter
"""

import os
import sys
import subprocess

# Auto-detect incompatible Python 3.13 freethreaded environment and redirect to Python 3.12
if sys.version_info >= (3, 13):
    try:
        import numpy
        import cv2
        import torch
        import ultralytics
        import customtkinter
    except Exception:
        print("=" * 70)
        print("[WARN] Python 3.13 freethreaded detected (C-extension incompatibility).")
        print("[INFO] Automatically switching to Python 3.12 environment...")
        print("=" * 70)
        try:
            res = subprocess.run(["py", "-3.12", os.path.abspath(__file__)] + sys.argv[1:])
            sys.exit(res.returncode)
        except Exception:
            res = subprocess.run(["python", os.path.abspath(__file__)] + sys.argv[1:])
            sys.exit(res.returncode)

import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

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

# Import core fall detector and utilities
from fall_core import YOLOv8FallDetector
from utils.resource import (
    get_resource_path,
    get_model_path,
    check_weights_exist,
    play_alert_sound,
    save_fall_snapshot,
)

# Configure CustomTkinter Theme
ctk.set_appearance_mode("dark")  # "dark" or "light"
ctk.set_default_color_theme("blue")  # "blue", "dark-blue", "green"


class FallDetectionGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("AI Human Fall Detection System - YOLOv8 Pose")
        self.geometry("1280x820")
        self.minsize(1050, 700)

        # Set Window Icon if available
        icon_file = get_resource_path("app_icon.ico")
        if os.path.exists(icon_file):
            try:
                self.iconbitmap(icon_file)
            except Exception:
                pass

        # Threading & Stream State
        self.is_running = False
        self.is_paused = False
        self.cap = None
        self.worker_thread = None
        self.current_frame = None
        self.current_stats = None
        self.lock = threading.Lock()

        # Alert State
        self.fall_alert_active = False
        self.fall_alert_end_time = 0.0
        self.flash_state = False
        self.last_sound_time = 0.0

        # Metrics
        self.total_incidents = 0
        self.incident_history = []

        # Initialize Detector Engine
        self.detector = None
        self.init_detector()

        # Build UI Layout
        self._build_ui()

        # Handle Window Close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start GUI Update Loop
        self.after(20, self._gui_update_loop)

    def init_detector(self):
        """Initialize YOLOv8 Fall Detector."""
        try:
            model_file = get_model_path("yolov8n-pose.pt")
            check_weights_exist(model_file, raise_error=False)
            self.detector = YOLOv8FallDetector(
                model_path=model_file,
                window_size=15,
                fps=30,
                v_thresh=55.0,
                ar_thresh=0.35,
                dy_thresh=18.0,
                conf_thresh=0.35,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            messagebox.showerror(
                "Model Load Error",
                f"Failed to initialize YOLOv8-pose model:\n{e}",
            )

    def _build_ui(self):
        """Construct the full modern dark-themed GUI."""
        # Top-level grid configuration
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # -------------------------------------------------------------
        # 1. TOP HEADER & ALERT BANNER
        # -------------------------------------------------------------
        self.header_frame = ctk.CTkFrame(self, height=65, corner_radius=8)
        self.header_frame.grid(
            row=0, column=0, columnspan=2, padx=12, pady=(10, 6), sticky="ew"
        )
        self.header_frame.grid_columnconfigure(1, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text=" AI Human Fall Detection System",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.title_label.grid(row=0, column=0, padx=16, pady=12, sticky="w")

        # Dynamic Status Banner
        self.status_banner = ctk.CTkFrame(
            self.header_frame, height=42, corner_radius=6, fg_color="#1b4332"
        )
        self.status_banner.grid(
            row=0, column=1, padx=16, pady=8, sticky="ew"
        )

        self.status_text = ctk.CTkLabel(
            self.status_banner,
            text="🟢 STATUS: SYSTEM READY - STANDBY",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#d8f3dc",
        )
        self.status_text.pack(expand=True, fill="both", padx=10, pady=4)

        # -------------------------------------------------------------
        # 2. LEFT CONTROL SIDEBAR
        # -------------------------------------------------------------
        self.sidebar = ctk.CTkScrollableFrame(
            self, width=320, corner_radius=8, label_text="⚙️ System Controls"
        )
        self.sidebar.grid(
            row=1, column=0, padx=(12, 6), pady=6, sticky="nsew"
        )

        # --- Input Source Group ---
        self.src_label = ctk.CTkLabel(
            self.sidebar,
            text="🎥 Video Input Source",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.src_label.pack(anchor="w", padx=10, pady=(8, 4))

        self.source_mode = ctk.StringVar(value="webcam")
        self.source_segmented = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["Webcam", "Video File", "RTSP Stream"],
            command=self._on_source_mode_changed,
        )
        self.source_segmented.set("Webcam")
        self.source_segmented.pack(fill="x", padx=10, pady=4)

        # Container for dynamic source inputs
        self.source_container = ctk.CTkFrame(
            self.sidebar, fg_color="transparent"
        )
        self.source_container.pack(fill="x", padx=10, pady=4)

        # Sub-frame: Webcam
        self.webcam_frame = ctk.CTkFrame(
            self.source_container, fg_color="transparent"
        )
        self.cam_idx_label = ctk.CTkLabel(
            self.webcam_frame, text="Camera Device Index:"
        )
        self.cam_idx_label.pack(side="left", padx=4)
        self.cam_idx_combo = ctk.CTkComboBox(
            self.webcam_frame,
            values=["0 (Default)", "1", "2", "3"],
            width=110,
        )
        self.cam_idx_combo.set("0 (Default)")
        self.cam_idx_combo.pack(side="right", padx=4)
        self.webcam_frame.pack(fill="x", pady=2)

        # Sub-frame: File
        self.file_frame = ctk.CTkFrame(
            self.source_container, fg_color="transparent"
        )
        self.file_path_entry = ctk.CTkEntry(
            self.file_frame, placeholder_text="Select video file..."
        )
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=4)
        self.browse_btn = ctk.CTkButton(
            self.file_frame,
            text="Browse",
            width=70,
            command=self._browse_video_file,
        )
        self.browse_btn.pack(side="right", padx=4)

        # Sub-frame: RTSP
        self.rtsp_frame = ctk.CTkFrame(
            self.source_container, fg_color="transparent"
        )
        self.rtsp_entry = ctk.CTkEntry(
            self.rtsp_frame,
            placeholder_text="rtsp://admin:pass@192.168.1.100:554/live",
        )
        self.rtsp_entry.pack(fill="x", padx=4, pady=2)

        # --- Stream Control Buttons ---
        self.btn_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.btn_container.pack(fill="x", padx=10, pady=(12, 6))

        self.start_btn = ctk.CTkButton(
            self.btn_container,
            text="▶ Start Detection",
            fg_color="#2b9348",
            hover_color="#1b4332",
            font=ctk.CTkFont(weight="bold"),
            command=self.start_stream,
        )
        self.start_btn.pack(fill="x", pady=4)

        self.pause_btn = ctk.CTkButton(
            self.btn_container,
            text="⏸ Pause / Resume",
            fg_color="#3a86ff",
            hover_color="#265df2",
            state="disabled",
            command=self.toggle_pause,
        )
        self.pause_btn.pack(fill="x", pady=4)

        self.stop_btn = ctk.CTkButton(
            self.btn_container,
            text="⏹ Stop Stream",
            fg_color="#d90429",
            hover_color="#9a031e",
            state="disabled",
            command=self.stop_stream,
        )
        self.stop_btn.pack(fill="x", pady=4)

        self.snapshot_btn = ctk.CTkButton(
            self.btn_container,
            text="📸 Capture Snapshot",
            fg_color="#6c757d",
            hover_color="#495057",
            command=self.manual_snapshot,
        )
        self.snapshot_btn.pack(fill="x", pady=4)

        # --- Detection Parameters Group ---
        self.param_label = ctk.CTkLabel(
            self.sidebar,
            text="🎛️ Detection Thresholds",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.param_label.pack(anchor="w", padx=10, pady=(12, 4))

        # Velocity Threshold Slider
        self.v_label = ctk.CTkLabel(
            self.sidebar, text="Descent Velocity Thresh (v): 55 px/s"
        )
        self.v_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.v_slider = ctk.CTkSlider(
            self.sidebar,
            from_=15,
            to=120,
            number_of_steps=105,
            command=self._on_param_changed,
        )
        self.v_slider.set(55)
        self.v_slider.pack(fill="x", padx=10, pady=(0, 6))

        # Vertical Displacement Slider (dy)
        self.dy_label = ctk.CTkLabel(
            self.sidebar, text="Vertical Drop Thresh (Δy): 18 px"
        )
        self.dy_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.dy_slider = ctk.CTkSlider(
            self.sidebar,
            from_=5,
            to=60,
            number_of_steps=55,
            command=self._on_param_changed,
        )
        self.dy_slider.set(18)
        self.dy_slider.pack(fill="x", padx=10, pady=(0, 6))

        # Aspect Ratio Delta Slider
        self.ar_label = ctk.CTkLabel(
            self.sidebar, text="Aspect Ratio Delta (ΔAR): 0.35"
        )
        self.ar_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.ar_slider = ctk.CTkSlider(
            self.sidebar,
            from_=0.10,
            to=0.80,
            number_of_steps=70,
            command=self._on_param_changed,
        )
        self.ar_slider.set(0.35)
        self.ar_slider.pack(fill="x", padx=10, pady=(0, 6))

        # Confidence Slider
        self.conf_label = ctk.CTkLabel(
            self.sidebar, text="Pose Confidence Thresh: 0.35"
        )
        self.conf_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.conf_slider = ctk.CTkSlider(
            self.sidebar,
            from_=0.15,
            to=0.90,
            number_of_steps=75,
            command=self._on_param_changed,
        )
        self.conf_slider.set(0.35)
        self.conf_slider.pack(fill="x", padx=10, pady=(0, 6))

        # Temporal Window Size Slider
        self.win_label = ctk.CTkLabel(
            self.sidebar, text="Temporal Window: 15 frames"
        )
        self.win_label.pack(anchor="w", padx=10, pady=(2, 0))
        self.win_slider = ctk.CTkSlider(
            self.sidebar,
            from_=5,
            to=40,
            number_of_steps=35,
            command=self._on_param_changed,
        )
        self.win_slider.set(15)
        self.win_slider.pack(fill="x", padx=10, pady=(0, 6))

        # --- Alerts & Notifications Group ---
        self.alert_grp_label = ctk.CTkLabel(
            self.sidebar,
            text="🔔 Alert Preferences",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.alert_grp_label.pack(anchor="w", padx=10, pady=(12, 4))

        self.audio_alert_var = ctk.BooleanVar(value=True)
        self.audio_alert_cb = ctk.CTkCheckBox(
            self.sidebar,
            text="Enable Audio Alarm Beep",
            variable=self.audio_alert_var,
        )
        self.audio_alert_cb.pack(anchor="w", padx=10, pady=4)

        self.auto_snap_var = ctk.BooleanVar(value=True)
        self.auto_snap_cb = ctk.CTkCheckBox(
            self.sidebar,
            text="Auto-Save Fall Snapshots",
            variable=self.auto_snap_var,
        )
        self.auto_snap_cb.pack(anchor="w", padx=10, pady=4)

        # -------------------------------------------------------------
        # 3. RIGHT MAIN DISPLAY & LOGS
        # -------------------------------------------------------------
        self.main_frame = ctk.CTkFrame(self, corner_radius=8)
        self.main_frame.grid(
            row=1, column=1, padx=(6, 12), pady=6, sticky="nsew"
        )
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- KPI Metrics Bar ---
        self.kpi_frame = ctk.CTkFrame(
            self.main_frame, height=45, corner_radius=6
        )
        self.kpi_frame.grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="ew"
        )
        self.kpi_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.kpi_fps = ctk.CTkLabel(
            self.kpi_frame,
            text="⚡ FPS: 0.0",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.kpi_fps.grid(row=0, column=0, padx=8, pady=8)

        self.kpi_persons = ctk.CTkLabel(
            self.kpi_frame,
            text="👥 Persons Detected: 0",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.kpi_persons.grid(row=0, column=1, padx=8, pady=8)

        self.kpi_incidents = ctk.CTkLabel(
            self.kpi_frame,
            text="🚨 Total Incidents: 0",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#ff4d6d",
        )
        self.kpi_incidents.grid(row=0, column=2, padx=8, pady=8)

        self.kpi_device = ctk.CTkLabel(
            self.kpi_frame,
            text=f"💻 Engine: {self.detector.device.upper() if self.detector else 'CPU'}",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.kpi_device.grid(row=0, column=3, padx=8, pady=8)

        # --- Video Canvas Viewport ---
        self.video_canvas = tk.Canvas(
            self.main_frame, bg="#0d1117", highlightthickness=0
        )
        self.video_canvas.grid(
            row=1, column=0, padx=10, pady=4, sticky="nsew"
        )

        # Default placeholder image on canvas
        self._show_placeholder_canvas("Camera Feed Standby\nClick '▶ Start Detection' to begin")

        # --- Incident Event Log Drawer ---
        self.log_frame = ctk.CTkFrame(
            self.main_frame, height=140, corner_radius=6
        )
        self.log_frame.grid(
            row=2, column=0, padx=10, pady=(4, 10), sticky="ew"
        )
        self.log_frame.grid_columnconfigure(0, weight=1)

        self.log_header = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        self.log_header.pack(fill="x", padx=8, pady=(4, 2))

        self.log_title = ctk.CTkLabel(
            self.log_header,
            text="📋 Real-Time Incident Audit Log",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.log_title.pack(side="left")

        self.clear_log_btn = ctk.CTkButton(
            self.log_header,
            text="Clear",
            width=50,
            height=24,
            command=self.clear_logs,
        )
        self.clear_log_btn.pack(side="right", padx=4)

        self.log_textbox = ctk.CTkTextbox(
            self.log_frame, height=90, font=ctk.CTkFont(family="Consolas", size=11)
        )
        self.log_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        self.log_textbox.insert(
            "end",
            f"[{time.strftime('%H:%M:%S')}] System initialized. YOLOv8 Pose engine ready.\n",
        )

    # -------------------------------------------------------------
    # Dynamic UI Callbacks & Parameter Updates
    # -------------------------------------------------------------
    def _on_source_mode_changed(self, mode: str):
        """Toggle active input controls based on chosen source mode."""
        self.webcam_frame.pack_forget()
        self.file_frame.pack_forget()
        self.rtsp_frame.pack_forget()

        if mode == "Webcam":
            self.webcam_frame.pack(fill="x", pady=2)
        elif mode == "Video File":
            self.file_frame.pack(fill="x", pady=2)
        elif mode == "RTSP Stream":
            self.rtsp_frame.pack(fill="x", pady=2)

    def _browse_video_file(self):
        """Open file dialog to pick a video."""
        filetypes = [
            ("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("All Files", "*.*"),
        ]
        filename = filedialog.askopenfilename(
            title="Select Video File", filetypes=filetypes
        )
        if filename:
            self.file_path_entry.delete(0, "end")
            self.file_path_entry.insert(0, filename)

    def _on_param_changed(self, _=None):
        """Sync slider values to detector thresholds."""
        v = round(self.v_slider.get(), 1)
        dy = round(self.dy_slider.get(), 1)
        ar = round(self.ar_slider.get(), 2)
        conf = round(self.conf_slider.get(), 2)
        win = int(self.win_slider.get())

        self.v_label.configure(text=f"Descent Velocity Thresh (v): {v} px/s")
        self.dy_label.configure(text=f"Vertical Drop Thresh (Δy): {dy} px")
        self.ar_label.configure(text=f"Aspect Ratio Delta (ΔAR): {ar:.2f}")
        self.conf_label.configure(text=f"Pose Confidence Thresh: {conf:.2f}")
        self.win_label.configure(text=f"Temporal Window: {win} frames")

        if self.detector:
            self.detector.set_parameters(
                v_thresh=v,
                dy_thresh=dy,
                ar_thresh=ar,
                conf_thresh=conf,
                window_size=win,
            )

    def _log_event(self, text: str):
        """Append timestamped message to the log box."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_textbox.insert("end", f"[{timestamp}] {text}\n")
        self.log_textbox.see("end")

    def clear_logs(self):
        self.log_textbox.delete("1.0", "end")

    def _show_placeholder_canvas(self, message: str):
        """Render a placeholder state on video canvas."""
        self.video_canvas.delete("all")
        w = max(100, self.video_canvas.winfo_width())
        h = max(100, self.video_canvas.winfo_height())
        self.video_canvas.create_text(
            w // 2,
            h // 2,
            text=message,
            fill="#6c757d",
            font=("Segoe UI", 16, "bold"),
            justify="center",
        )

    # -------------------------------------------------------------
    # Stream Lifecycle Management
    # -------------------------------------------------------------
    def start_stream(self):
        """Start the video capture and inference worker thread."""
        if self.is_running:
            return

        mode = self.source_segmented.get()
        source_target = None

        if mode == "Webcam":
            val = self.cam_idx_combo.get().split()[0]
            try:
                source_target = int(val)
            except ValueError:
                source_target = 0
        elif mode == "Video File":
            source_target = self.file_path_entry.get().strip()
            if not source_target or not os.path.exists(source_target):
                messagebox.showwarning(
                    "Invalid File", "Please select a valid video file path."
                )
                return
        elif mode == "RTSP Stream":
            source_target = self.rtsp_entry.get().strip()
            if not source_target:
                messagebox.showwarning(
                    "Invalid URL", "Please enter a valid RTSP or HTTP camera URL."
                )
                return

        self._log_event(f"Connecting to source: {mode} ({source_target})...")

        # Open Video Capture
        self.cap = cv2.VideoCapture(source_target)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Capture Error",
                f"Failed to open video source:\n{source_target}",
            )
            self._log_event("❌ Failed to open video source.")
            return

        self.is_running = True
        self.is_paused = False

        # Update Button States
        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="⏸ Pause")
        self.stop_btn.configure(state="normal")

        # Update Status Banner
        self.status_banner.configure(fg_color="#1b4332")
        self.status_text.configure(
            text="🟢 STATUS: NORMAL - MONITORING ACTIVE", text_color="#d8f3dc"
        )

        # Launch Worker Thread
        self.worker_thread = threading.Thread(
            target=self._worker_capture_and_infer, daemon=True
        )
        self.worker_thread.start()
        self._log_event("▶ Detection stream started successfully.")

    def toggle_pause(self):
        """Pause or resume the stream."""
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="▶ Resume")
            self._log_event("⏸ Stream paused.")
        else:
            self.pause_btn.configure(text="⏸ Pause")
            self._log_event("▶ Stream resumed.")

    def stop_stream(self):
        """Stop stream and clean up resources."""
        self.is_running = False
        self.is_paused = False

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.start_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="⏸ Pause")
        self.stop_btn.configure(state="disabled")

        self.status_banner.configure(fg_color="#1b4332")
        self.status_text.configure(
            text="🟢 STATUS: SYSTEM READY - STANDBY", text_color="#d8f3dc"
        )
        self.kpi_fps.configure(text="⚡ FPS: 0.0")
        self.kpi_persons.configure(text="👥 Persons Detected: 0")

        self._show_placeholder_canvas("Stream Stopped.\nClick '▶ Start Detection' to restart")
        self._log_event("⏹ Stream stopped.")

    def manual_snapshot(self):
        """Take a snapshot of the current view."""
        with self.lock:
            if self.current_frame is not None:
                path = save_fall_snapshot(
                    self.current_frame, prefix="manual_snap"
                )
                if path:
                    self._log_event(f"📸 Snapshot saved: {os.path.basename(path)}")
                    messagebox.showinfo(
                        "Snapshot Saved", f"Image saved to:\n{path}"
                    )
            else:
                messagebox.showwarning(
                    "No Feed", "No active video stream to capture."
                )

    # -------------------------------------------------------------
    # Background Worker Thread (Capture + Inference)
    # -------------------------------------------------------------
    def _worker_capture_and_infer(self):
        """Dedicated background thread for OpenCV capture and YOLOv8 inference."""
        prev_time = time.time()

        while self.is_running and self.cap and self.cap.isOpened():
            if self.is_paused:
                time.sleep(0.05)
                continue

            success, frame = self.cap.read()
            if not success:
                # Video file reached end or stream disconnected
                self._log_event("End of video stream reached.")
                break

            # Run YOLOv8 Pose + Fall Detection Engine
            try:
                annotated_frame, stats = self.detector.process_frame(
                    frame, prev_time=prev_time
                )
                prev_time = stats["new_time"]

                with self.lock:
                    self.current_frame = annotated_frame
                    self.current_stats = stats

                # Handle Fall Alert Triggers
                if stats["has_fall"]:
                    self._handle_fall_trigger(stats["fall_events"], frame)

            except Exception as e:
                print(f"[ERROR] Inference step error: {e}")
                time.sleep(0.01)

        # When loop terminates
        self.after(0, self.stop_stream)

    def _handle_fall_trigger(self, fall_events: list, raw_frame: np.ndarray):
        """Handle detected fall events: activate visual banner, sound, auto-snapshot."""
        self.fall_alert_active = True
        self.fall_alert_end_time = time.time() + 2.5  # hold alert for 2.5s

        # Sound Alert
        now = time.time()
        if self.audio_alert_var.get() and (now - self.last_sound_time > 1.2):
            self.last_sound_time = now
            play_alert_sound(freq=1300, duration_ms=400)

        # Auto-Save Snapshot
        if self.auto_snap_var.get():
            saved_path = save_fall_snapshot(
                raw_frame, prefix="fall_alert"
            )
            if saved_path:
                for event in fall_events:
                    self._log_event(
                        f"🚨 FALL DETECTED! ID: #{event['tracker_id']} [{event['tag']}] -> Saved: {os.path.basename(saved_path)}"
                    )
        else:
            for event in fall_events:
                self._log_event(
                    f"🚨 FALL DETECTED! ID: #{event['tracker_id']} [{event['tag']}]"
                )

    # -------------------------------------------------------------
    # GUI Main-Thread Render Loop
    # -------------------------------------------------------------
    def _gui_update_loop(self):
        """Periodic UI update running on the main Tkinter thread."""
        try:
            with self.lock:
                frame = (
                    self.current_frame.copy()
                    if self.current_frame is not None
                    else None
                )
                stats = self.current_stats

            # 1. Update Video Canvas
            if frame is not None:
                self._render_frame_to_canvas(frame)

            # 2. Update KPI Metrics & Banner
            if stats is not None and self.is_running:
                self.kpi_fps.configure(text=f"⚡ FPS: {stats['fps']:.1f}")
                self.kpi_persons.configure(
                    text=f"👥 Persons Detected: {stats['active_persons']}"
                )
                self.kpi_incidents.configure(
                    text=f"🚨 Incidents: {self.detector.total_fall_incidents}"
                )

            # 3. Dynamic Alert Banner Flashing
            now = time.time()
            if self.fall_alert_active and now < self.fall_alert_end_time:
                self.flash_state = not self.flash_state
                banner_color = "#d90429" if self.flash_state else "#7f1d1d"
                self.status_banner.configure(fg_color=banner_color)
                self.status_text.configure(
                    text="🚨 CRITICAL ALERT: HUMAN FALL EVENT CONFIRMED!",
                    text_color="#ffffff",
                )
            elif self.is_running:
                self.fall_alert_active = False
                self.status_banner.configure(fg_color="#1b4332")
                self.status_text.configure(
                    text="🟢 STATUS: NORMAL - MONITORING ACTIVE",
                    text_color="#d8f3dc",
                )

        except Exception as e:
            pass

        # Schedule next loop iteration (approx 30-40 FPS UI refresh)
        self.after(25, self._gui_update_loop)

    def _render_frame_to_canvas(self, cv_frame: np.ndarray):
        """Letterbox and draw the OpenCV BGR frame onto Tkinter Canvas."""
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()

        if canvas_w < 10 or canvas_h < 10:
            return

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb_frame.shape[:2]

        # Calculate Aspect Ratio Fit
        scale = min(canvas_w / orig_w, canvas_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        resized = cv2.resize(
            rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        pil_img = Image.fromarray(resized)
        self.tk_image = ImageTk.PhotoImage(image=pil_img)

        # Center on canvas
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2

        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            x, y, anchor="nw", image=self.tk_image
        )

    def on_closing(self):
        """Gracefully release threads and close application."""
        self.stop_stream()
        self.destroy()


def main():
    app = FallDetectionGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
