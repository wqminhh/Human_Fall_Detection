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

APP_NAME = "AI Human Fall Detection System"
APP_VERSION = "1.2.0"
APP_EXE_NAME = "FallDetectionApp.exe"

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

# ── Palette ────────────────────────────────────────────────────────────────────
CLR_BG        = "#111318"   # root background
CLR_CARD      = "#1a1d24"   # card / panel background
CLR_CARD2     = "#20242e"   # slightly lighter card
CLR_BORDER    = "#2a2f3d"   # subtle border / divider
CLR_ACCENT    = "#4f8ef7"   # primary accent (blue)
CLR_ACCENT2   = "#364fc7"   # accent hover
CLR_GREEN     = "#2d9e5f"   # running / normal state
CLR_GREEN_BG  = "#0d2a1c"
CLR_RED       = "#e03131"   # danger / fall alert
CLR_RED_BG    = "#2a0d0d"
CLR_AMBER     = "#f59f00"   # pause state
CLR_MUTED     = "#6c7693"   # secondary text
CLR_TEXT      = "#dde3f0"   # primary text

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _section_label(parent, text: str, **kwargs) -> ctk.CTkLabel:
    """Uniform section header label."""
    return ctk.CTkLabel(
        parent,
        text=text,
        font=ctk.CTkFont(size=11, weight="bold"),
        text_color=CLR_MUTED,
        **kwargs,
    )


def _divider(parent) -> ctk.CTkFrame:
    """Thin horizontal rule."""
    return ctk.CTkFrame(parent, height=1, fg_color=CLR_BORDER, corner_radius=0)


class FallDetectionGUI(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(f"{APP_NAME} {APP_VERSION}  ·  YOLOv8 Pose")
        self.geometry("1300x840")
        self.minsize(1050, 700)
        self.configure(fg_color=CLR_BG)

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
        self.video_writer = None          # cv2.VideoWriter for session recording
        self.video_writer_path = ""       # path of the currently recording file
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
        """Construct the clean, card-based dark GUI."""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._build_header()
        self._build_sidebar()
        self._build_main_area()

    # ── Header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = ctk.CTkFrame(self, height=58, corner_radius=10, fg_color=CLR_CARD)
        hdr.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 6), sticky="ew")
        hdr.grid_columnconfigure(1, weight=1)
        hdr.grid_propagate(False)

        ctk.CTkLabel(
            hdr,
            text=f"  ◈  {APP_NAME}",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=CLR_TEXT,
        ).grid(row=0, column=0, padx=18, pady=0, sticky="w")

        ctk.CTkLabel(
            hdr,
            text="Powered by YOLOv8-Pose",
            font=ctk.CTkFont(size=11),
            text_color=CLR_MUTED,
        ).grid(row=0, column=1, padx=0, pady=0, sticky="w")

        # Status pill (compact)
        self.status_pill = ctk.CTkFrame(hdr, corner_radius=20, fg_color=CLR_GREEN_BG)
        self.status_pill.grid(row=0, column=2, padx=16, pady=10, sticky="e")

        self.status_dot = ctk.CTkLabel(
            self.status_pill, text="●", font=ctk.CTkFont(size=13),
            text_color=CLR_GREEN, width=20,
        )
        self.status_dot.pack(side="left", padx=(10, 2), pady=6)

        self.status_text = ctk.CTkLabel(
            self.status_pill,
            text="Ready",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=CLR_GREEN,
        )
        self.status_text.pack(side="left", padx=(0, 14), pady=6)

    # ── Left Sidebar ────────────────────────────────────────────────────────
    def _build_sidebar(self):
        self.sidebar = ctk.CTkScrollableFrame(
            self, width=300, corner_radius=10,
            fg_color=CLR_CARD,
            scrollbar_button_color=CLR_BORDER,
            scrollbar_button_hover_color=CLR_ACCENT,
            label_text="",
        )
        self.sidebar.grid(row=1, column=0, padx=(14, 6), pady=(0, 14), sticky="nsew")

        # ── Input Source
        _section_label(self.sidebar, "VIDEO INPUT SOURCE").pack(
            anchor="w", padx=14, pady=(16, 6)
        )
        self.source_segmented = ctk.CTkSegmentedButton(
            self.sidebar,
            values=["Webcam", "Video File", "RTSP"],
            command=self._on_source_mode_changed,
            font=ctk.CTkFont(size=12),
            selected_color=CLR_ACCENT,
            selected_hover_color=CLR_ACCENT2,
            unselected_color=CLR_CARD2,
            unselected_hover_color=CLR_BORDER,
        )
        self.source_segmented.set("Webcam")
        self.source_segmented.pack(fill="x", padx=14, pady=(0, 8))

        self.source_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.source_container.pack(fill="x", padx=14, pady=(0, 4))

        # Webcam sub-frame
        self.webcam_frame = ctk.CTkFrame(self.source_container, fg_color=CLR_CARD2, corner_radius=8)
        cam_row = ctk.CTkFrame(self.webcam_frame, fg_color="transparent")
        cam_row.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(cam_row, text="Camera Index", font=ctk.CTkFont(size=12), text_color=CLR_MUTED).pack(side="left")
        self.cam_idx_combo = ctk.CTkComboBox(
            cam_row, values=["0 (Default)", "1", "2", "3"], width=120,
            fg_color=CLR_CARD, border_color=CLR_BORDER, button_color=CLR_ACCENT,
        )
        self.cam_idx_combo.set("0 (Default)")
        self.cam_idx_combo.pack(side="right")
        self.webcam_frame.pack(fill="x", pady=2)

        # File sub-frame
        self.file_frame = ctk.CTkFrame(self.source_container, fg_color=CLR_CARD2, corner_radius=8)
        file_row = ctk.CTkFrame(self.file_frame, fg_color="transparent")
        file_row.pack(fill="x", padx=10, pady=8)
        self.file_path_entry = ctk.CTkEntry(
            file_row, placeholder_text="Select video file...",
            fg_color=CLR_CARD, border_color=CLR_BORDER,
        )
        self.file_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.browse_btn = ctk.CTkButton(
            file_row, text="Browse", width=72, height=28,
            fg_color=CLR_ACCENT, hover_color=CLR_ACCENT2,
            font=ctk.CTkFont(size=12), command=self._browse_video_file,
        )
        self.browse_btn.pack(side="right")

        # RTSP sub-frame
        self.rtsp_frame = ctk.CTkFrame(self.source_container, fg_color=CLR_CARD2, corner_radius=8)
        self.rtsp_entry = ctk.CTkEntry(
            self.rtsp_frame,
            placeholder_text="rtsp://[IP_ADDRESS]/live",
            fg_color=CLR_CARD, border_color=CLR_BORDER,
        )
        self.rtsp_entry.pack(fill="x", padx=10, pady=8)

        # ── Stream Controls
        _divider(self.sidebar).pack(fill="x", padx=14, pady=(14, 0))
        _section_label(self.sidebar, "STREAM CONTROLS").pack(anchor="w", padx=14, pady=(10, 8))

        btn_cfg = dict(font=ctk.CTkFont(size=13, weight="bold"), height=36, corner_radius=8)

        self.start_btn = ctk.CTkButton(
            self.sidebar, text="▶  Start Detection",
            fg_color=CLR_GREEN, hover_color="#1d7a49",
            command=self.start_stream, **btn_cfg,
        )
        self.start_btn.pack(fill="x", padx=14, pady=(0, 6))

        self.pause_btn = ctk.CTkButton(
            self.sidebar, text="⏸  Pause",
            fg_color=CLR_AMBER, hover_color="#b07800",
            state="disabled", command=self.toggle_pause, **btn_cfg,
        )
        self.pause_btn.pack(fill="x", padx=14, pady=(0, 6))

        self.stop_btn = ctk.CTkButton(
            self.sidebar, text="⏹  Stop",
            fg_color=CLR_RED, hover_color="#a82323",
            state="disabled", command=self.stop_stream, **btn_cfg,
        )
        self.stop_btn.pack(fill="x", padx=14, pady=(0, 6))

        self.snapshot_btn = ctk.CTkButton(
            self.sidebar, text="📸  Capture Snapshot",
            fg_color=CLR_CARD2, hover_color=CLR_BORDER, text_color=CLR_TEXT,
            command=self.manual_snapshot, **btn_cfg,
        )
        self.snapshot_btn.pack(fill="x", padx=14, pady=(0, 4))

        # ── Detection Thresholds
        _divider(self.sidebar).pack(fill="x", padx=14, pady=(14, 0))
        _section_label(self.sidebar, "DETECTION THRESHOLDS").pack(anchor="w", padx=14, pady=(10, 4))

        self._add_slider(
            "Descent Velocity (v)", "px/s",
            "v_slider", "v_label", from_=15, to=120, steps=105, init=55,
        )
        self._add_slider(
            "Vertical Drop (Δy)", "px",
            "dy_slider", "dy_label", from_=5, to=60, steps=55, init=18,
        )
        self._add_slider(
            "Aspect Ratio (ΔAR)", "",
            "ar_slider", "ar_label", from_=0.10, to=0.80, steps=70, init=0.35, fmt="{:.2f}",
        )
        self._add_slider(
            "Pose Confidence", "",
            "conf_slider", "conf_label", from_=0.15, to=0.90, steps=75, init=0.35, fmt="{:.2f}",
        )
        self._add_slider(
            "Temporal Window", "frames",
            "win_slider", "win_label", from_=5, to=40, steps=35, init=15, fmt="{:.0f}",
        )

        # ── Alert Preferences
        _divider(self.sidebar).pack(fill="x", padx=14, pady=(14, 0))
        _section_label(self.sidebar, "ALERT PREFERENCES").pack(anchor="w", padx=14, pady=(10, 6))

        self.audio_alert_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self.sidebar, text="Audio Alarm on Fall",
            variable=self.audio_alert_var,
            checkbox_width=18, checkbox_height=18,
            fg_color=CLR_ACCENT, hover_color=CLR_ACCENT2,
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=14, pady=(0, 6))

        self.auto_snap_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self.sidebar, text="Auto-Save Fall Snapshots",
            variable=self.auto_snap_var,
            checkbox_width=18, checkbox_height=18,
            fg_color=CLR_ACCENT, hover_color=CLR_ACCENT2,
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=14, pady=(0, 6))

        # ── Snapshot save folder
        snap_row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        snap_row.pack(fill="x", padx=14, pady=(4, 14))
        ctk.CTkLabel(
            snap_row, text="Save folder",
            font=ctk.CTkFont(size=12), text_color=CLR_MUTED,
        ).pack(side="left")
        ctk.CTkButton(
            snap_row, text="Browse", width=60, height=24,
            fg_color=CLR_BORDER, hover_color=CLR_CARD2, text_color=CLR_TEXT,
            font=ctk.CTkFont(size=11), corner_radius=6,
            command=self._browse_snapshot_dir,
        ).pack(side="right", padx=(6, 0))

        snap_entry_row = ctk.CTkFrame(self.sidebar, fg_color=CLR_CARD2, corner_radius=8)
        snap_entry_row.pack(fill="x", padx=14, pady=(0, 14))
        _default_snap_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "fall_snapshots"
        )
        self.snap_dir_entry = ctk.CTkEntry(
            snap_entry_row,
            placeholder_text=_default_snap_dir,
            fg_color="transparent", border_width=0,
            font=ctk.CTkFont(size=11), text_color=CLR_TEXT,
        )
        self.snap_dir_entry.insert(0, _default_snap_dir)
        self.snap_dir_entry.pack(fill="x", padx=8, pady=6)

    def _add_slider(self, label: str, unit: str, slider_attr: str, lbl_attr: str,
                    from_, to, steps, init, fmt="{:.1f}"):
        """Compact threshold row: label left, small entry right (no big slider)."""
        if not hasattr(self, "_thresh_grid"):
            # Create the grid container once on first call
            self._thresh_grid = ctk.CTkFrame(self.sidebar, fg_color=CLR_CARD2, corner_radius=8)
            self._thresh_grid.pack(fill="x", padx=14, pady=(0, 4))
            self._thresh_row_idx = 0
            self._thresh_grid.grid_columnconfigure(1, weight=1)
            self._thresh_grid.grid_columnconfigure(3, weight=1)

        r = self._thresh_row_idx // 2  # grid row
        c = (self._thresh_row_idx % 2) * 2  # col 0 or 2

        ctk.CTkLabel(
            self._thresh_grid, text=label,
            font=ctk.CTkFont(size=11), text_color=CLR_MUTED, anchor="w",
        ).grid(row=r * 2, column=c, columnspan=2, sticky="w", padx=(10, 4), pady=(6, 0))

        entry = ctk.CTkEntry(
            self._thresh_grid,
            width=80, height=26,
            fg_color=CLR_CARD, border_color=CLR_BORDER,
            font=ctk.CTkFont(size=12, weight="bold"),
            justify="center",
        )
        entry.insert(0, fmt.format(init))
        entry.grid(row=r * 2 + 1, column=c, columnspan=2, padx=(10, 4), pady=(2, 8), sticky="ew")

        # bind on focus-out to apply changes
        entry.bind("<FocusOut>", lambda e: self._on_param_changed())
        entry.bind("<Return>",   lambda e: self._on_param_changed())

        setattr(self, lbl_attr, entry)   # reuse lbl_attr to store the entry widget
        setattr(self, slider_attr, entry)  # slider_attr also points to same entry
        self._thresh_row_idx += 1

    # ── Main Content Area ────────────────────────────────────────────────────
    def _build_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color=CLR_CARD)
        self.main_frame.grid(row=1, column=1, padx=(0, 14), pady=(0, 14), sticky="nsew")
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # ── KPI bar
        kpi = ctk.CTkFrame(self.main_frame, height=52, corner_radius=8, fg_color=CLR_CARD2)
        kpi.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        kpi.grid_columnconfigure((0, 1, 2, 3), weight=1)
        kpi.grid_propagate(False)

        def _kpi_cell(col, icon, label, color=CLR_TEXT):
            f = ctk.CTkFrame(kpi, fg_color="transparent")
            f.grid(row=0, column=col, padx=8, pady=6, sticky="nsew")
            ctk.CTkLabel(f, text=icon, font=ctk.CTkFont(size=16)).pack(side="left", padx=(4, 4))
            lbl = ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color=color)
            lbl.pack(side="left")
            return lbl

        self.kpi_fps      = _kpi_cell(0, "⚡", "FPS: 0.0")
        self.kpi_persons  = _kpi_cell(1, "👤", "Persons: 0")
        self.kpi_incidents = _kpi_cell(2, "🚨", "Incidents: 0", color="#ff6b6b")
        device_str = self.detector.device.upper() if self.detector else "CPU"
        self.kpi_device   = _kpi_cell(3, "💻", f"Engine: {device_str}", color=CLR_MUTED)

        # ── Video canvas
        self.video_canvas = tk.Canvas(
            self.main_frame, bg="#0a0c12", highlightthickness=0,
        )
        self.video_canvas.grid(row=1, column=0, padx=12, pady=(0, 6), sticky="nsew")
        self._show_placeholder_canvas("No feed  ·  Click ▶ Start Detection")

        # ── Incident Log card
        log_card = ctk.CTkFrame(self.main_frame, corner_radius=8, fg_color=CLR_CARD2, height=130)
        log_card.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_propagate(False)

        log_hdr = ctk.CTkFrame(log_card, fg_color="transparent")
        log_hdr.grid(row=0, column=0, padx=10, pady=(6, 2), sticky="ew")

        ctk.CTkLabel(
            log_hdr, text="Incident Log",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=CLR_TEXT,
        ).pack(side="left")

        ctk.CTkButton(
            log_hdr, text="Clear", width=54, height=24,
            fg_color=CLR_BORDER, hover_color=CLR_CARD, text_color=CLR_MUTED,
            font=ctk.CTkFont(size=11), corner_radius=6,
            command=self.clear_logs,
        ).pack(side="right")

        self.log_textbox = ctk.CTkTextbox(
            log_card, height=78,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color="transparent", text_color=CLR_MUTED,
            border_width=0,
        )
        self.log_textbox.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")
        self.log_textbox.insert(
            "end",
            f"[{time.strftime('%H:%M:%S')}] System initialized — YOLOv8 Pose engine ready.\n",
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
        elif mode == "RTSP":
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

    def _browse_snapshot_dir(self):
        """Open directory dialog to pick the auto-save folder."""
        folder = filedialog.askdirectory(title="Select Snapshot Save Folder")
        if folder:
            self.snap_dir_entry.delete(0, "end")
            self.snap_dir_entry.insert(0, folder)

    def _get_snapshot_dir(self) -> str | None:
        """Return user-chosen snapshot folder, or None to use the default."""
        val = self.snap_dir_entry.get().strip()
        return val if val else None

    def _on_param_changed(self, _=None):
        """Read entry values and sync to detector thresholds."""
        def _safe(entry, default, cast=float):
            try:
                return cast(entry.get())
            except (ValueError, AttributeError):
                return default

        v    = _safe(self.v_slider,    55.0)
        dy   = _safe(self.dy_slider,   18.0)
        ar   = _safe(self.ar_slider,   0.35)
        conf = _safe(self.conf_slider, 0.35)
        win  = _safe(self.win_slider,  15,   cast=int)

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
            w // 2, h // 2 - 14,
            text="◈",
            fill=CLR_BORDER,
            font=("Segoe UI", 36),
            justify="center",
        )
        self.video_canvas.create_text(
            w // 2, h // 2 + 28,
            text=message,
            fill=CLR_MUTED,
            font=("Segoe UI", 13),
            justify="center",
        )

    # ── Status Pill Helpers ──────────────────────────────────────────────────
    def _set_status(self, text: str, dot_color: str, bg_color: str, text_color: str):
        self.status_pill.configure(fg_color=bg_color)
        self.status_dot.configure(text_color=dot_color)
        self.status_text.configure(text=text, text_color=text_color)

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
        self.is_paused  = False

        # ── Start session video recorder ─────────────────────────────
        try:
            rec_dir = self._get_snapshot_dir() or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "fall_snapshots"
            )
            os.makedirs(rec_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.video_writer_path = os.path.join(rec_dir, f"session_{ts}.mp4")

            src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
            src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_writer_path, fourcc, src_fps, (src_w, src_h)
            )
            if not self.video_writer.isOpened():
                self.video_writer = None
                self.video_writer_path = ""
                self._log_event("⚠️  Could not open VideoWriter — recording disabled.")
            else:
                self._log_event(f"🎥  Recording session → {os.path.basename(self.video_writer_path)}")
        except Exception as _e:
            self.video_writer = None
            self.video_writer_path = ""
            print(f"[WARN] VideoWriter init failed: {_e}")
        # ─────────────────────────────────────────────────────────────

        self.start_btn.configure(state="disabled")
        self.pause_btn.configure(state="normal", text="⏸  Pause")
        self.stop_btn.configure(state="normal")

        self._set_status("Monitoring", CLR_GREEN, CLR_GREEN_BG, CLR_GREEN)

        self.worker_thread = threading.Thread(
            target=self._worker_capture_and_infer, daemon=True
        )
        self.worker_thread.start()
        self._log_event("▶  Detection stream started.")

    def toggle_pause(self):
        """Pause or resume the stream."""
        if not self.is_running:
            return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.configure(text="▶  Resume")
            self._set_status("Paused", CLR_AMBER, "#2a2000", CLR_AMBER)
            self._log_event("⏸  Stream paused.")
        else:
            self.pause_btn.configure(text="⏸  Pause")
            self._set_status("Monitoring", CLR_GREEN, CLR_GREEN_BG, CLR_GREEN)
            self._log_event("▶  Stream resumed.")

    def stop_stream(self):
        """Stop stream, flush video recording, and clean up resources."""
        self.is_running = False
        self.is_paused  = False

        # Clear the last frame so canvas goes blank, not frozen
        with self.lock:
            self.current_frame = None
            self.current_stats = None

        # ── Finalize video recording ──────────────────────────────────
        if self.video_writer:
            try:
                self.video_writer.release()
                self._log_event(
                    f"💾  Session saved → {os.path.basename(self.video_writer_path)}"
                )
            except Exception:
                pass
            self.video_writer = None
            self.video_writer_path = ""
        # ─────────────────────────────────────────────────────────────

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.start_btn.configure(state="normal")
        self.pause_btn.configure(state="disabled", text="⏸  Pause")
        self.stop_btn.configure(state="disabled")

        self._set_status("Ready", CLR_GREEN, CLR_GREEN_BG, CLR_GREEN)
        self.kpi_fps.configure(text="FPS: 0.0")
        self.kpi_persons.configure(text="Persons: 0")

        self._show_placeholder_canvas("Stream stopped  ·  Click ▶ Start Detection")
        self._log_event("⏹  Stream stopped.")

    def manual_snapshot(self):
        """Take a snapshot of the current view."""
        with self.lock:
            if self.current_frame is not None:
                path = save_fall_snapshot(
                    self.current_frame, prefix="manual_snap",
                    save_dir=self._get_snapshot_dir(),
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

                # Write annotated frame to session video
                if self.video_writer:
                    try:
                        self.video_writer.write(annotated_frame)
                    except Exception:
                        pass

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
                raw_frame, prefix="fall_alert",
                save_dir=self._get_snapshot_dir(),
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

            # 2. Update KPI Metrics
            if stats is not None and self.is_running:
                self.kpi_fps.configure(text=f"FPS: {stats['fps']:.1f}")
                self.kpi_persons.configure(text=f"Persons: {stats['active_persons']}")
                self.kpi_incidents.configure(
                    text=f"Incidents: {self.detector.total_fall_incidents}"
                )

            # 3. Alert status pill flashing
            now = time.time()
            if self.fall_alert_active and now < self.fall_alert_end_time:
                self.flash_state = not self.flash_state
                alert_bg  = CLR_RED_BG if self.flash_state else "#1a0505"
                dot_color = CLR_RED    if self.flash_state else "#a01c1c"
                self.status_pill.configure(fg_color=alert_bg)
                self.status_dot.configure(text_color=dot_color)
                self.status_text.configure(text="FALL DETECTED!", text_color=CLR_RED)
            elif self.is_running and not self.is_paused:
                self.fall_alert_active = False
                self._set_status("Monitoring", CLR_GREEN, CLR_GREEN_BG, CLR_GREEN)

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
