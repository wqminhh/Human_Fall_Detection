"""
Resource and Path Management Utilities for Human Fall Detection System.

Handles dynamic path resolution for PyInstaller bundled executables (_MEIPASS)
and local development environments, ensuring seamless access to model weights,
configuration files, and UI assets.

Usage Example:
    from utils import get_resource_path, get_model_path, check_weights_exist

    # 1. Get path to a model file inside the weights directory:
    model_path = get_model_path("yolov8n-pose.pt")

    # 2. Check if weights are present before loading:
    if not check_weights_exist("yolov8n-pose.pt"):
        raise FileNotFoundError(f"Model file missing: {model_path}")

    # 3. Resolve any general asset (e.g., icon, config):
    icon_path = get_resource_path("app_icon.ico")
"""

import os
import sys
import threading
from pathlib import Path
from typing import Union, List, Optional


def get_resource_path(relative_path: Union[str, Path]) -> str:
    """
    Get absolute path to a resource, supporting both local development and PyInstaller bundles.

    When packaged with PyInstaller (--onefile or --onedir mode), assets bundled
    via --add-data are extracted to or located in sys._MEIPASS (or sys._MEIPASS2).
    During local execution, paths are resolved relative to the repository project root.

    Args:
        relative_path (str | Path): Relative path to resource (e.g., 'weights/yolov8n-pose.pt', 'app_icon.ico')

    Returns:
        str: Normalized absolute path to the requested resource.
    """
    rel_str = str(relative_path).lstrip("/\\")

    try:
        # Check if running in a PyInstaller bundle
        if hasattr(sys, "_MEIPASS"):
            base_dir = Path(getattr(sys, "_MEIPASS"))
        elif getattr(sys, "frozen", False):
            base_dir = Path(sys.executable).parent
        else:
            # Running as standard Python script
            current_file = Path(__file__).resolve()
            # If this file is inside a 'utils' subfolder, go one level up to project root
            if current_file.parent.name == "utils":
                base_dir = current_file.parent.parent
            else:
                base_dir = current_file.parent
    except Exception:
        base_dir = Path.cwd()

    # 1. Primary candidate: inside base_dir (bundle directory or project root)
    candidate_path = (base_dir / rel_str).resolve()
    if candidate_path.exists():
        return str(candidate_path)

    # 2. Secondary candidate: inside current working directory
    cwd_candidate = (Path.cwd() / rel_str).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    # 3. Fallback: return the resolved base candidate even if not yet created
    return str(candidate_path)


def get_model_path(model_name: str = "yolov8n-pose.pt") -> str:
    """
    Specifically resolve and return the absolute path to a PyTorch model weight file.

    Checks:
    1. weights/<model_name> inside bundle or project root
    2. <model_name> at the root level
    3. Custom absolute path if provided

    Args:
        model_name (str): Model filename (e.g. 'yolov8n-pose.pt', 'yolov7-w6-pose.pt', 'fall_classifier.pth')

    Returns:
        str: Absolute path to the model weight file.
    """
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name

    # Check weights/ subdirectory first
    weights_subpath = os.path.join("weights", os.path.basename(model_name))
    resolved_weights_path = get_resource_path(weights_subpath)
    if os.path.exists(resolved_weights_path):
        return resolved_weights_path

    # Check root level
    resolved_root_path = get_resource_path(os.path.basename(model_name))
    if os.path.exists(resolved_root_path):
        return resolved_root_path

    # Return default expected path under weights/
    return resolved_weights_path


def check_weights_exist(
    model_names: Optional[Union[str, List[str]]] = None,
    raise_error: bool = False,
) -> bool:
    """
    Validate that required model weight files exist before initializing inference models.

    Args:
        model_names (str | list[str] | None): Model filename or list of filenames to verify.
                                              Defaults to ['yolov8n-pose.pt'].
        raise_error (bool): If True, raises FileNotFoundError when a weight file is missing.

    Returns:
        bool: True if all specified weight files exist, False otherwise.

    Raises:
        FileNotFoundError: If raise_error is True and any weight file is missing.
    """
    if model_names is None:
        model_names = ["yolov8n-pose.pt"]
    elif isinstance(model_names, str):
        model_names = [model_names]

    missing_weights = []

    for name in model_names:
        resolved = get_model_path(name)
        if not os.path.exists(resolved):
            missing_weights.append((name, resolved))

    if missing_weights:
        error_msg = (
            "Required model weight files were not found:\n"
            + "\n".join(
                [f"  - '{name}' expected at: {path}" for name, path in missing_weights]
            )
            + "\nPlease ensure weights are placed in the 'weights/' folder or downloaded."
        )
        if raise_error:
            raise FileNotFoundError(error_msg)
        else:
            print(f"[WARN] {error_msg}")
            return False

    return True


def play_alert_sound(freq: int = 1200, duration_ms: int = 350) -> None:
    """
    Play an audio alert tone asynchronously without blocking the GUI/inference thread.
    Uses Windows winsound API or system bell as fallback.
    """

    def _beep():
        try:
            if sys.platform == "win32":
                import winsound

                winsound.Beep(freq, duration_ms)
            else:
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


def save_fall_snapshot(
    frame, prefix: str = "fall_incident", out_dir: str = "fall_snapshots",
    save_dir: str | None = None,
) -> str:
    """
    Save an annotated frame as a high-resolution incident snapshot.

    Args:
        frame: OpenCV BGR image matrix
        prefix (str): Prefix name for output file
        out_dir (str): Default destination folder
        save_dir (str | None): Override destination folder (takes priority over out_dir)

    Returns:
        str: Absolute path to the saved image file, or empty string on error.
    """
    try:
        import time
        import cv2

        target_dir = save_dir if save_dir else out_dir
        os.makedirs(target_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"{prefix}_{timestamp}_{ms}.jpg"
        filepath = os.path.abspath(os.path.join(target_dir, filename))
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        print(f"[WARN] Failed to save snapshot: {e}")
        return ""
