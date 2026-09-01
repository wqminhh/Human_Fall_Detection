"""
Dynamic Resource Path and Utility Module for Human Fall Detection.

Handles PyInstaller _MEIPASS path resolution for PyTorch weights (.pt/.pth),
configuration files, and UI assets.

Usage in detector.py or app_gui.py:
-----------------------------------
    from utils import get_resource_path, get_model_path, check_weights_exist

    # 1. Validate and resolve model weights:
    if check_weights_exist("yolov8n-pose.pt"):
        weight_path = get_model_path("yolov8n-pose.pt")
        model = YOLO(weight_path)

    # 2. Resolve UI icons or configuration assets:
    icon_path = get_resource_path("app_icon.ico")
"""

from utils.resource import (
    get_resource_path,
    get_model_path,
    check_weights_exist,
    play_alert_sound,
    save_fall_snapshot,
)

__all__ = [
    "get_resource_path",
    "get_model_path",
    "check_weights_exist",
    "play_alert_sound",
    "save_fall_snapshot",
]
