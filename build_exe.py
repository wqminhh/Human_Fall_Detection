"""
Automated PyInstaller Build Script for Human Fall Detection Desktop App
Packages the application into a standalone Windows directory (onedir + windowed mode).
"""

import os
import sys
import shutil
import subprocess

# Auto-detect incompatible Python 3.13 freethreaded environment and redirect to Python 3.12
if sys.version_info >= (3, 13):
    try:
        import numpy
        import cv2
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

import PyInstaller.__main__
import customtkinter
import ultralytics


def build_executable():
    print("=" * 70)
    print("🚀 Building Standalone Windows Executable: FallDetectionApp")
    print("=" * 70)

    # 1. Project Directory Paths
    base_dir = os.path.abspath(os.path.dirname(__file__))
    entry_script = os.path.join(base_dir, "app_gui.py")
    weights_dir = os.path.join(base_dir, "weights")
    dist_dir = os.path.join(base_dir, "dist")
    build_dir = os.path.join(base_dir, "build")

    # 2. Ensure weights exist
    os.makedirs(weights_dir, exist_ok=True)
    default_weight = os.path.join(weights_dir, "yolov8n-pose.pt")
    if not os.path.exists(default_weight):
        print("[INFO] Default weight not found in weights/. Downloading yolov8n-pose.pt...")
        model = ultralytics.YOLO("yolov8n-pose.pt")
        if os.path.exists("yolov8n-pose.pt"):
            shutil.move("yolov8n-pose.pt", default_weight)

    # 3. Locate Package Asset Folders
    ctk_path = os.path.abspath(customtkinter.__path__[0])
    ultralytics_path = os.path.abspath(ultralytics.__path__[0])

    print(f"📦 CustomTkinter assets: {ctk_path}")
    print(f"📦 Ultralytics assets:   {ultralytics_path}")
    print(f"📦 Model Weights:        {weights_dir}")

    # Separator for PyInstaller --add-data (';' for Windows, ':' for Linux/macOS)
    sep = ";" if sys.platform == "win32" else ":"

    # 4. Construct PyInstaller Arguments
    pyinstaller_args = [
        entry_script,
        "--name=FallDetectionApp",
        "--onedir",  # Folder mode to avoid unpacking delays
        "--windowed",  # No console window (GUI mode)
        "--clean",  # Clean cache before build
        "--noconfirm",  # Overwrite existing output directory
        # Bundle required asset folders
        f"--add-data={weights_dir}{sep}weights",
        f"--add-data={os.path.join(ctk_path, 'assets')}{sep}customtkinter/assets",
        f"--add-data={ctk_path}{sep}customtkinter",
        f"--add-data={ultralytics_path}{sep}ultralytics",
        f"--add-data={os.path.join(base_dir, 'app_icon.ico')}{sep}.",
        # Hidden imports for dynamic loader resolution
        "--hidden-import=ultralytics",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=customtkinter",
        "--hidden-import=darkdetect",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=dotenv",
        "--hidden-import=scipy",
        "--hidden-import=matplotlib",
        "--hidden-import=tqdm",
        "--hidden-import=pywin32_ctypes",
        # Collect all metadata/data files
        "--collect-all=customtkinter",
        "--collect-all=ultralytics",
    ]

    # Optional app icon if present
    icon_path = os.path.join(base_dir, "app_icon.ico")
    if os.path.exists(icon_path):
        pyinstaller_args.append(f"--icon={icon_path}")

    print("\n[INFO] Running PyInstaller with arguments:")
    for arg in pyinstaller_args:
        print(f"  {arg}")
    print("\n⏳ Packaging in progress, please wait...\n")

    # 5. Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)

    # 6. Post-Build Asset Guarantee & Verification
    output_app_dir = os.path.join(dist_dir, "FallDetectionApp")
    internal_dir = os.path.join(output_app_dir, "_internal")

    # Ensure CustomTkinter assets are present in both _internal and app root
    ctk_assets_src = os.path.join(ctk_path, "assets")
    if os.path.exists(ctk_assets_src):
        # Target 1: _internal/customtkinter/assets
        dest_internal = os.path.join(internal_dir, "customtkinter", "assets")
        os.makedirs(dest_internal, exist_ok=True)
        shutil.copytree(ctk_assets_src, dest_internal, dirs_exist_ok=True)

        # Target 2: customtkinter/assets (root fallback)
        dest_root = os.path.join(output_app_dir, "customtkinter", "assets")
        os.makedirs(dest_root, exist_ok=True)
        shutil.copytree(ctk_assets_src, dest_root, dirs_exist_ok=True)

    # Ensure weights are present in both _internal/weights and dist root
    if os.path.exists(weights_dir):
        dest_w1 = os.path.join(internal_dir, "weights")
        dest_w2 = os.path.join(output_app_dir, "weights")
        os.makedirs(dest_w1, exist_ok=True)
        os.makedirs(dest_w2, exist_ok=True)
        shutil.copytree(weights_dir, dest_w1, dirs_exist_ok=True)
        shutil.copytree(weights_dir, dest_w2, dirs_exist_ok=True)

    # Ensure app_icon.ico is present
    if os.path.exists(icon_path):
        shutil.copy2(icon_path, os.path.join(output_app_dir, "app_icon.ico"))
        shutil.copy2(icon_path, os.path.join(internal_dir, "app_icon.ico"))

    exe_name = (
        "FallDetectionApp.exe"
        if sys.platform == "win32"
        else "FallDetectionApp"
    )
    output_exe = os.path.join(output_app_dir, exe_name)

    print("\n" + "=" * 70)
    if os.path.exists(output_exe):
        exe_size_mb = os.path.getsize(output_exe) / (1024 * 1024)
        theme_check = os.path.join(dest_internal, "themes", "blue.json")
        theme_ok = "✅ Verified" if os.path.exists(theme_check) else "⚠️ Missing"
        print("✅ BUILD SUCCESSFUL!")
        print(f"📁 Output Directory:      {output_app_dir}")
        print(f"⚡ Executable File:       {output_exe} ({exe_size_mb:.2f} MB)")
        print(f"🎨 CustomTkinter Assets:  {theme_ok}")
        print(
            "\n💡 You can test and run your application by executing:"
        )
        print(f"   {output_exe}")
    else:
        print("❌ BUILD FAILED: Executable not found at expected location.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    build_executable()
