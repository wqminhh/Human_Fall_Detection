# AI Human Fall Detection & Alert System (YOLOv8 Pose)

Hệ thống nhận diện và cảnh báo té ngã ở người (Human Fall Detection System) theo thời gian thực sử dụng mô hình ước lượng tư thế **YOLOv8-Pose** kết hợp thuật toán phân tích động học tư thế (vận tốc trọng tâm, độ sụt trục Y và biến thiên tỉ lệ cơ thể). Hệ thống tích hợp giao diện người dùng máy tính (Desktop GUI) hiện đại và hỗ trợ đóng gói thành file thực thi độc lập (`.exe`) cho Windows.

---

## 📌 Tính năng nổi bật

- 🎯 **Ước lượng tư thế YOLOv8 (17 Keypoints)**: Nhận diện khung xương người thời gian thực với độ chính xác cao.
- ⚡ **Thuật toán phân tích té ngã thông minh**:
  - Tốc độ sụt trọng tâm $v$ (`SpeedDrop`).
  - Độ dịch chuyển trục dọc $\Delta y$ và biến thiên tỉ lệ dáng đứng sang nằm $\Delta AR$ (`DownFlat`, `SuddenDrop`).
- 🖥️ **Giao diện Desktop GUI Hiện Đại (`app_gui.py`)**:
  - Hỗ trợ 3 nguồn đầu vào: **Webcam trực tiếp**, **File video cục bộ** (`.mp4`, `.avi`, `.mov`, ...), và **Luồng RTSP/IP Camera**.
  - Bảng cảnh báo trạng thái trực quan nhấp nháy đỏ (`🚨 CRITICAL ALERT: FALL DETECTED`).
  - Âm thanh cảnh báo tự động và chụp ảnh lưu lại khoảnh khắc té ngã (`fall_snapshots/`).
  - Thanh trượt điều chỉnh ngưỡng nhận diện trực tiếp trên giao diện.
  - Bảng nhật ký sự kiện (Audit Log) theo thời gian thực.
- 📦 **Đóng gói Windows Executable Độc Lập (`build_exe.py`)**:
  - Sử dụng PyInstaller chế độ `--onedir` và `--windowed`.
  - Tự động đóng gói model weights (`weights/yolov8n-pose.pt`) và assets cần thiết.

---

## 🚀 Hướng dẫn Cài đặt & Sử dụng

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Chạy ứng dụng Desktop GUI

Khởi chạy giao diện Desktop bằng lệnh:
```bash
python app_gui.py
```
*(Hoặc nhấp đúp vào file `run_app.bat`)*

### 3. Đóng gói ứng dụng thành file `.exe` cho Windows

**Bước 3.1: Đóng gói thành thư mục Portable (.exe) bằng PyInstaller**:
```bash
python build_exe.py
```
*(Hoặc chạy file `build.bat`)*

**Bước 3.2: Tạo file cài đặt Setup Wizard (.exe) bằng Inno Setup (Tùy chọn)**:
Mở file [`setup_script.iss`](file:///d:/Human-Fall-Detection-master/Human-Fall-Detection-master/setup_script.iss) bằng phần mềm [Inno Setup Compiler](https://jrsoftware.org/isinfo.php) và nhấn `Compile` (Ctrl + F9) để xuất ra bộ cài đặt hoàn chỉnh `dist/FallDetectionApp_Setup_v1.0.exe`.

---

## 📂 Cấu trúc thư mục dự án

```text
Human-Fall-Detection/
├── app_gui.py              # Giao diện Desktop GUI (CustomTkinter + YOLOv8)
├── app_icon.ico            # Biểu tượng Icon ứng dụng đa độ phân giải
├── setup_script.iss        # Script Inno Setup tạo file cài đặt Windows Installer
├── build_exe.py            # Script tự động đóng gói PyInstaller (.exe)
├── build.bat               # File batch tự động đóng gói PyInstaller
├── run_app.bat             # File khởi chạy nhanh ứng dụng trên Windows
├── fall_core.py            # Bộ xử lý YOLOv8-pose và thuật toán phát hiện té ngã
├── utils.py                # Tiện ích giải quyết đường dẫn tài nguyên & âm thanh cảnh báo
├── requirements.txt        # Danh sách thư viện phụ thuộc
├── weights/                # Thư mục chứa trọng số mô hình (.pt)
│   └── yolov8n-pose.pt     # Trọng số mô hình YOLOv8 Pose
├── dist/                   # Thư mục chứa file thực thi sau khi đóng gói
│   └── FallDetectionApp/   # Ứng dụng Desktop chạy độc lập (.exe)
├── realtime.py             # Script nhận diện qua webcam (CLI)
├── video.py                # Script xử lý video hàng loạt (CLI)
└── fall_snapshots/         # Ảnh chụp tự động khi phát hiện sự cố ngã
```

---

## ⚙️ Tùy chỉnh tham số nhận diện

Bạn có thể tinh chỉnh trực tiếp trên thanh trượt của GUI hoặc qua file [`config.py`](file:///d:/Human-Fall-Detection-master/Human-Fall-Detection-master/config.py):

| Tham số | Mặc định | Ý nghĩa |
| :--- | :---: | :--- |
| `v_thresh` | `55.0 px/s` | Ngưỡng vận tốc sụt trọng tâm |
| `dy_thresh` | `18.0 px` | Ngưỡng dịch chuyển đi xuống theo trục Y |
| `ar_thresh` | `0.35` | Ngưỡng biến thiên tỉ lệ cơ thể (ngang/dọc) |
| `conf_thresh` | `0.35` | Độ tin cậy nhận diện tư thế người |
| `window_size` | `15 frames` | Số frame trong cửa sổ trượt phân tích |
