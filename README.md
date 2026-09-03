# AI Human Fall Detection System - YOLOv8 Pose

Dự án nhận diện té ngã ở người sử dụng **YOLOv8-Pose** để trích xuất khung xương 17 keypoints, sau đó kết hợp logic phân tích tư thế và chuyển động để phân loại trạng thái **Fall / Normal**. Hệ thống hỗ trợ chạy real-time bằng GUI, xử lý ảnh/video, và đánh giá model bằng script benchmark tự động.

---

## Model

Model chính của dự án là:

```text
weights/yolov8n-pose.pt
```

Pipeline nhận diện gồm 2 tầng:

1. **Pose Estimation - YOLOv8-Pose**
   - Phát hiện người trong ảnh/video.
   - Trích xuất 17 COCO keypoints: vai, hông, đầu gối, cổ chân, v.v.
   - Trả về bounding box, độ tin cậy (confidence) và tọa độ keypoint theo từng frame.

2. **Fall Classifier / FSM Logic**
   - Phân tích tư thế và chuyển động của người dựa trên keypoints.
   - Dùng normalized hip-drop velocity (vận tốc rơi của hông đã chuẩn hóa):

```text
norm_v = (current_hip_y - previous_hip_y) / bbox_height
```

   - Dùng body angle (góc nghiêng thân mình) giữa mid-shoulders và mid-hips:
     - `angle < 30 deg`: thân người gần nằm ngang, hỗ trợ xác định trạng thái nằm/ngã (lying/fall).
     - `angle > 60 deg` và vận tốc dọc thấp: được xem là Bình thường (Normal).
   - Nếu hip keypoint có độ tin cậy thấp hơn `0.3`, pipeline sẽ chuyển sang cơ chế dự phòng (fallback) lấy trung bình tọa độ Y của vai trái/phải.

Thay vì dùng khoảng cách rơi pixel tuyệt đối, model hiện tại dùng tỷ lệ theo chiều cao bounding box. Cách này giúp kết quả ổn định hơn trên nhiều độ phân giải camera khác nhau.

---

## Cách Sử Dụng Model

### Cài Đặt

```powershell
pip install -r requirements.txt
```

### Chạy Ứng Dụng GUI

```powershell
python app_gui.py
```

GUI sử dụng `YOLOv8FallDetector` trong `fall_core.py`, load model từ `weights/yolov8n-pose.pt`, sau đó hiển thị bounding box, khung xương (skeleton), trạng thái cảnh báo té ngã (fall alert) và nhật ký sự kiện (audit log) theo thời gian thực.

### Chạy Benchmark Trên Dataset Ảnh

Dataset hiện có trong repository:

```text
fall_dataset/images/
  fall/      349 images
  not-fall/  226 images
```

Vì dataset này là ảnh tĩnh, nên sử dụng chế độ `--image-mode static`:

```powershell
python evaluate_model.py ".\fall_dataset\images" --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_static
```

Kết quả sẽ được lưu tại:

```text
evaluation_results_static/confusion_matrix.png
evaluation_results_static/predictions.csv
```

### Chạy Benchmark Trên Video

Với video, cần sắp xếp dữ liệu theo các thư mục nhãn:

```text
test_videos/
  fall/
    sample_01.mp4
  not-fall/
    sample_02.mp4
```

Lệnh đánh giá video:

```powershell
python evaluate_model.py ".\test_videos" --label-set binary --v-thresh 25 --dy-thresh 15 --output-dir evaluation_results_video
```

`--v-thresh 25` có nghĩa là ngưỡng hip-drop khoảng `25%` chiều cao cơ thể trong cửa sổ frame. Có thể truyền `0.25` nếu muốn viết trực tiếp theo dạng tỷ lệ (ratio).

---

## Kết Quả Model

Benchmark hiện tại được thực hiện trên `fall_dataset/images` với 575 ảnh:

```text
Fall:     349 images
Normal:   226 images
Total:    575 images
```

### Static Image Mode

Lệnh benchmark:

```powershell
python evaluate_model.py ".\fall_dataset\images" --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_static
```

Kết quả:

```text
Accuracy:           77.74%
Weighted F1-Score:  77.43%
```

Confusion matrix (Ma trận nhầm lẫn):

| Ground Truth | Pred Fall | Pred Normal |
| --- | ---: | ---: |
| Fall | 299 | 50 |
| Normal | 78 | 148 |

Per-class metrics (Chỉ số chi tiết theo lớp):

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| Fall | 79.31% | 85.67% | 82.37% | 349 |
| Normal | 74.75% | 65.49% | 69.81% | 226 |

### FSM/Evaluation Baseline

Kết quả benchmark lưu trong `evaluation_results`:

```text
Accuracy:           73.22%
Weighted F1-Score:  73.53%
```

Confusion matrix (Ma trận nhầm lẫn):

| Ground Truth | Pred Fall | Pred Normal |
| --- | ---: | ---: |
| Fall | 244 | 105 |
| Normal | 49 | 177 |

Kết quả cho thấy chế độ static phù hợp hơn với dataset ảnh tĩnh, trong khi FSM phù hợp hơn với video/real-time vì cần lịch sử nhiều frame để tính toán chuyển động rơi.

---

## Cấu Trúc Dự Án

```text
Human-Fall-Detection/
  app_gui.py                         GUI desktop real-time
  fall_core.py                       YOLOv8-Pose + fall detection logic
  evaluate_model.py                  Evaluation and benchmark script
  requirements.txt                   Python dependencies
  weights/
    yolov8n-pose.pt                  YOLOv8 pose model weights
  fall_dataset/
    images/
      fall/
      not-fall/
  evaluation_results/
    confusion_matrix.png
    predictions.csv
  evaluation_results_static/
    confusion_matrix.png
    predictions.csv
```

---

## Tham Số Quan Trọng

| Tham số | Ý nghĩa | Giá trị gợi ý |
| --- | --- | --- |
| `conf_thresh` / `--conf` | Ngưỡng confidence của YOLOv8-Pose | `0.15 - 0.35` |
| `v_thresh` / `--v-thresh` | Ngưỡng hip-drop normalized theo body height | `25` hoặc `0.25` |
| `dy_thresh` / `--dy-thresh` | Ngưỡng drop phụ, cũng được normalized | `15` hoặc `0.15` |
| `ar_thresh` / `--ar-thresh` | Ngưỡng thay đổi aspect ratio | `0.25 - 0.35` |
| `--static-fall-ar-thresh` | Ngưỡng width/height cho ảnh tĩnh | `0.65 - 0.85` |
| `--static-fall-score` | Điểm heuristic tối thiểu để gán nhãn Fall | `1.8 - 2.2` |

---

## Hướng Nâng Cấp Model

- Dùng model pose lớn hơn để tăng chất lượng trích xuất keypoints:

```powershell
python evaluate_model.py ".\fall_dataset\images" --model yolov8s-pose.pt --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_yolov8s
```

- Bổ sung video test có nhãn `fall/not-fall` để đánh giá đúng FSM temporal.
- Huấn luyện thêm classifier riêng trên đặc trưng keypoints, vì YOLOv8-Pose hiện tại chỉ đóng vai trò là pose estimator, còn quyết định té ngã là do logic classifier/FSM của dự án.
- Cân bằng lại dataset, vì tập hiện tại có nhiều ảnh Fall hơn Normal.
