# AI Human Fall Detection System - YOLOv8 Pose

Du an nhan dien te nga o nguoi su dung **YOLOv8-Pose** de trich xuat khung xuong 17 keypoints, sau do ket hop logic phan tich tu the va chuyen dong de phan loai trang thai **Fall / Normal**. He thong ho tro chay real-time bang GUI, xu ly anh/video, va danh gia model bang script benchmark tu dong.

## Model

Model chinh cua du an la:

```text
weights/yolov8n-pose.pt
```

Pipeline nhan dien gom 2 tang:

1. **Pose Estimation - YOLOv8-Pose**
   - Phat hien nguoi trong anh/video.
   - Trich xuat 17 COCO keypoints: shoulders, hips, knees, ankles, etc.
   - Tra ve bounding box, confidence va toa do keypoint theo tung frame.

2. **Fall Classifier / FSM Logic**
   - Phan tich tu the va chuyen dong cua nguoi dua tren keypoints.
   - Dung normalized hip-drop velocity:

```text
norm_v = (current_hip_y - previous_hip_y) / bbox_height
```

   - Dung body angle giua mid-shoulders va mid-hips:
     - `angle < 30 deg`: than nguoi gan nam ngang, ho tro trang thai lying/fall.
     - `angle > 60 deg` va van toc doc thap: xem la Normal.
   - Neu hip keypoint co confidence thap hon `0.3`, pipeline fallback sang average Y cua left/right shoulders.

Thay vi dung pixel drop tuyet doi, model hien tai dung ty le theo chieu cao bounding box. Cach nay giup ket qua on dinh hon voi nhieu do phan giai camera khac nhau.

## Cach Su Dung Model

### Cai Dat

```powershell
pip install -r requirements.txt
```

### Chay Ung Dung GUI

```powershell
python app_gui.py
```

GUI su dung `YOLOv8FallDetector` trong `fall_core.py`, load model tu `weights/yolov8n-pose.pt`, sau do hien thi bounding box, skeleton, trang thai fall alert va audit log theo thoi gian thuc.

### Chay Benchmark Tren Dataset Anh

Dataset hien co trong repo:

```text
fall_dataset/images/
  fall/      349 images
  not-fall/  226 images
```

Vi dataset nay la anh tinh, nen nen dung che do `--image-mode static`:

```powershell
python evaluate_model.py ".\fall_dataset\images" --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_static
```

Ket qua se duoc luu tai:

```text
evaluation_results_static/confusion_matrix.png
evaluation_results_static/predictions.csv
```

### Chay Benchmark Tren Video

Voi video, can sap xep du lieu theo folder nhan:

```text
test_videos/
  fall/
    sample_01.mp4
  not-fall/
    sample_02.mp4
```

Lenh danh gia video:

```powershell
python evaluate_model.py ".\test_videos" --label-set binary --v-thresh 25 --dy-thresh 15 --output-dir evaluation_results_video
```

`--v-thresh 25` co nghia la nguong hip-drop khoang `25%` chieu cao co the trong cua so frame. Co the truyen `0.25` neu muon viet truc tiep theo ratio.

## Ket Qua Model

Benchmark hien tai duoc thuc hien tren `fall_dataset/images` voi 575 anh:

```text
Fall:     349 images
Normal:   226 images
Total:    575 images
```

### Static Image Mode

Lenh benchmark:

```powershell
python evaluate_model.py ".\fall_dataset\images" --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_static
```

Ket qua:

```text
Accuracy:           77.74%
Weighted F1-Score:  77.43%
```

Confusion matrix:

| Ground Truth | Pred Fall | Pred Normal |
| --- | ---: | ---: |
| Fall | 299 | 50 |
| Normal | 78 | 148 |

Per-class metrics:

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| Fall | 79.31% | 85.67% | 82.37% | 349 |
| Normal | 74.75% | 65.49% | 69.81% | 226 |

### FSM/Evaluation Baseline

Ket qua benchmark luu trong `evaluation_results`:

```text
Accuracy:           73.22%
Weighted F1-Score:  73.53%
```

Confusion matrix:

| Ground Truth | Pred Fall | Pred Normal |
| --- | ---: | ---: |
| Fall | 244 | 105 |
| Normal | 49 | 177 |

Ket qua cho thay che do static phu hop hon voi dataset anh tinh, trong khi FSM phu hop hon voi video/real-time vi can lich su nhieu frame de tinh chuyen dong roi.

## Cau Truc Du An

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

## Tham So Quan Trong

| Tham so | Y nghia | Gia tri goi y |
| --- | --- | --- |
| `conf_thresh` / `--conf` | Nguong confidence cua YOLOv8-Pose | `0.15 - 0.35` |
| `v_thresh` / `--v-thresh` | Nguong hip-drop normalized theo body height | `25` hoac `0.25` |
| `dy_thresh` / `--dy-thresh` | Nguong drop phu, cung normalized | `15` hoac `0.15` |
| `ar_thresh` / `--ar-thresh` | Nguong thay doi aspect ratio | `0.25 - 0.35` |
| `--static-fall-ar-thresh` | Nguong width/height cho anh tinh | `0.65 - 0.85` |
| `--static-fall-score` | Diem heuristic toi thieu de gan nhan Fall | `1.8 - 2.2` |

## Huong Nang Cap Model

- Dung model pose lon hon de tang chat luong keypoints:

```powershell
python evaluate_model.py ".\fall_dataset\images" --model yolov8s-pose.pt --label-set binary --image-mode static --conf 0.15 --static-fall-ar-thresh 0.75 --static-fall-score 2.0 --output-dir evaluation_results_yolov8s
```

- Bo sung video test co nhan `fall/not-fall` de danh gia dung FSM temporal.
- Train them classifier rieng tren feature keypoints, vi YOLOv8-Pose hien tai chi la pose estimator, con fall decision la logic classifier/FSM cua du an.
- Can bang lai dataset, vi tap hien tai co nhieu anh Fall hon Normal.
