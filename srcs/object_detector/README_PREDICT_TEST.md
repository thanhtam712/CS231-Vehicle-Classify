# Prediction and Evaluation on Test Set

## Scripts

### 1. `predict_test.py`
Dự đoán class cho toàn bộ tập test và tính accuracy cho một model cụ thể.

**Usage:**
```bash
python predict_test.py \
    --test_meta /path/to/test_meta.json \
    --dataset_dir /path/to/vehicle-10 \
    --model /path/to/model.pkl \
    --output_dir /path/to/output \
    --device cuda  # optional: cuda or cpu
```

**Arguments:**
- `--test_meta`: Đường dẫn đến file test_meta.json
- `--dataset_dir`: Thư mục chứa dataset (vehicle-10)
- `--model`: Đường dẫn đến model SVM đã train (.pkl file)
- `--output_dir`: Thư mục lưu kết quả (default: output)
- `--device`: Device to run model on (cuda/cpu, optional)

**Example:**
```bash
cd /mlcv3/WorkingSpace/Personal/baotg/TTam/CS231/srcs/object_detector

python predict_test.py \
    --test_meta ../../vehicle-10/test_meta.json \
    --dataset_dir ../../vehicle-10 \
    --model ../../models/svm_clip.pkl \
    --output_dir ../../output/svm_clip
```

**Output:**
Script sẽ tạo 2 files trong output_dir:
- `test_results.json`: Kết quả chi tiết bao gồm predictions, accuracy, per-class metrics
- `predictions.txt`: Danh sách predictions ở dạng đơn giản (image_path, label, class_name)

Console sẽ hiển thị:
- Overall accuracy
- Per-class accuracy
- Số lượng failed images (nếu có)

---

### 2. `evaluate_all_models.py`
Tự động chạy prediction và evaluation cho TẤT CẢ các models trong thư mục models/.

**Usage:**
```bash
python evaluate_all_models.py
```

Script sẽ:
1. Tự động tìm tất cả .pkl files trong thư mục models/
2. Chạy predict_test.py cho từng model
3. Lưu kết quả vào output/<model_name>/

**Example:**
```bash
cd /mlcv3/WorkingSpace/Personal/baotg/TTam/CS231/srcs/object_detector
python evaluate_all_models.py
```

**Output Structure:**
```
output/
├── svm_clip/
│   ├── test_results.json
│   └── predictions.txt
├── svm_efficientnet/
│   ├── test_results.json
│   └── predictions.txt
├── svm_hog/
│   ├── test_results.json
│   └── predictions.txt
├── svm_resnet/
│   ├── test_results.json
│   └── predictions.txt
└── svm_vgg16/
    ├── test_results.json
    └── predictions.txt
```

---

## Output Format

### test_results.json
```json
{
  "model_path": "path/to/model.pkl",
  "feature_method": "clip",
  "timestamp": "2024-11-16 12:30:45",
  "total_images": 7202,
  "valid_predictions": 7202,
  "failed_images": 0,
  "accuracy": 0.8523,
  "correct_predictions": 6140,
  "class_accuracy": {
    "bicycle": 0.8234,
    "boat": 0.8567,
    ...
  },
  "predictions": [
    {
      "image_path": "taxi/000302_01.jpg",
      "true_label": 7,
      "true_class": "taxi",
      "predicted_label": 7,
      "predicted_class": "taxi",
      "correct": true
    },
    ...
  ],
  "failed_images_list": []
}
```

### predictions.txt
Simple tab-separated format:
```
taxi/000302_01.jpg	7	taxi
taxi/002153_17.jpg	7	taxi
...
```

---

## Requirements

Make sure you have installed:
- numpy
- scikit-learn
- torch (for CLIP, ResNet, EfficientNet, VGG16)
- transformers (for CLIP)
- pillow (for image loading)
- scikit-image (for HOG)
- tqdm (for progress bar)

Install with:
```bash
pip install numpy scikit-learn torch torchvision transformers pillow scikit-image tqdm
```

---

## Notes

1. Script tự động xử lý:
   - Feature extraction cho mỗi image
   - Normalization (nếu cần)
   - Missing images
   - Failed feature extractions

2. Progress tracking với tqdm progress bar

3. Kết quả được lưu trong JSON format để dễ dàng phân tích sau này

4. Per-class accuracy giúp identify classes khó dự đoán

5. Console output formatted để dễ đọc với ASCII tables
