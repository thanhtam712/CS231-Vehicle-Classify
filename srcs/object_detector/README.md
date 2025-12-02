# SVM Vehicle Classifier

SVM classifier cho vehicle classification sử dụng các features đã trích xuất (CLIP, ResNet, EfficientNet, VGG16, HOG).

## Pipeline

```
1. Load danh sách ảnh + label
2. Với mỗi ảnh:
   - Trích đặc trưng: HOG / ResNet / VGG / EfficientNet / CLIP
   - Lưu vào X[i]
3. Train SVM từ X, y
4. Dùng SVM để dự đoán class cho ảnh mới (sau khi trích đặc trưng cùng cách)
```

## Cài đặt

```bash
pip install scikit-learn
```

## Sử dụng

### 1. Train SVM cho một phương pháp

```bash
cd /mlcv3/WorkingSpace/Personal/baotg/TTam/CS231/srcs/object_detector

# Train SVM với CLIP features
python svm.py \
    --train_features ../../features/clip/train/features_clip.npy \
    --train_metadata ../../features/clip/train/features_clip_metadata.json \
    --val_features ../../features/clip/val/features_clip.npy \
    --val_metadata ../../features/clip/val/features_clip_metadata.json \
    --output ../../models/svm_clip.pkl \
    --method clip \
    --kernel rbf \
    --C 1.0

# Train SVM với ResNet features
python svm.py \
    --train_features ../../features/resnet/train/features_resnet.npy \
    --train_metadata ../../features/resnet/train/features_resnet_metadata.json \
    --val_features ../../features/resnet/val/features_resnet.npy \
    --val_metadata ../../features/resnet/val/features_resnet_metadata.json \
    --output ../../models/svm_resnet.pkl \
    --method resnet
```

### 2. Train SVM cho tất cả phương pháp

```bash
python train_all_svm.py
```

Script này sẽ tự động train SVM cho tất cả 5 phương pháp:
- CLIP
- ResNet
- EfficientNet
- VGG16
- HOG

### 3. Dự đoán class cho ảnh mới

```bash
# Dự đoán một ảnh
python predict.py \
    --image ../../vehicle-10/bicycle/000001_00.jpg \
    --model ../../models/svm_clip.pkl

# Dự đoán nhiều ảnh
python predict.py \
    --images \
        ../../vehicle-10/bicycle/000001_00.jpg \
        ../../vehicle-10/car/000001_00.jpg \
        ../../vehicle-10/bus/000001_00.jpg \
    --model ../../models/svm_clip.pkl

# Chỉ định device (GPU/CPU)
python predict.py \
    --image ../../vehicle-10/bicycle/000001_00.jpg \
    --model ../../models/svm_clip.pkl \
    --device cuda
```

## Sử dụng trong Python code

### Train SVM

```python
from svm import train_svm_from_features

# Class names
class_names = [
    'bicycle', 'boat', 'bus', 'car', 'helicopter',
    'minibus', 'motorcycle', 'taxi', 'train', 'truck'
]

# Train SVM
train_svm_from_features(
    train_features_path='../../features/clip/train/features_clip.npy',
    train_metadata_path='../../features/clip/train/features_clip_metadata.json',
    val_features_path='../../features/clip/val/features_clip.npy',
    val_metadata_path='../../features/clip/val/features_clip_metadata.json',
    output_model_path='../../models/svm_clip.pkl',
    feature_method='clip',
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_names=class_names
)
```

### Dự đoán với SVM

```python
from svm import SVMClassifier
from image_extractor.clip_method import CLIPExtractor

# Load model
classifier = SVMClassifier()
classifier.load_model('../../models/svm_clip.pkl')

# Extract features from new image
extractor = CLIPExtractor()
features = extractor.extract_features('path/to/image.jpg')

# Predict
predicted_class = classifier.predict(features)
print(f"Predicted class: {predicted_class}")
```

## Tham số SVM

### Kernel types
- `linear`: Linear kernel (tốt cho dữ liệu linearly separable)
- `rbf`: Radial Basis Function kernel (default, tốt cho hầu hết trường hợp)
- `poly`: Polynomial kernel
- `sigmoid`: Sigmoid kernel

### Hyperparameters
- `C`: Regularization parameter (default: 1.0)
  - C nhỏ: Smooth decision boundary, có thể underfit
  - C lớn: Complex decision boundary, có thể overfit
  
- `gamma`: Kernel coefficient (default: 'scale')
  - `'scale'`: 1 / (n_features * X.var())
  - `'auto'`: 1 / n_features
  - float value: Custom value

## Cấu trúc files

```
object_detector/
├── svm.py                  # SVM classifier class
├── train_all_svm.py       # Train SVM cho tất cả methods
├── predict.py             # Dự đoán class cho ảnh mới
└── README.md

models/                     # Trained models
├── svm_clip.pkl
├── svm_resnet.pkl
├── svm_efficientnet.pkl
├── svm_vgg16.pkl
└── svm_hog.pkl
```

## Output format

### Training output
```
============================================================
SVM Training Pipeline
============================================================

Initialized SVM Classifier
  Feature method: clip
  Kernel: rbf
  C: 1.0
  Gamma: scale

Loading features from ../../features/clip/train/features_clip.npy...
  Loaded 28000 samples
  Feature dimension: 512
  Number of classes: 10

============================================================
Training SVM Classifier
============================================================
Training samples: 28000
Feature dimension: 512
Number of classes: 10

Normalizing features...
Training SVM...

✓ Training completed!
Training accuracy: 0.9850 (98.50%)

============================================================
Evaluating SVM Classifier
============================================================
Test samples: 7000

Making predictions...

============================================================
RESULTS
============================================================
Test accuracy: 0.9642 (96.42%)

Classification Report:
------------------------------------------------------------
              precision    recall  f1-score   support

     bicycle     0.9650    0.9700    0.9675       700
        boat     0.9580    0.9620    0.9600       700
         bus     0.9720    0.9680    0.9700       700
         car     0.9690    0.9710    0.9700       700
  helicopter     0.9710    0.9650    0.9680       700
     minibus     0.9600    0.9590    0.9595       700
  motorcycle     0.9630    0.9640    0.9635       700
        taxi     0.9580    0.9600    0.9590       700
       train     0.9670    0.9700    0.9685       700
       truck     0.9590    0.9630    0.9610       700
```

### Prediction output
```
Predicting class for: ../../vehicle-10/bicycle/000001_00.jpg
Using feature method: clip
Extracting features...
Predicting class...

============================================================
PREDICTION RESULT
============================================================
Predicted class: 0
Predicted label: bicycle
============================================================
```

## Lưu ý

- Features phải được trích xuất trước khi train SVM
- Model SVM sẽ tự động normalize features khi train và predict
- Để dự đoán ảnh mới, cần trích xuất features bằng cùng phương pháp đã train
- Có thể tune hyperparameters (kernel, C, gamma) để cải thiện accuracy
