# Image Feature Extractor

Thư viện trích xuất đặc trưng hình ảnh sử dụng 5 phương pháp khác nhau.

## Các phương pháp

1. **CLIP** - Vision Transformer (ViT-B/32) - 512 chiều
2. **ResNet** - ResNet50 - 2048 chiều
3. **EfficientNet** - EfficientNet-B0 - 1280 chiều
4. **VGG16** - VGG16 - 4096 chiều
5. **HOG** - Histogram of Oriented Gradients - 8100 chiều

## Cài đặt

```bash
cd /mlcv3/WorkingSpace/Personal/baotg/TTam/CS231/srcs
pip install -r requirements.txt
```

## Cách sử dụng

### 1. Trích xuất từ thư mục

```bash
cd image_extractor

# Sử dụng CLIP
python img_extractor.py --method clip --data_dir ../../vehicle-10 --output_dir ../../features/clip

# Sử dụng ResNet
python img_extractor.py --method resnet --data_dir ../../vehicle-10 --output_dir ../../features/resnet

# Sử dụng EfficientNet
python img_extractor.py --method efficientnet --data_dir ../../vehicle-10 --output_dir ../../features/efficientnet

# Sử dụng VGG16
python img_extractor.py --method vgg16 --data_dir ../../vehicle-10 --output_dir ../../features/vgg16

# Sử dụng HOG
python img_extractor.py --method hog --data_dir ../../vehicle-10 --output_dir ../../features/hog
```

### 2. Sử dụng trong code Python

```python
from img_extractor import ImageFeatureExtractor

# Khởi tạo extractor
extractor = ImageFeatureExtractor(method='clip', device='cuda')

# Trích xuất từ một ảnh
features = extractor.extract_single_image('path/to/image.jpg')
print(f"Feature shape: {features.shape}")

# Trích xuất từ thư mục
features_array, metadata = extractor.extract_from_directory(
    data_dir='../../vehicle-10',
    output_dir='../../features',
    metadata_file='../../vehicle-10/train_split.json'
)
```

### 3. Trích xuất cho cả tập train và validation

```python
from img_extractor import ImageFeatureExtractor

# Chọn phương pháp
method = 'clip'  # hoặc 'resnet', 'efficientnet', 'vgg16', 'hog'

extractor = ImageFeatureExtractor(method=method)

# Train set
train_features, train_meta = extractor.extract_from_directory(
    data_dir='../../vehicle-10',
    output_dir=f'../../features/{method}/train',
    metadata_file='../../vehicle-10/train_split.json'
)

# Validation set
val_features, val_meta = extractor.extract_from_directory(
    data_dir='../../vehicle-10',
    output_dir=f'../../features/{method}/val',
    metadata_file='../../vehicle-10/valid_split.json'
)
```

## Cấu trúc output

Sau khi trích xuất, bạn sẽ có:

```
features/
├── clip/
│   ├── features_clip.npy          # Mảng numpy của features
│   └── features_clip_metadata.json # Metadata (đường dẫn, nhãn, ...)
├── resnet/
│   ├── features_resnet.npy
│   └── features_resnet_metadata.json
├── efficientnet/
│   ├── features_efficientnet.npy
│   └── features_efficientnet_metadata.json
├── vgg16/
│   ├── features_vgg16.npy
│   └── features_vgg16_metadata.json
└── hog/
    ├── features_hog.npy
    └── features_hog_metadata.json
```

## Kích thước features

- **CLIP**: 512 chiều
- **ResNet50**: 2048 chiều
- **EfficientNet-B0**: 1280 chiều
- **VGG16**: 4096 chiều
- **HOG**: 8100 chiều (có thể thay đổi tùy theo cấu hình)

## Tham số tùy chỉnh

### HOG
```python
from hog_method import HOGExtractor

extractor = HOGExtractor(
    orientations=9,           # Số bin hướng
    pixels_per_cell=(8, 8),   # Kích thước cell
    cells_per_block=(2, 2),   # Số cell trong mỗi block
    resize_shape=(128, 128)   # Kích thước resize ảnh
)
```

### ResNet
```python
from resnet_method import ResNetExtractor

# Chọn model: resnet18, resnet34, resnet50, resnet101, resnet152
extractor = ResNetExtractor(model_name='resnet50', device='cuda')
```

### EfficientNet
```python
from efficientnet_method import EfficientNetExtractor

# Chọn model: efficientnet_b0 đến efficientnet_b7
extractor = EfficientNetExtractor(model_name='efficientnet_b0', device='cuda')
```

## Lưu ý

- Các phương pháp deep learning (CLIP, ResNet, EfficientNet, VGG16) yêu cầu GPU để tăng tốc
- HOG không cần GPU và chạy nhanh trên CPU
- Đảm bảo đủ dung lượng RAM khi trích xuất nhiều ảnh
