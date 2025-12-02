"""
Demo script để test trích xuất features từ một vài ảnh mẫu
"""
import os
from img_extractor import ImageFeatureExtractor
import numpy as np

def demo_single_method(method, test_image):
    """Demo trích xuất features cho một phương pháp"""
    print(f"\n{'='*60}")
    print(f"Testing {method.upper()} Feature Extraction")
    print(f"{'='*60}")
    
    try:
        # Khởi tạo extractor
        extractor = ImageFeatureExtractor(method=method)
        
        # Trích xuất features từ một ảnh
        print(f"\nExtracting features from: {test_image}")
        features = extractor.extract_single_image(test_image)
        
        if features is not None:
            print(f"✓ Success!")
            print(f"  Feature shape: {features.shape}")
            print(f"  Feature dim: {extractor.extractor.get_feature_dim()}")
            print(f"  Feature stats:")
            print(f"    - Mean: {features.mean():.4f}")
            print(f"    - Std: {features.std():.4f}")
            print(f"    - Min: {features.min():.4f}")
            print(f"    - Max: {features.max():.4f}")
        else:
            print(f"✗ Failed to extract features")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")


def demo_all_methods():
    """Demo tất cả các phương pháp"""
    
    # Tìm một ảnh test
    test_image = None
    base_dir = '../../vehicle-10'
    
    # Tìm ảnh đầu tiên trong thư mục bicycle
    bicycle_dir = os.path.join(base_dir, 'bicycle')
    if os.path.exists(bicycle_dir):
        images = [f for f in os.listdir(bicycle_dir) if f.endswith('.jpg')]
        if images:
            test_image = os.path.join(bicycle_dir, images[0])
    
    if not test_image:
        print("No test image found!")
        return
    
    print(f"Using test image: {test_image}\n")
    
    # Test tất cả các phương pháp
    methods = ['clip', 'resnet', 'efficientnet', 'vgg16', 'hog']
    
    results = {}
    for method in methods:
        demo_single_method(method, test_image)
        
    print(f"\n{'='*60}")
    print("Demo completed!")
    print(f"{'='*60}\n")


def demo_batch_extraction():
    """Demo trích xuất features từ một batch nhỏ"""
    print(f"\n{'='*60}")
    print("Testing Batch Feature Extraction")
    print(f"{'='*60}\n")
    
    # Tìm một vài ảnh test
    base_dir = '../../vehicle-10/bicycle'
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    # Lấy 5 ảnh đầu tiên
    images = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.jpg')][:5]
    
    print(f"Testing with {len(images)} images\n")
    
    # Test với CLIP
    method = 'clip'
    print(f"Extracting features using {method.upper()}...")
    
    extractor = ImageFeatureExtractor(method=method)
    
    features_list = []
    for img_path in images:
        features = extractor.extract_single_image(img_path)
        if features is not None:
            features_list.append(features)
    
    if features_list:
        features_array = np.array(features_list)
        print(f"✓ Batch extraction successful!")
        print(f"  Features shape: {features_array.shape}")
        print(f"  ({features_array.shape[0]} images × {features_array.shape[1]} features)")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'all':
            demo_all_methods()
        elif sys.argv[1] == 'batch':
            demo_batch_extraction()
        else:
            method = sys.argv[1]
            test_image = sys.argv[2] if len(sys.argv) > 2 else None
            if test_image:
                demo_single_method(method, test_image)
            else:
                print("Usage: python demo.py <method> <image_path>")
    else:
        print("Demo options:")
        print("  python demo.py all          - Test all methods")
        print("  python demo.py batch        - Test batch extraction")
        print("  python demo.py <method> <image_path> - Test specific method")
        print()
        demo_all_methods()
