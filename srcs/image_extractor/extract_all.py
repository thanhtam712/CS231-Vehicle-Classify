"""
Script để trích xuất features cho tất cả 5 phương pháp
"""
import os
from img_extractor import ImageFeatureExtractor

# Cấu hình
DATA_DIR = '../../vehicle-10'
OUTPUT_BASE_DIR = '../../features'
TRAIN_META = '../../vehicle-10/train_split.json'
VAL_META = '../../vehicle-10/valid_split.json'

# Các phương pháp cần trích xuất
METHODS = ['clip', 'resnet', 'efficientnet', 'vgg16', 'hog']

def extract_all_methods():
    """Trích xuất features cho tất cả các phương pháp"""
    
    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"Extracting features using {method.upper()}")
        print(f"{'='*60}\n")
        
        try:
            # Khởi tạo extractor
            extractor = ImageFeatureExtractor(method=method)
            
            # Tạo thư mục output
            train_output = os.path.join(OUTPUT_BASE_DIR, method, 'train')
            val_output = os.path.join(OUTPUT_BASE_DIR, method, 'val')
            
            # Trích xuất train set
            print(f"\n--- Extracting TRAIN set ---")
            train_features, train_meta = extractor.extract_from_directory(
                data_dir=DATA_DIR,
                output_dir=train_output,
                metadata_file=TRAIN_META if os.path.exists(TRAIN_META) else None
            )
            
            # Trích xuất validation set
            print(f"\n--- Extracting VALIDATION set ---")
            val_features, val_meta = extractor.extract_from_directory(
                data_dir=DATA_DIR,
                output_dir=val_output,
                metadata_file=VAL_META if os.path.exists(VAL_META) else None
            )
            
            print(f"\n✓ {method.upper()} extraction completed successfully!")
            print(f"  Train: {train_features.shape}")
            print(f"  Val: {val_features.shape}")
            
        except Exception as e:
            print(f"\n✗ Error extracting {method.upper()}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("All extractions completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    extract_all_methods()
