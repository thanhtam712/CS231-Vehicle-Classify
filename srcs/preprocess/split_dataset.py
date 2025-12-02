"""
Script để chia dữ liệu từ train_meta.json thành 80% train và 20% validation
theo từng class (stratified split)
"""

import json
import random
import os
from collections import defaultdict
import argparse


def split_dataset_by_class(input_json, output_train_json, output_val_json, 
                           train_ratio=0.8, random_seed=42):
    """
    Chia dataset thành train và validation theo tỷ lệ cho mỗi class
    
    Args:
        input_json: Đường dẫn file JSON đầu vào
        output_train_json: Đường dẫn file JSON train output
        output_val_json: Đường dẫn file JSON validation output
        train_ratio: Tỷ lệ dữ liệu cho train (default: 0.8)
        random_seed: Seed cho random để reproduce (default: 42)
    """
    
    # Set random seed để có thể reproduce kết quả
    random.seed(random_seed)
    
    # Đọc dữ liệu từ file JSON
    print(f"Reading data from {input_json}...")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    paths = data['path']
    labels = data['label']
    
    print(f"Total samples: {len(paths)}")
    print(f"Total labels: {len(labels)}")
    
    # Nhóm các samples theo class
    class_samples = defaultdict(list)
    for i, (path, label) in enumerate(zip(paths, labels)):
        class_samples[label].append((path, label, i))
    
    # Hiển thị thông tin về các class
    print(f"\nFound {len(class_samples)} classes:")
    for label in sorted(class_samples.keys()):
        print(f"  Class {label}: {len(class_samples[label])} samples")
    
    # Chia dữ liệu cho mỗi class
    train_paths = []
    train_labels = []
    val_paths = []
    val_labels = []
    
    print(f"\nSplitting data with train_ratio={train_ratio}...")
    
    for label in sorted(class_samples.keys()):
        samples = class_samples[label]
        
        # Shuffle samples của class này
        random.shuffle(samples)
        
        # Tính số lượng samples cho train
        n_samples = len(samples)
        n_train = int(n_samples * train_ratio)
        
        # Chia train và val
        train_samples = samples[:n_train]
        val_samples = samples[n_train:]
        
        # Thêm vào danh sách
        for path, lbl, _ in train_samples:
            train_paths.append(path)
            train_labels.append(lbl)
        
        for path, lbl, _ in val_samples:
            val_paths.append(path)
            val_labels.append(lbl)
        
        print(f"  Class {label}: {len(train_samples)} train, {len(val_samples)} val")
    
    # Tạo dictionary cho train và val
    train_data = {
        'path': train_paths,
        'label': train_labels
    }
    
    val_data = {
        'path': val_paths,
        'label': val_labels
    }
    
    # Lưu train data
    print(f"\nSaving train data to {output_train_json}...")
    os.makedirs(os.path.dirname(output_train_json) if os.path.dirname(output_train_json) else '.', exist_ok=True)
    with open(output_train_json, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Lưu validation data
    print(f"Saving validation data to {output_val_json}...")
    os.makedirs(os.path.dirname(output_val_json) if os.path.dirname(output_val_json) else '.', exist_ok=True)
    with open(output_val_json, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Hiển thị thống kê cuối cùng
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Total samples: {len(train_paths) + len(val_paths)}")
    print(f"Train ratio: {len(train_paths) / (len(train_paths) + len(val_paths)):.2%}")
    print(f"Validation ratio: {len(val_paths) / (len(train_paths) + len(val_paths)):.2%}")
    print("="*60)
    
    # Kiểm tra phân bố của từng class
    print("\nClass distribution:")
    print(f"{'Class':<10} {'Train':<10} {'Val':<10} {'Total':<10} {'Train %':<10}")
    print("-" * 50)
    
    for label in sorted(class_samples.keys()):
        train_count = train_labels.count(label)
        val_count = val_labels.count(label)
        total_count = train_count + val_count
        train_pct = (train_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{label:<10} {train_count:<10} {val_count:<10} {total_count:<10} {train_pct:<10.2f}%")
    
    print("\n✓ Split completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Split dataset into train and validation sets by class'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../../vehicle-10/train_meta.json',
        help='Input JSON file path (default: ../../vehicle-10/train_meta.json)'
    )
    
    parser.add_argument(
        '--output_train',
        type=str,
        default='../../vehicle-10/train_split.json',
        help='Output train JSON file path (default: ../../vehicle-10/train_split.json)'
    )
    
    parser.add_argument(
        '--output_val',
        type=str,
        default='../../vehicle-10/val_split.json',
        help='Output validation JSON file path (default: ../../vehicle-10/val_split.json)'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Ratio of data for training (default: 0.8 for 80%%)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    split_dataset_by_class(
        input_json=args.input,
        output_train_json=args.output_train,
        output_val_json=args.output_val,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
