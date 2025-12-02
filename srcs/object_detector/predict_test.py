"""
Script để dự đoán class cho toàn bộ tập test và tính accuracy
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_extractor.clip_method import CLIPExtractor
from image_extractor.resnet_method import ResNetExtractor
from image_extractor.efficientnet_method import EfficientNetExtractor
from image_extractor.vgg16_method import VGG16Extractor
from image_extractor.hog_method import HOGExtractor
from svm import SVMClassifier


# Class names mapping
CLASS_NAMES = [
    'bicycle', 'boat', 'bus', 'car', 'helicopter',
    'minibus', 'motorcycle', 'taxi', 'train', 'truck'
]


def get_feature_extractor(method, device='cuda'):
    """
    Get feature extractor for given method
    
    Args:
        method: Feature extraction method
        device: Device to run model on (for deep learning methods)
    
    Returns:
        Feature extractor instance
    """
    method = method.lower()
    
    if method == 'clip':
        return CLIPExtractor(device=device)
    elif method == 'resnet':
        return ResNetExtractor(model_name='resnet50', device=device)
    elif method == 'efficientnet':
        return EfficientNetExtractor(model_name='efficientnet_b0', device=device)
    elif method == 'vgg16':
        return VGG16Extractor(device=device)
    elif method == 'hog':
        return HOGExtractor()
    else:
        raise ValueError(f"Unknown method: {method}")


def predict_test_set(test_meta_path, dataset_dir, model_path, output_dir, device='cuda'):
    """
    Dự đoán class cho toàn bộ tập test và tính accuracy
    
    Args:
        test_meta_path: Đường dẫn đến file test_meta.json
        dataset_dir: Thư mục chứa dataset
        model_path: Đường dẫn đến model SVM đã train
        output_dir: Thư mục lưu kết quả
        device: Device to run model on
    
    Returns:
        results: Dictionary chứa kết quả dự đoán và metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test metadata
    print(f"\nLoading test metadata from: {test_meta_path}")
    with open(test_meta_path, 'r') as f:
        test_data = json.load(f)
    
    image_paths = test_data['path']
    true_labels = test_data['label']
    
    print(f"Total test images: {len(image_paths)}")
    
    # Load SVM model
    print(f"\nLoading SVM model from: {model_path}")
    classifier = SVMClassifier()
    classifier.load_model(model_path)
    print(f"Feature method: {classifier.feature_method}")
    
    # Get feature extractor
    extractor = get_feature_extractor(classifier.feature_method, device=device)
    
    # Extract features and predict for all images
    print("\nExtracting features and predicting...")
    predictions = []
    valid_indices = []
    failed_images = []
    
    for idx, img_path in enumerate(tqdm(image_paths, desc="Processing images")):
        full_path = os.path.join(dataset_dir, img_path)
        
        # Check if file exists
        if not os.path.exists(full_path):
            print(f"\nWarning: File not found: {full_path}")
            failed_images.append(img_path)
            continue
        
        # Extract features
        features = extractor.extract_features(full_path)
        
        if features is not None:
            # Predict
            pred_class = classifier.predict(features.reshape(1, -1), normalize=True)[0]
            predictions.append(int(pred_class))
            valid_indices.append(idx)
        else:
            print(f"\nWarning: Failed to extract features from {full_path}")
            failed_images.append(img_path)
    
    # Get valid true labels
    valid_true_labels = [true_labels[i] for i in valid_indices]
    valid_image_paths = [image_paths[i] for i in valid_indices]
    
    # Calculate accuracy
    correct = sum([1 for pred, true in zip(predictions, valid_true_labels) if pred == true])
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-class accuracy
    class_correct = {i: 0 for i in range(len(CLASS_NAMES))}
    class_total = {i: 0 for i in range(len(CLASS_NAMES))}
    
    for pred, true in zip(predictions, valid_true_labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    class_accuracy = {}
    for i in range(len(CLASS_NAMES)):
        if class_total[i] > 0:
            class_accuracy[CLASS_NAMES[i]] = class_correct[i] / class_total[i]
        else:
            class_accuracy[CLASS_NAMES[i]] = 0.0
    
    # Prepare results
    results = {
        'model_path': model_path,
        'feature_method': classifier.feature_method,
        'test_meta_path': test_meta_path,
        'dataset_dir': dataset_dir,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(image_paths),
        'valid_predictions': total,
        'failed_images': len(failed_images),
        'accuracy': accuracy,
        'correct_predictions': correct,
        'class_accuracy': class_accuracy,
        'predictions': [
            {
                'image_path': img_path,
                'true_label': int(true_label),
                'true_class': CLASS_NAMES[true_label],
                'predicted_label': int(pred_label),
                'predicted_class': CLASS_NAMES[pred_label],
                'correct': pred_label == true_label
            }
            for img_path, true_label, pred_label in zip(valid_image_paths, valid_true_labels, predictions)
        ],
        'failed_images_list': failed_images
    }
    
    # Save results to JSON
    results_file = os.path.join(output_dir, 'test_results.json')
    print(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save predictions in simple format (for submission)
    predictions_file = os.path.join(output_dir, 'predictions.txt')
    with open(predictions_file, 'w') as f:
        for pred in results['predictions']:
            f.write(f"{pred['image_path']}\t{pred['predicted_label']}\t{pred['predicted_class']}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Feature Method: {classifier.feature_method}")
    print(f"{'='*80}")
    print(f"Total test images: {len(image_paths)}")
    print(f"Valid predictions: {total}")
    print(f"Failed images: {len(failed_images)}")
    print(f"\nOVERALL ACCURACY: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*80}")
    print(f"\nPER-CLASS ACCURACY:")
    print(f"{'-'*80}")
    for class_name in CLASS_NAMES:
        class_idx = CLASS_NAMES.index(class_name)
        class_acc = class_accuracy[class_name]
        class_count = class_total[class_idx]
        class_corr = class_correct[class_idx]
        print(f"{class_name:15s}: {class_acc:.4f} ({class_corr}/{class_count})")
    print(f"{'='*80}")
    
    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for img in failed_images[:10]:  # Show first 10
            print(f"  - {img}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {predictions_file}")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict on test set and calculate accuracy')
    
    parser.add_argument(
        '--test_meta',
        type=str,
        required=True,
        help='Path to test_meta.json file'
    )
    
    parser.add_argument(
        '--dataset_dir',
        type=str,
        required=True,
        help='Path to dataset directory containing images'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained SVM model (.pkl file)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Directory to save results (default: output)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run model on (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Run prediction on test set
    results = predict_test_set(
        test_meta_path=args.test_meta,
        dataset_dir=args.dataset_dir,
        model_path=args.model,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
