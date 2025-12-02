"""
Script để dự đoán class cho ảnh mới sử dụng SVM đã train
"""

import os
import sys
import numpy as np
from pathlib import Path
import argparse

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


def get_feature_extractor(method, device=None):
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


def predict_image(image_path, model_path, device=None):
    """
    Dự đoán class cho một ảnh
    
    Args:
        image_path: Đường dẫn đến ảnh
        model_path: Đường dẫn đến model SVM đã train
        device: Device to run model on
    
    Returns:
        predicted_class: Class dự đoán
        predicted_label: Label name của class
    """
    # Load SVM model
    classifier = SVMClassifier()
    classifier.load_model(model_path)
    
    print(f"\nPredicting class for: {image_path}")
    print(f"Using feature method: {classifier.feature_method}")
    
    # Get feature extractor
    extractor = get_feature_extractor(classifier.feature_method, device=device)
    
    # Extract features
    print("Extracting features...")
    features = extractor.extract_features(image_path)
    
    if features is None:
        raise ValueError(f"Failed to extract features from {image_path}")
    
    # Predict
    print("Predicting class...")
    predicted_class = classifier.predict(features, normalize=True)[0]
    predicted_label = CLASS_NAMES[predicted_class] if predicted_class < len(CLASS_NAMES) else str(predicted_class)
    
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"Predicted class: {predicted_class}")
    print(f"Predicted label: {predicted_label}")
    print(f"{'='*60}\n")
    
    return predicted_class, predicted_label


def predict_batch(image_paths, model_path, device=None):
    """
    Dự đoán class cho nhiều ảnh
    
    Args:
        image_paths: List đường dẫn đến các ảnh
        model_path: Đường dẫn đến model SVM đã train
        device: Device to run model on
    
    Returns:
        predictions: List các kết quả dự đoán
    """
    # Load SVM model
    classifier = SVMClassifier()
    classifier.load_model(model_path)
    
    print(f"\nPredicting {len(image_paths)} images...")
    print(f"Using feature method: {classifier.feature_method}")
    
    # Get feature extractor
    extractor = get_feature_extractor(classifier.feature_method, device=device)
    
    # Extract features for all images
    print("\nExtracting features...")
    features_list = []
    valid_paths = []
    
    for img_path in image_paths:
        features = extractor.extract_features(img_path)
        if features is not None:
            features_list.append(features)
            valid_paths.append(img_path)
        else:
            print(f"Warning: Failed to extract features from {img_path}")
    
    if not features_list:
        raise ValueError("No valid features extracted!")
    
    # Convert to array
    X = np.array(features_list)
    
    # Predict
    print("Predicting classes...")
    predicted_classes = classifier.predict(X, normalize=True)
    
    # Prepare results
    predictions = []
    for img_path, pred_class in zip(valid_paths, predicted_classes):
        pred_label = CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
        predictions.append({
            'image_path': img_path,
            'predicted_class': int(pred_class),
            'predicted_label': pred_label
        })
    
    # Print results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS")
    print(f"{'='*60}")
    for pred in predictions:
        print(f"{pred['image_path']}")
        print(f"  → {pred['predicted_label']} (class {pred['predicted_class']})")
    print(f"{'='*60}\n")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict vehicle class using trained SVM')
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Paths to multiple image files'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained SVM model (.pkl file)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run model on (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Check inputs
    if not args.image and not args.images:
        parser.error("Either --image or --images must be provided")
    
    # Single image prediction
    if args.image:
        predict_image(args.image, args.model, device=args.device)
    
    # Batch prediction
    if args.images:
        predict_batch(args.images, args.model, device=args.device)


if __name__ == '__main__':
    main()
