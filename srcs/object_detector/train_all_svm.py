"""
Script để train SVM cho tất cả 5 phương pháp feature extraction
"""

import os
from svm import train_svm_from_features

# Base directories
FEATURES_DIR = '../../features'
MODELS_DIR = '../../models'

# Feature extraction methods
METHODS = ['clip', 'resnet', 'efficientnet', 'vgg16', 'hog']

# Class names
CLASS_NAMES = [
    'bicycle', 'boat', 'bus', 'car', 'helicopter',
    'minibus', 'motorcycle', 'taxi', 'train', 'truck'
]

# SVM hyperparameters (có thể tune cho từng method)
SVM_PARAMS = {
    'clip': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'resnet': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'efficientnet': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'vgg16': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    'hog': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
}


def train_all_methods():
    """Train SVM cho tất cả các phương pháp"""
    
    # Create models directory
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    results = {}
    
    for method in METHODS:
        print(f"\n\n{'='*80}")
        print(f"Training SVM for {method.upper()}")
        print(f"{'='*80}\n")
        
        try:
            # Define paths
            train_features = os.path.join(FEATURES_DIR, method, 'train', f'features_{method}.npy')
            train_metadata = os.path.join(FEATURES_DIR, method, 'train', f'features_{method}_metadata.json')
            val_features = os.path.join(FEATURES_DIR, method, 'val', f'features_{method}.npy')
            val_metadata = os.path.join(FEATURES_DIR, method, 'val', f'features_{method}_metadata.json')
            output_model = os.path.join(MODELS_DIR, f'svm_{method}.pkl')
            
            # Check if files exist
            if not os.path.exists(train_features):
                print(f"✗ Error: Training features not found at {train_features}")
                print(f"  Please run feature extraction first!")
                continue
            
            if not os.path.exists(val_features):
                print(f"✗ Error: Validation features not found at {val_features}")
                print(f"  Please run feature extraction first!")
                continue
            
            # Get SVM parameters for this method
            params = SVM_PARAMS[method]
            
            # Train SVM
            train_svm_from_features(
                train_features_path=train_features,
                train_metadata_path=train_metadata,
                val_features_path=val_features,
                val_metadata_path=val_metadata,
                output_model_path=output_model,
                feature_method=method,
                kernel=params['kernel'],
                C=params['C'],
                gamma=params['gamma'],
                class_names=CLASS_NAMES
            )
            
            results[method] = {
                'status': 'success',
                'model_path': output_model
            }
            
            print(f"\n✓ {method.upper()} training completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error training {method.upper()}: {str(e)}")
            results[method] = {
                'status': 'failed',
                'error': str(e)
            }
            continue
    
    # Print summary
    print(f"\n\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    for method in METHODS:
        if method in results:
            status = results[method]['status']
            if status == 'success':
                print(f"✓ {method.upper():<15} - Success - Model: {results[method]['model_path']}")
            else:
                print(f"✗ {method.upper():<15} - Failed - Error: {results[method]['error']}")
        else:
            print(f"- {method.upper():<15} - Not attempted")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    train_all_methods()
