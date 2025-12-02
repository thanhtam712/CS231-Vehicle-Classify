"""
SVM Classifier cho vehicle classification
Pipeline:
1. Load features đã trích xuất (HOG/ResNet/VGG/EfficientNet/CLIP)
2. Train SVM từ features
3. Dự đoán class cho ảnh mới
"""

import os
import json
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import argparse
from pathlib import Path


class SVMClassifier:
    """
    SVM Classifier for vehicle classification using extracted features
    """
    
    def __init__(self, feature_method='clip', kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize SVM classifier
        
        Args:
            feature_method: Feature extraction method ('clip', 'resnet', 'efficientnet', 'vgg16', 'hog')
            kernel: SVM kernel ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        """
        self.feature_method = feature_method.lower()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        # Initialize SVM model
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            random_state=42,
            verbose=False
        )
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Class names mapping
        self.class_names = None
        
        print(f"Initialized SVM Classifier")
        print(f"  Feature method: {self.feature_method}")
        print(f"  Kernel: {self.kernel}")
        print(f"  C: {self.C}")
        print(f"  Gamma: {self.gamma}")
    
    def load_features(self, features_path, metadata_path):
        """
        Load extracted features and labels
        
        Args:
            features_path: Path to .npy file containing features
            metadata_path: Path to JSON file containing metadata (labels, paths)
        
        Returns:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            metadata: Dictionary with paths and other info
        """
        print(f"\nLoading features from {features_path}...")
        X = np.load(features_path)
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Map class names to indices
        class_name_to_idx = {
            'bicycle': 0, 'boat': 1, 'bus': 2, 'car': 3, 'helicopter': 4,
            'minibus': 5, 'motorcycle': 6, 'taxi': 7, 'train': 8, 'truck': 9
        }
        
        # Convert labels to integers (handle both string and int labels)
        labels = metadata['labels']
        y_list = []
        
        for label in labels:
            if isinstance(label, str):
                # String label - convert using mapping
                y_list.append(class_name_to_idx.get(label.lower(), int(label) if label.isdigit() else 0))
            else:
                # Numeric label
                y_list.append(int(label))
        
        y = np.array(y_list, dtype=int)
        
        print(f"  Loaded {X.shape[0]} samples")
        print(f"  Feature dimension: {X.shape[1]}")
        print(f"  Number of classes: {len(np.unique(y))}")
        print(f"  Label range: [{y.min()}, {y.max()}]")
        
        return X, y, metadata
    
    def train(self, X_train, y_train, normalize=True):
        """
        Train SVM model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            normalize: Whether to normalize features (recommended)
        
        Returns:
            self
        """
        print(f"\n{'='*60}")
        print(f"Training SVM Classifier")
        print(f"{'='*60}")
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Feature dimension: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        # Normalize features
        if normalize:
            print("\nNormalizing features...")
            X_train = self.scaler.fit_transform(X_train)
        
        # Train SVM
        print("\nTraining SVM...")
        self.model.fit(X_train, y_train)
        
        # Training accuracy
        y_train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        print(f"\n✓ Training completed!")
        print(f"Training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        
        return self
    
    def evaluate(self, X_test, y_test, normalize=True, class_names=None):
        """
        Evaluate SVM model on test set
        
        Args:
            X_test: Test features (n_samples, n_features)
            y_test: Test labels (n_samples,)
            normalize: Whether to normalize features
            class_names: Optional list of class names for better reporting
        
        Returns:
            accuracy: Test accuracy
            report: Classification report
            conf_matrix: Confusion matrix
        """
        print(f"\n{'='*60}")
        print(f"Evaluating SVM Classifier")
        print(f"{'='*60}")
        
        print(f"Test samples: {X_test.shape[0]}")
        
        # Normalize features
        if normalize:
            X_test = self.scaler.transform(X_test)
        
        # Predict
        print("\nMaking predictions...")
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print("-" * 60)
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        return accuracy, report, conf_matrix
    
    def predict(self, X, normalize=True):
        """
        Predict class for new samples
        
        Args:
            X: Features (n_samples, n_features) or (n_features,)
            normalize: Whether to normalize features
        
        Returns:
            predictions: Predicted class labels
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize
        if normalize:
            X = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, X, normalize=True):
        """
        Predict class probabilities for new samples
        Note: SVM needs to be trained with probability=True for this to work
        
        Args:
            X: Features (n_samples, n_features) or (n_features,)
            normalize: Whether to normalize features
        
        Returns:
            probabilities: Predicted class probabilities
        """
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize
        if normalize:
            X = self.scaler.transform(X)
        
        # Predict probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)
        else:
            # If probability not enabled, return decision function scores
            print("Warning: Model not trained with probability=True. Returning decision scores.")
            probabilities = self.model.decision_function(X)
        
        return probabilities
    
    def save_model(self, save_path):
        """
        Save trained model and scaler
        
        Args:
            save_path: Path to save the model (.pkl file)
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_method': self.feature_method,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'class_names': self.class_names
        }
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {save_path}")
    
    def load_model(self, model_path):
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to the saved model (.pkl file)
        """
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_method = model_data['feature_method']
        self.kernel = model_data['kernel']
        self.C = model_data['C']
        self.gamma = model_data['gamma']
        self.class_names = model_data.get('class_names', None)
        
        print(f"✓ Model loaded successfully")
        print(f"  Feature method: {self.feature_method}")
        print(f"  Kernel: {self.kernel}")


def train_svm_from_features(
    train_features_path,
    train_metadata_path,
    val_features_path,
    val_metadata_path,
    output_model_path,
    feature_method='clip',
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_names=None
):
    """
    Train SVM from extracted features
    
    Args:
        train_features_path: Path to training features .npy
        train_metadata_path: Path to training metadata .json
        val_features_path: Path to validation features .npy
        val_metadata_path: Path to validation metadata .json
        output_model_path: Path to save trained model
        feature_method: Feature extraction method
        kernel: SVM kernel
        C: Regularization parameter
        gamma: Kernel coefficient
        class_names: Optional list of class names
    """
    print(f"\n{'='*60}")
    print(f"SVM Training Pipeline")
    print(f"{'='*60}\n")
    
    # Initialize classifier
    classifier = SVMClassifier(
        feature_method=feature_method,
        kernel=kernel,
        C=C,
        gamma=gamma
    )
    
    # Load training data
    X_train, y_train, train_metadata = classifier.load_features(
        train_features_path,
        train_metadata_path
    )
    
    # Load validation data
    X_val, y_val, val_metadata = classifier.load_features(
        val_features_path,
        val_metadata_path
    )
    
    # Set class names if provided
    if class_names:
        classifier.class_names = class_names
    
    # Train model
    classifier.train(X_train, y_train, normalize=True)
    
    # Evaluate on validation set
    classifier.evaluate(X_val, y_val, normalize=True, class_names=class_names)
    
    # Save model
    classifier.save_model(output_model_path)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train SVM classifier from extracted features')
    
    parser.add_argument(
        '--train_features',
        type=str,
        required=True,
        help='Path to training features .npy file'
    )
    
    parser.add_argument(
        '--train_metadata',
        type=str,
        required=True,
        help='Path to training metadata .json file'
    )
    
    parser.add_argument(
        '--val_features',
        type=str,
        required=True,
        help='Path to validation features .npy file'
    )
    
    parser.add_argument(
        '--val_metadata',
        type=str,
        required=True,
        help='Path to validation metadata .json file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../../models/svm_model.pkl',
        help='Path to save trained model (default: ../../models/svm_model.pkl)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='clip',
        choices=['clip', 'resnet', 'efficientnet', 'vgg16', 'hog'],
        help='Feature extraction method (default: clip)'
    )
    
    parser.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        choices=['linear', 'rbf', 'poly', 'sigmoid'],
        help='SVM kernel (default: rbf)'
    )
    
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='Regularization parameter (default: 1.0)'
    )
    
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Kernel coefficient (default: scale)'
    )
    
    args = parser.parse_args()
    
    # Define class names (Vehicle-10 dataset)
    class_names = [
        'bicycle', 'boat', 'bus', 'car', 'helicopter',
        'minibus', 'motorcycle', 'taxi', 'train', 'truck'
    ]
    
    # Train SVM
    train_svm_from_features(
        train_features_path=args.train_features,
        train_metadata_path=args.train_metadata,
        val_features_path=args.val_features,
        val_metadata_path=args.val_metadata,
        output_model_path=args.output,
        feature_method=args.method,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        class_names=class_names
    )


if __name__ == '__main__':
    main()
