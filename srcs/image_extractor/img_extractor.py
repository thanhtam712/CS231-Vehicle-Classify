import os
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from clip_method import CLIPExtractor
from resnet_method import ResNetExtractor
from efficientnet_method import EfficientNetExtractor
from vgg16_method import VGG16Extractor
from hog_method import HOGExtractor


class ImageFeatureExtractor:
    """
    Main class to extract image features using various methods
    """
    def __init__(self, method='clip', device=None):
        """
        Initialize the feature extractor
        Args:
            method: Feature extraction method ('clip', 'resnet', 'efficientnet', 'vgg16', 'hog')
            device: Device to run deep learning models on (cuda/cpu)
        """
        self.method = method.lower()
        self.device = device
        
        # Initialize the appropriate extractor
        if self.method == 'clip':
            self.extractor = CLIPExtractor(device=device)
        elif self.method == 'resnet':
            self.extractor = ResNetExtractor(model_name='resnet50', device=device)
        elif self.method == 'efficientnet':
            self.extractor = EfficientNetExtractor(model_name='efficientnet_b0', device=device)
        elif self.method == 'vgg16':
            self.extractor = VGG16Extractor(device=device)
        elif self.method == 'hog':
            self.extractor = HOGExtractor()
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: clip, resnet, efficientnet, vgg16, hog")
        
        print(f"Initialized {self.method.upper()} extractor")
        print(f"Feature dimension: {self.extractor.get_feature_dim()}")
    
    def extract_from_directory(self, data_dir, output_dir, metadata_file=None):
        """
        Extract features from all images in a directory structure
        Args:
            data_dir: Root directory containing image folders
            output_dir: Directory to save extracted features
            metadata_file: Optional JSON file with image metadata
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_paths = []
        labels = []
        
        if metadata_file and os.path.exists(metadata_file):
            # Load from metadata file
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # metadata has structure: {"path": [...], "label": [...]}
            paths = metadata['path']
            lbls = metadata['label']
            
            for path, label in zip(paths, lbls):
                img_path = data_dir / path
                if img_path.exists():
                    image_paths.append(str(img_path))
                    labels.append(label)
        else:
            # Scan directory structure
            for class_folder in sorted(data_dir.iterdir()):
                if class_folder.is_dir():
                    class_name = class_folder.name
                    for img_file in sorted(class_folder.glob('*.jpg')):
                        image_paths.append(str(img_file))
                        labels.append(class_name)
        
        print(f"Found {len(image_paths)} images")
        
        # Extract features
        features_list = []
        valid_paths = []
        valid_labels = []
        
        for img_path, label in tqdm(zip(image_paths, labels), total=len(image_paths), 
                                     desc=f"Extracting {self.method} features"):
            features = self.extractor.extract_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_paths.append(img_path)
                valid_labels.append(label)
        
        # Convert to numpy array
        features_array = np.array(features_list)
        
        # Save features
        output_name = f"features_{self.method}"
        np.save(output_dir / f"{output_name}.npy", features_array)
        
        # Save metadata
        metadata = {
            'method': self.method,
            'feature_dim': self.extractor.get_feature_dim(),
            'num_samples': len(features_list),
            'image_paths': valid_paths,
            'labels': valid_labels
        }
        
        with open(output_dir / f"{output_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nFeatures saved to: {output_dir / output_name}.npy")
        print(f"Feature shape: {features_array.shape}")
        print(f"Metadata saved to: {output_dir / output_name}_metadata.json")
        
        return features_array, metadata
    
    def extract_single_image(self, image_path):
        """
        Extract features from a single image
        Args:
            image_path: Path to the image
        Returns:
            numpy array of features
        """
        return self.extractor.extract_features(image_path)


def main():
    parser = argparse.ArgumentParser(description='Extract image features using various methods')
    parser.add_argument('--method', type=str, default='clip',
                       choices=['clip', 'resnet', 'efficientnet', 'vgg16', 'hog'],
                       help='Feature extraction method')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='./features',
                       help='Directory to save extracted features')
    parser.add_argument('--metadata', type=str, default=None,
                       help='Optional metadata JSON file')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to run model on (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = ImageFeatureExtractor(method=args.method, device=args.device)
    
    # Extract features
    extractor.extract_from_directory(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata
    )


if __name__ == '__main__':
    main()
