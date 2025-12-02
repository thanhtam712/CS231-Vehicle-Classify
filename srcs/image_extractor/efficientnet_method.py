import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class EfficientNetExtractor:
    def __init__(self, model_name='efficientnet_b0', device=None):
        """
        Initialize EfficientNet feature extractor
        Args:
            model_name: EfficientNet variant (efficientnet_b0 to efficientnet_b7)
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained EfficientNet model
        if model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.feature_dim = 1280
        elif model_name == 'efficientnet_b1':
            self.model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            self.feature_dim = 1280
        elif model_name == 'efficientnet_b2':
            self.model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            self.feature_dim = 1408
        elif model_name == 'efficientnet_b3':
            self.model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.feature_dim = 1536
        elif model_name == 'efficientnet_b4':
            self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            self.feature_dim = 1792
        elif model_name == 'efficientnet_b5':
            self.model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.DEFAULT)
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b6':
            self.model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
            self.feature_dim = 2304
        elif model_name == 'efficientnet_b7':
            self.model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
            self.feature_dim = 2560
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Remove the final classification layer
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"EfficientNet model ({model_name}) loaded on {self.device}")
    
    def extract_features(self, image_path):
        """
        Extract features from an image using EfficientNet
        Args:
            image_path: Path to the image file
        Returns:
            numpy array of features
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze()
            
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def get_feature_dim(self):
        """Return the dimensionality of extracted features"""
        return self.feature_dim
