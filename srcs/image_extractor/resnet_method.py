import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ResNetExtractor:
    def __init__(self, model_name='resnet50', device=None):
        """
        Initialize ResNet feature extractor
        Args:
            model_name: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained ResNet model
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.feature_dim = 2048
        elif model_name == 'resnet152':
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Remove the final classification layer
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
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
        
        print(f"ResNet model ({model_name}) loaded on {self.device}")
    
    def extract_features(self, image_path):
        """
        Extract features from an image using ResNet
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
