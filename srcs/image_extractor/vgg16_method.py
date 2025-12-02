import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VGG16Extractor:
    def __init__(self, device=None):
        """
        Initialize VGG16 feature extractor
        Args:
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained VGG16 model
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # Remove the final classification layers, keep features + avgpool + first fc layer
        self.model.classifier = torch.nn.Sequential(*list(self.model.classifier.children())[:-3])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 4096
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"VGG16 model loaded on {self.device}")
    
    def extract_features(self, image_path):
        """
        Extract features from an image using VGG16
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
