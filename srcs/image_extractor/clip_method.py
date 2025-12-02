import torch
import clip
from PIL import Image
import numpy as np

class CLIPExtractor:
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Initialize CLIP feature extractor
        Args:
            model_name: CLIP model variant (default: ViT-B/32)
            device: Device to run model on (cuda/cpu)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        print(f"CLIP model loaded on {self.device}")
    
    def extract_features(self, image_path):
        """
        Extract features from an image using CLIP
        Args:
            image_path: Path to the image file
        Returns:
            numpy array of features
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def get_feature_dim(self):
        """Return the dimensionality of extracted features"""
        return 512  # CLIP ViT-B/32 outputs 512-dim features
