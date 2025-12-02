import cv2
import numpy as np
from skimage.feature import hog
from skimage import io, color

class HOGExtractor:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), 
                 cells_per_block=(2, 2), resize_shape=(128, 128)):
        """
        Initialize HOG feature extractor
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell
            cells_per_block: Number of cells in each block
            resize_shape: Target size to resize images
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.resize_shape = resize_shape
        
        # Calculate feature dimension
        # For 128x128 image with (8,8) cells and (2,2) blocks
        # cells: 16x16, blocks: 15x15, features: 15*15*2*2*9 = 8100
        cells_x = self.resize_shape[0] // self.pixels_per_cell[0]
        cells_y = self.resize_shape[1] // self.pixels_per_cell[1]
        blocks_x = cells_x - self.cells_per_block[0] + 1
        blocks_y = cells_y - self.cells_per_block[1] + 1
        self.feature_dim = (blocks_x * blocks_y * 
                           self.cells_per_block[0] * self.cells_per_block[1] * 
                           self.orientations)
        
        print(f"HOG extractor initialized with feature dimension: {self.feature_dim}")
    
    def extract_features(self, image_path):
        """
        Extract HOG features from an image
        Args:
            image_path: Path to the image file
        Returns:
            numpy array of HOG features
        """
        try:
            # Read image
            image = io.imread(image_path)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = color.rgb2gray(image)
            
            # Resize image
            image = cv2.resize(image, self.resize_shape)
            
            # Extract HOG features
            features = hog(image, 
                          orientations=self.orientations,
                          pixels_per_cell=self.pixels_per_cell,
                          cells_per_block=self.cells_per_block,
                          block_norm='L2-Hys',
                          visualize=False,
                          feature_vector=True)
            
            return features
        except Exception as e:
            print(f"Error extracting HOG features from {image_path}: {str(e)}")
            return None
    
    def get_feature_dim(self):
        """Return the dimensionality of extracted features"""
        return self.feature_dim
