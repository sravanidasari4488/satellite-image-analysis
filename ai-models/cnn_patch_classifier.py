"""
CNN Patch Classifier
Uses trained EfficientNet model for 64x64 patch classification

Scientific Rationale:
- EuroSAT model trained specifically on 64x64 Sentinel-2 patches
- Direct application to full images violates training distribution
- Patch-level classification maintains model accuracy
- Batch processing improves efficiency
"""

import numpy as np
import os
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. CNN classification disabled.")

class CNNPatchClassifier:
    """
    Classifies 64x64 patches using trained EuroSAT EfficientNet model
    
    Model expects:
    - Input: (64, 64, 3) RGB image, normalized [0, 1]
    - Output: (10,) softmax probabilities for EuroSAT classes
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: Path to trained model (.h5 file)
        """
        self.model = None
        self.model_path = model_path or 'models/multispectral_landcover_model.h5'
        self.eurosat_classes = [
            'AnnualCrop',      # 0
            'Forest',          # 1
            'HerbaceousVegetation',  # 2
            'Highway',         # 3
            'Industrial',      # 4
            'Pasture',         # 5
            'PermanentCrop',   # 6
            'Residential',     # 7
            'River',           # 8
            'SeaLake'          # 9
        ]
        self.load_model()
    
    def load_model(self):
        """Load trained EfficientNet model"""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available. Cannot load CNN model.")
            return
        
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
                return
            
            self.model = load_model(self.model_path)
            logger.info(f"Loaded CNN model from {self.model_path}")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and TENSORFLOW_AVAILABLE
    
    def classify_patch(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Classify a single 64x64 patch
        
        Args:
            patch: RGB image patch (64, 64, 3) or (64, 64) grayscale
        
        Returns:
            Dictionary with EuroSAT class probabilities
        """
        if not self.is_available():
            raise RuntimeError("CNN model not available")
        
        # Ensure correct shape
        if len(patch.shape) == 2:
            # Grayscale -> RGB (duplicate channels)
            patch = np.stack([patch] * 3, axis=2)
        elif patch.shape[2] == 1:
            patch = np.repeat(patch, 3, axis=2)
        elif patch.shape[2] != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {patch.shape[2]}")
        
        # Resize if needed (should be 64x64, but handle edge cases)
        if patch.shape[0] != 64 or patch.shape[1] != 64:
            import cv2
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        if patch.max() > 1.0:
            patch = patch.astype(np.float32) / 255.0
        
        # Reshape for model input: (1, 64, 64, 3)
        patch_batch = np.expand_dims(patch, axis=0)
        
        # Predict
        try:
            predictions = self.model.predict(patch_batch, verbose=0)[0]
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise
        
        # Convert to dictionary
        result = {
            class_name: float(prob)
            for class_name, prob in zip(self.eurosat_classes, predictions)
        }
        
        return result
    
    def classify_batch(self, patches: List[np.ndarray], 
                      batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Classify multiple patches efficiently using batch processing
        
        Args:
            patches: List of patch arrays
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        if not self.is_available():
            raise RuntimeError("CNN model not available")
        
        if not patches:
            return []
        
        # Preprocess all patches
        processed_patches = []
        for patch in patches:
            # Ensure correct shape and normalization
            if len(patch.shape) == 2:
                patch = np.stack([patch] * 3, axis=2)
            elif patch.shape[2] == 1:
                patch = np.repeat(patch, 3, axis=2)
            
            if patch.shape[0] != 64 or patch.shape[1] != 64:
                import cv2
                patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
            
            if patch.max() > 1.0:
                patch = patch.astype(np.float32) / 255.0
            
            processed_patches.append(patch)
        
        # Convert to batch array
        batch_array = np.array(processed_patches)
        
        # Batch prediction
        try:
            predictions = self.model.predict(batch_array, batch_size=batch_size, verbose=0)
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
        
        # Convert to list of dictionaries
        results = []
        for pred in predictions:
            results.append({
                class_name: float(prob)
                for class_name, prob in zip(self.eurosat_classes, pred)
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if not self.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'model_path': self.model_path,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.eurosat_classes),
            'classes': self.eurosat_classes
        }


