"""
Random Forest training script for land cover classification
Alternative to CNN for comparison and faster training
"""

import numpy as np
import cv2
from pathlib import Path
import os
import logging
from typing import Tuple, Optional
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.error("Scikit-learn not available. Install with: pip install scikit-learn")

# EuroSAT classes
EUROSAT_CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

class RandomForestTrainer:
    """
    Random Forest trainer for land cover classification
    Uses spectral indices and color features
    """
    
    def __init__(self, data_dir: str = 'data/eurosat'):
        """
        Initialize Random Forest trainer
        
        Args:
            data_dir: Directory containing EuroSAT dataset
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for Random Forest training")
        
        self.data_dir = Path(data_dir)
        self.class_names = EUROSAT_CLASSES
        self.num_classes = len(EUROSAT_CLASSES)
        self.model = None
        self.scaler = StandardScaler()
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from image for Random Forest
        
        Args:
            image: RGB or multispectral image
            
        Returns:
            Feature vector
        """
        features = []
        
        # Extract RGB bands
        if len(image.shape) == 3:
            r = image[:, :, 0].astype(np.float32)
            g = image[:, :, 1].astype(np.float32)
            b = image[:, :, 2].astype(np.float32)
        else:
            raise ValueError("Expected 3D image")
        
        # Normalize
        if r.max() > 1.0:
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0
        
        # Color statistics
        features.extend([
            np.mean(r), np.std(r),
            np.mean(g), np.std(g),
            np.mean(b), np.std(b)
        ])
        
        # Approximate NIR for indices (if not available)
        nir_approx = g * 1.2
        
        # Calculate spectral indices
        # NDVI
        ndvi = (nir_approx - r) / (nir_approx + r + 1e-10)
        features.extend([np.mean(ndvi), np.std(ndvi), np.min(ndvi), np.max(ndvi)])
        
        # NDWI
        ndwi = (g - nir_approx) / (g + nir_approx + 1e-10)
        features.extend([np.mean(ndwi), np.std(ndwi)])
        
        # NDBI (approximate SWIR as blue)
        ndbi = (b - nir_approx) / (b + nir_approx + 1e-10)
        features.extend([np.mean(ndbi), np.std(ndbi)])
        
        # Convert to HSV
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        features.extend([np.mean(h), np.std(h), np.mean(s), np.std(s), np.mean(v), np.std(v)])
        
        # Texture features
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.extend([np.mean(gradient_magnitude), np.std(gradient_magnitude)])
        
        return np.array(features)
    
    def load_dataset(self, target_size: Tuple[int, int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EuroSAT dataset
        
        Args:
            target_size: Target image size
            
        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels_list = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            logger.info(f"Loading {len(image_files)} images for class: {class_name}")
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, target_size)
                    
                    # Extract features
                    features = self.extract_features(img)
                    features_list.append(features)
                    labels_list.append(class_idx)
                    
                except Exception as e:
                    logger.warning(f"Error loading {img_path}: {e}")
                    continue
        
        return np.array(features_list), np.array(labels_list)
    
    def train(self, n_estimators: int = 100, max_depth: Optional[int] = None,
             min_samples_split: int = 2, random_state: int = 42):
        """
        Train Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            random_state: Random seed
        """
        logger.info("Loading dataset...")
        X, y = self.load_dataset()
        
        if len(X) == 0:
            raise ValueError("No data loaded. Check dataset path.")
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        logger.info("Training Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        logger.info("Evaluating model...")
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        logger.info(f"Training Accuracy: {train_acc:.2%}")
        logger.info(f"Test Accuracy: {test_acc:.2%}")
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, test_pred, target_names=self.class_names))
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': dict(zip(
                [f'feature_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_.tolist()
            ))
        }
    
    def save_model(self, model_path: str = 'models/random_forest_model.pkl',
                  scaler_path: str = 'models/random_forest_scaler.pkl'):
        """Save trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'feature_count': len(self.model.feature_importances_)
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Random Forest land cover classifier')
    parser.add_argument('--data-dir', type=str, default='data/eurosat',
                       help='Path to EuroSAT dataset')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in forest')
    parser.add_argument('--max-depth', type=int, default=None,
                       help='Maximum depth of trees')
    parser.add_argument('--min-samples-split', type=int, default=2,
                       help='Minimum samples to split node')
    
    args = parser.parse_args()
    
    if not SKLEARN_AVAILABLE:
        logger.error("Scikit-learn is required. Install with: pip install scikit-learn")
        return
    
    # Check dataset
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Dataset not found at {args.data_dir}")
        logger.info("Download EuroSAT from: https://github.com/phelber/eurosat")
        return
    
    # Initialize trainer
    trainer = RandomForestTrainer(data_dir=str(data_path))
    
    # Train
    results = trainer.train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    )
    
    # Save model
    trainer.save_model()
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {results['test_accuracy']:.2%}")
    logger.info("Model saved to: models/random_forest_model.pkl")

if __name__ == "__main__":
    main()






