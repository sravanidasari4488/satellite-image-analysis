"""
Multispectral Analysis Module for Sentinel-2 Satellite Imagery
Implements ML-driven methods using supervised learning on multispectral bands
Supports NDVI, NDWI, NDBI calculations and 10-class land cover classification
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ML libraries
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Some ML features will be limited.")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Random Forest classifier will not be available.")

# EuroSAT 10-class land cover classification
EUROSAT_CLASSES = [
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

class MultispectralAnalyzer:
    """
    Advanced multispectral analysis using Sentinel-2 bands
    Supports supervised learning with CNNs and Random Forests
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = 'cnn'):
        """
        Initialize multispectral analyzer
        
        Args:
            model_path: Path to trained model file
            model_type: 'cnn' or 'random_forest'
        """
        self.model = None
        self.model_type = model_type
        self.class_names = EUROSAT_CLASSES
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        # Load model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Try default location
            default_path = 'models/multispectral_landcover_model.h5'
            if os.path.exists(default_path):
                self.load_model(default_path)
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI)
        
        Args:
            red_band: Red band (Band 4 in Sentinel-2)
            nir_band: Near Infrared band (Band 8 in Sentinel-2)
            
        Returns:
            NDVI array (-1 to 1)
        """
        red = red_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Normalize if needed (assuming 0-255 or 0-10000 range)
        if red.max() > 1.0:
            if red.max() > 1000:
                red = red / 10000.0  # Sentinel-2 scaling
            else:
                red = red / 255.0
        
        if nir.max() > 1.0:
            if nir.max() > 1000:
                nir = nir / 10000.0  # Sentinel-2 scaling
            else:
                nir = nir / 255.0
        
        # NDVI = (NIR - Red) / (NIR + Red)
        denominator = nir + red + 1e-10
        ndvi = (nir - red) / denominator
        
        return np.clip(ndvi, -1, 1)
    
    def calculate_ndwi(self, green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index (NDWI)
        Used for water body detection
        
        Args:
            green_band: Green band (Band 3 in Sentinel-2)
            nir_band: Near Infrared band (Band 8 in Sentinel-2)
            
        Returns:
            NDWI array (-1 to 1)
        """
        green = green_band.astype(np.float32)
        nir = nir_band.astype(np.float32)
        
        # Normalize if needed
        if green.max() > 1.0:
            if green.max() > 1000:
                green = green / 10000.0
            else:
                green = green / 255.0
        
        if nir.max() > 1.0:
            if nir.max() > 1000:
                nir = nir / 10000.0
            else:
                nir = nir / 255.0
        
        # NDWI = (Green - NIR) / (Green + NIR)
        denominator = green + nir + 1e-10
        ndwi = (green - nir) / denominator
        
        return np.clip(ndwi, -1, 1)
    
    def calculate_ndbi(self, nir_band: np.ndarray, swir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Built-up Index (NDBI)
        Used for urban/built-up area detection
        
        Args:
            nir_band: Near Infrared band (Band 8 in Sentinel-2)
            swir_band: Shortwave Infrared band (Band 11 in Sentinel-2)
            
        Returns:
            NDBI array (-1 to 1)
        """
        nir = nir_band.astype(np.float32)
        swir = swir_band.astype(np.float32)
        
        # Normalize if needed
        if nir.max() > 1.0:
            if nir.max() > 1000:
                nir = nir / 10000.0
            else:
                nir = nir / 255.0
        
        if swir.max() > 1.0:
            if swir.max() > 1000:
                swir = swir / 10000.0
            else:
                swir = swir / 255.0
        
        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        denominator = swir + nir + 1e-10
        ndbi = (swir - nir) / denominator
        
        return np.clip(ndbi, -1, 1)
    
    def calculate_indices_from_rgb_nir(self, rgb_image: np.ndarray, nir_band: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate all spectral indices from RGB image and optional NIR band
        
        Args:
            rgb_image: RGB image (H, W, 3)
            nir_band: Optional NIR band (H, W)
            
        Returns:
            Dictionary with NDVI, NDWI, NDBI arrays
        """
        r = rgb_image[:, :, 0]
        g = rgb_image[:, :, 1]
        b = rgb_image[:, :, 2]
        
        indices = {}
        
        if nir_band is not None:
            # Use real NIR band
            indices['ndvi'] = self.calculate_ndvi(r, nir_band)
            indices['ndwi'] = self.calculate_ndwi(g, nir_band)
            # For NDBI, we need SWIR - approximate using blue band
            indices['ndbi'] = self.calculate_ndbi(nir_band, b)
        else:
            # Approximate NIR from RGB using vegetation index
            # This is less accurate but works when NIR is not available
            nir_approx = g * 1.2  # Approximate NIR as enhanced green
            
            indices['ndvi'] = self.calculate_ndvi(r, nir_approx)
            indices['ndwi'] = self.calculate_ndwi(g, nir_approx)
            indices['ndbi'] = self.calculate_ndbi(nir_approx, b)
        
        return indices
    
    def segment_landcover(self, multispectral_image: np.ndarray, 
                         use_indices: bool = True,
                         use_model: bool = True) -> Dict[str, any]:
        """
        Segment land cover using multispectral bands and spectral indices
        Uses trained ML model if available, otherwise falls back to rule-based segmentation
        
        Args:
            multispectral_image: Multispectral image (H, W, bands)
                                 Can be RGB (3 bands) or RGB+NIR (4 bands)
            use_indices: Whether to use spectral indices for segmentation
            use_model: Whether to use trained ML model if available
            
        Returns:
            Dictionary with land cover segmentation results
        """
        if len(multispectral_image.shape) != 3:
            raise ValueError(f"Expected 3D image, got shape: {multispectral_image.shape}")
        
        height, width = multispectral_image.shape[:2]
        num_bands = multispectral_image.shape[2]
        
        # Extract bands
        if num_bands >= 3:
            r = multispectral_image[:, :, 0]
            g = multispectral_image[:, :, 1]
            b = multispectral_image[:, :, 2]
        else:
            raise ValueError("At least 3 bands (RGB) required")
        
        nir = None
        if num_bands >= 4:
            nir = multispectral_image[:, :, 3]
        
        # Calculate spectral indices
        indices = self.calculate_indices_from_rgb_nir(
            multispectral_image[:, :, :3], nir
        )
        
        # Try to use trained model if available
        segmentation = None
        method_used = "rule_based"
        
        if use_model and self.model is not None and TENSORFLOW_AVAILABLE:
            try:
                # Use ML model for segmentation
                segmentation = self._segment_with_model(multispectral_image)
                method_used = "ml_model"
                logger.info("Using trained ML model for segmentation")
            except Exception as e:
                logger.warning(f"ML model segmentation failed, falling back to rule-based: {e}")
                segmentation = None
        
        # Fall back to rule-based segmentation
        if segmentation is None:
            segmentation = self._segment_with_indices(r, g, b, indices, nir)
        
        return {
            'segmentation': segmentation,
            'indices': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in indices.items()},
            'statistics': self._calculate_segmentation_stats(segmentation, indices),
            'method': method_used
        }
    
    def _segment_with_indices(self, r: np.ndarray, g: np.ndarray, b: np.ndarray,
                              indices: Dict[str, np.ndarray], nir: Optional[np.ndarray]) -> np.ndarray:
        """
        Segment land cover using spectral indices
        
        Returns:
            Segmentation map with class labels (0-9 for EuroSAT classes)
        """
        height, width = r.shape
        segmentation = np.zeros((height, width), dtype=np.int32)
        
        ndvi = indices['ndvi']
        ndwi = indices['ndwi']
        ndbi = indices['ndbi']
        
        # Class 8: River / Class 9: SeaLake (Water bodies)
        water_mask = ndwi > 0.3
        
        # Class 1: Forest (High NDVI, green)
        forest_mask = (ndvi > 0.4) & (g > r * 1.1) & ~water_mask
        
        # Class 2: HerbaceousVegetation (Medium NDVI)
        herbaceous_mask = (ndvi > 0.2) & (ndvi <= 0.4) & ~water_mask & ~forest_mask
        
        # Class 5: Pasture (Low-medium NDVI, bright)
        pasture_mask = (ndvi > 0.1) & (ndvi <= 0.3) & (g > 100) & ~water_mask & ~forest_mask & ~herbaceous_mask
        
        # Class 0: AnnualCrop / Class 6: PermanentCrop
        crop_mask = (ndvi > 0.15) & (ndvi <= 0.35) & ~water_mask & ~forest_mask & ~herbaceous_mask & ~pasture_mask
        
        # Class 7: Residential / Class 4: Industrial (Urban - High NDBI)
        urban_mask = (ndbi > 0.1) & ~water_mask
        
        # Class 3: Highway (Linear features, low NDVI, high brightness)
        highway_mask = (ndvi < 0.1) & (r + g + b > 200) & ~water_mask & ~urban_mask
        
        # Assign classes
        segmentation[water_mask] = 9  # SeaLake (can be refined to River=8)
        segmentation[forest_mask] = 1  # Forest
        segmentation[herbaceous_mask] = 2  # HerbaceousVegetation
        segmentation[pasture_mask] = 5  # Pasture
        segmentation[crop_mask] = 0  # AnnualCrop (default, can be refined)
        segmentation[urban_mask] = 7  # Residential (default, can be refined to Industrial=4)
        segmentation[highway_mask] = 3  # Highway
        
        # Remaining pixels: Barren/Unknown (assign to closest class)
        unclassified = segmentation == 0
        if np.any(unclassified):
            # Use K-means or assign based on dominant index
            unclassified_ndvi = ndvi[unclassified]
            unclassified_ndbi = ndbi[unclassified]
            
            # Assign based on dominant characteristics
            low_veg = unclassified_ndvi < 0.1
            high_built = unclassified_ndbi > 0.05
            
            unclassified_indices = np.where(unclassified)
            segmentation[unclassified_indices[0][low_veg & ~high_built]] = 0  # AnnualCrop/Barren
            segmentation[unclassified_indices[0][high_built]] = 4  # Industrial
        
        return segmentation
    
    def _calculate_segmentation_stats(self, segmentation: np.ndarray, 
                                     indices: Dict[str, np.ndarray]) -> Dict[str, any]:
        """Calculate statistics for segmentation"""
        stats = {}
        
        for i, class_name in enumerate(self.class_names):
            mask = segmentation == i
            count = np.sum(mask)
            percentage = (count / segmentation.size) * 100 if segmentation.size > 0 else 0
            
            if count > 0:
                stats[class_name] = {
                    'pixel_count': int(count),
                    'percentage': float(percentage),
                    'mean_ndvi': float(np.mean(indices['ndvi'][mask])) if 'ndvi' in indices else 0.0,
                    'mean_ndwi': float(np.mean(indices['ndwi'][mask])) if 'ndwi' in indices else 0.0,
                    'mean_ndbi': float(np.mean(indices['ndbi'][mask])) if 'ndbi' in indices else 0.0
                }
        
        return stats
    
    def assess_flood_risk(self, ndwi: np.ndarray, ndvi: np.ndarray, 
                          elevation_data: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Assess flood risk using NDWI and elevation data
        
        Args:
            ndwi: Normalized Difference Water Index
            ndvi: Normalized Difference Vegetation Index
            elevation_data: Optional elevation data
            
        Returns:
            Flood risk assessment dictionary
        """
        # High NDWI indicates water presence
        water_presence = np.mean(ndwi > 0.3)
        
        # Low elevation + high water = flood risk
        risk_factors = []
        risk_score = 0.0
        
        if water_presence > 0.2:
            risk_factors.append('High water presence detected')
            risk_score += 0.4
        
        if elevation_data is not None:
            low_elevation = np.mean(elevation_data < np.percentile(elevation_data, 25))
            if low_elevation > 0.3:
                risk_factors.append('Low elevation area')
                risk_score += 0.3
        
        # Vegetation loss can indicate flooding
        low_vegetation = np.mean(ndvi < 0.1)
        if low_vegetation > 0.4 and water_presence > 0.15:
            risk_factors.append('Potential vegetation loss due to flooding')
            risk_score += 0.3
        
        # Determine risk level
        if risk_score >= 0.7:
            level = 'high'
        elif risk_score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'probability': float(risk_score),
            'factors': risk_factors,
            'water_presence': float(water_presence),
            'recommendations': self._get_flood_recommendations(level)
        }
    
    def assess_drought_risk(self, ndvi: np.ndarray, ndwi: np.ndarray,
                           precipitation_data: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Assess drought risk using NDVI and precipitation data
        
        Args:
            ndvi: Normalized Difference Vegetation Index
            ndwi: Normalized Difference Water Index
            precipitation_data: Optional precipitation data
            
        Returns:
            Drought risk assessment dictionary
        """
        mean_ndvi = np.mean(ndvi)
        low_vegetation = np.mean(ndvi < 0.2)
        low_water = np.mean(ndwi < 0.1)
        
        risk_factors = []
        risk_score = 0.0
        
        if mean_ndvi < 0.2:
            risk_factors.append('Low vegetation index')
            risk_score += 0.4
        
        if low_vegetation > 0.5:
            risk_factors.append('High percentage of low vegetation areas')
            risk_score += 0.3
        
        if low_water > 0.7:
            risk_factors.append('Low water presence')
            risk_score += 0.3
        
        if precipitation_data is not None:
            low_precipitation = np.mean(precipitation_data < np.percentile(precipitation_data, 25))
            if low_precipitation > 0.4:
                risk_factors.append('Low precipitation levels')
                risk_score += 0.2
        
        # Determine risk level
        if risk_score >= 0.7:
            level = 'high'
        elif risk_score >= 0.4:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'probability': float(risk_score),
            'factors': risk_factors,
            'mean_ndvi': float(mean_ndvi),
            'vegetation_health': 'critical' if mean_ndvi < 0.1 else 'poor' if mean_ndvi < 0.2 else 'fair',
            'recommendations': self._get_drought_recommendations(level)
        }
    
    def _get_flood_recommendations(self, level: str) -> List[str]:
        """Get flood risk recommendations"""
        recommendations = {
            'high': [
                'Immediate evacuation may be necessary',
                'Monitor water levels closely',
                'Prepare emergency supplies',
                'Check local flood warnings'
            ],
            'medium': [
                'Monitor weather conditions',
                'Prepare evacuation plan',
                'Secure important documents',
                'Stay informed about flood alerts'
            ],
            'low': [
                'Normal conditions',
                'Continue regular monitoring'
            ]
        }
        return recommendations.get(level, [])
    
    def _get_drought_recommendations(self, level: str) -> List[str]:
        """Get drought risk recommendations"""
        recommendations = {
            'high': [
                'Implement water conservation measures',
                'Monitor soil moisture levels',
                'Consider irrigation if available',
                'Prepare for crop yield reduction'
            ],
            'medium': [
                'Monitor vegetation health',
                'Implement water-saving practices',
                'Check water availability',
                'Monitor weather forecasts'
            ],
            'low': [
                'Normal conditions',
                'Continue regular monitoring'
            ]
        }
        return recommendations.get(level, [])
    
    def _segment_with_model(self, multispectral_image: np.ndarray) -> np.ndarray:
        """
        Segment image using trained ML model (sliding window approach)
        
        Args:
            multispectral_image: Multispectral image (H, W, bands)
            
        Returns:
            Segmentation map with class labels (0-9)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        height, width = multispectral_image.shape[:2]
        input_size = self.model.input_shape[1:3]  # Expected input size (e.g., 64x64)
        
        # For large images, use sliding window approach
        # For now, resize to model input size and predict
        # In production, you'd want to use a sliding window with overlap
        
        # Resize image to model input size
        resized = cv2.resize(multispectral_image, input_size)
        
        # Normalize
        if resized.max() > 1.0:
            resized = resized / 255.0
        
        # Predict
        processed = np.expand_dims(resized, axis=0)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get predicted class
        predicted_class = np.argmax(predictions)
        
        # Create segmentation map (all pixels get the same class for now)
        # For full segmentation, you'd need to process patches
        segmentation = np.full((height, width), predicted_class, dtype=np.int32)
        
        return segmentation
    
    def load_model(self, model_path: str):
        """Load trained model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot load CNN model.")
            return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded multispectral model from {model_path}")
        except Exception as e:
            # Model shape mismatch is expected if model was trained with different input channels
            # This is a fallback model - not critical for city-level GEE analysis
            if "Shape mismatch" in str(e) or "stem_conv" in str(e):
                logger.warning(f"Multispectral model shape mismatch (expected for fallback): {e}")
                logger.info("   → This is a fallback model. City-level GEE analysis will work without it.")
            else:
                logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def predict_with_model(self, multispectral_image: np.ndarray) -> Dict[str, float]:
        """
        Predict land cover classes using trained model
        
        Args:
            multispectral_image: Multispectral image (H, W, bands)
            
        Returns:
            Dictionary with class probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Preprocess image for model input
        if len(multispectral_image.shape) == 3:
            # Resize to model input size (typically 64x64 or 256x256)
            input_size = self.model.input_shape[1:3]
            processed = cv2.resize(multispectral_image, input_size)
            processed = np.expand_dims(processed, axis=0)
        
        # Normalize
        if processed.max() > 1.0:
            processed = processed / 255.0
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Map to class names
        result = {}
        for i, class_name in enumerate(self.class_names):
            result[class_name] = float(predictions[i])
        
        return result

