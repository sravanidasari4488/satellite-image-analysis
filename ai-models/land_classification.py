"""
Satellite Image Land Classification Model
This module provides AI-powered land classification for satellite images.
Uses pixel-based analysis with color segmentation and NDVI for accurate classification.
"""

import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional ML libraries - fallback to pixel-based analysis if not available
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Using pixel-based analysis only.")

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("Rasterio not available. Some features may be limited.")

class LandClassificationModel:
    """
    AI model for classifying land types in satellite images.
    Uses pixel-based color analysis and segmentation for accurate classification.
    Supports classification of: Forest, Water, Urban, Agricultural, Barren
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.class_names = ['Forest', 'Water', 'Urban', 'Agricultural', 'Barren']
        self.class_colors = {
            'Forest': [34, 139, 34],      # Forest Green
            'Water': [0, 100, 200],       # Blue
            'Urban': [128, 128, 128],     # Gray
            'Agricultural': [255, 215, 0], # Gold
            'Barren': [139, 69, 19]       # Brown
        }
        
        # Try to load trained model, fallback to pixel-based analysis
        if model_path:
            self.load_model(model_path)
        else:
            # Try to load from default location
            default_path = 'models/trained_land_classification.h5'
            if os.path.exists(default_path):
                try:
                    self.load_model(default_path)
                    logger.info("Loaded trained model for enhanced precision")
                except Exception as e:
                    logger.warning(f"Could not load trained model: {e}. Using pixel-based analysis.")
                    logger.info("Using pixel-based land classification (no model training required)")
            else:
                logger.info("Using pixel-based land classification. Train model for enhanced precision.")
    
    def create_model(self):
        """Create a CNN model for land classification (optional, not used in production)"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create CNN model.")
            return
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(5, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Land classification model created successfully")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot load trained model. Using pixel-based analysis.")
            return
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def classify_image(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classify land types in satellite image using trained ML model + pixel-based analysis
        Hybrid approach for maximum precision.
        
        Args:
            image: Input satellite image (RGB)
            
        Returns:
            Dictionary with land type percentages based on actual pixel analysis
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")
        
        # Ensure image is in correct format
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape: {image.shape}")
        
        # Convert to RGB if needed
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # If trained model is available, use it for enhanced precision
        if self.model is not None:
            try:
                return self._classify_with_model(image)
            except Exception as e:
                logger.warning(f"Model classification failed: {e}. Falling back to pixel-based analysis.")
        
        # Fallback to pixel-based analysis
        return self._classify_with_pixels(image)
    
    def _classify_with_model(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classify using trained ML model for maximum precision
        """
        original_shape = image.shape[:2]
        
        # Resize to model input size
        model_input_size = (256, 256)
        if self.model.input_shape[1:3] != (None, None):
            model_input_size = (self.model.input_shape[1], self.model.input_shape[2])
        
        image_resized = cv2.resize(image, model_input_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Get prediction
        predictions = self.model.predict(image_batch, verbose=0)[0]
        
        # Convert to percentages
        result = {}
        for i, class_name in enumerate(self.class_names):
            result[class_name.lower()] = float(predictions[i] * 100)
        
        # Refine with pixel-based analysis for better accuracy
        pixel_result = self._classify_with_pixels(image)
        
        # Weighted combination: 70% model, 30% pixel-based
        final_result = {}
        for class_name in self.class_names:
            key = class_name.lower()
            final_result[key] = float(
                predictions[self.class_names.index(class_name)] * 100 * 0.7 +
                pixel_result.get(key, 0) * 0.3
            )
        
        # Normalize to 100%
        total = sum(final_result.values())
        if total > 0:
            for key in final_result:
                final_result[key] = (final_result[key] / total) * 100
        
        return final_result
    
    def _classify_with_pixels(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classify using pixel-based color analysis (enhanced version)
        """
        height, width = image.shape[:2]
        total_pixels = height * width
        
        # Convert to different color spaces for better analysis
        rgb_image = image.copy()
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        
        # Initialize classification masks
        forest_mask = np.zeros((height, width), dtype=bool)
        water_mask = np.zeros((height, width), dtype=bool)
        urban_mask = np.zeros((height, width), dtype=bool)
        agricultural_mask = np.zeros((height, width), dtype=bool)
        barren_mask = np.zeros((height, width), dtype=bool)
        
        # Extract color channels
        r = rgb_image[:, :, 0].astype(np.float32)
        g = rgb_image[:, :, 1].astype(np.float32)
        b = rgb_image[:, :, 2].astype(np.float32)
        h = hsv_image[:, :, 0].astype(np.float32)
        s = hsv_image[:, :, 1].astype(np.float32)
        v = hsv_image[:, :, 2].astype(np.float32)
        
        # Calculate vegetation indices (approximate NDVI from RGB)
        # Using Visible Vegetation Index (VVI) as approximation
        vvi = ((g - r) / (g + r + 1e-10))
        vvi = np.clip(vvi, -1, 1)
        
        # WATER DETECTION: Blue/cyan colors with low saturation
        # Water typically has: high blue, low red, medium-low saturation
        water_condition = (
            (b > r * 1.2) &  # More blue than red
            (b > g * 1.1) &  # More blue than green
            (s < 100) &      # Low saturation
            (v > 50)         # Not too dark
        )
        water_mask = water_condition
        
        # FOREST DETECTION: Green colors with high vegetation index
        # Forest typically has: high green, high VVI, medium-high saturation
        forest_condition = (
            (vvi > 0.15) &           # Positive vegetation index
            (g > r * 1.1) &         # More green than red
            (g > b * 1.1) &         # More green than blue
            (s > 40) &              # Medium saturation
            (h > 40) & (h < 100)    # Green hue range
        )
        forest_mask = forest_condition & ~water_mask
        
        # AGRICULTURAL DETECTION: Light green/yellow colors
        # Agricultural areas: medium green, medium VVI, medium saturation
        agricultural_condition = (
            (vvi > 0.05) & (vvi <= 0.25) &  # Moderate vegetation
            (g > r * 1.05) &                # Slightly more green
            (s > 30) & (s < 80) &           # Medium saturation
            (v > 100)                       # Not too dark (cultivated areas are brighter)
        )
        agricultural_mask = agricultural_condition & ~water_mask & ~forest_mask
        
        # URBAN DETECTION: Gray, brown, or high brightness areas
        # Urban areas: low saturation, medium-high brightness, grayish
        urban_condition = (
            (s < 50) &                      # Low saturation (gray)
            (v > 80) &                      # Medium-high brightness
            (np.abs(r - g) < 30) &          # Similar red and green (gray)
            (np.abs(g - b) < 30)            # Similar green and blue (gray)
        ) | (
            (r > 100) & (g > 80) & (b < 100) &  # Brownish (buildings, roads)
            (s < 60)
        )
        urban_mask = urban_condition & ~water_mask
        
        # BARREN DETECTION: Brown, tan, or low vegetation areas
        # Barren areas: low vegetation index, brown/tan colors
        barren_condition = (
            (vvi < 0.05) &                   # Low/no vegetation
            (s < 60) &                       # Low-medium saturation
            (r > g * 0.9) & (r > b * 0.9)   # Reddish/brownish
        ) | (
            (vvi < -0.1) &                   # Negative vegetation (soil, sand)
            (s < 40)
        )
        barren_mask = barren_condition & ~water_mask & ~urban_mask & ~agricultural_mask
        
        # Assign remaining pixels to the most likely category
        unclassified = ~(water_mask | forest_mask | urban_mask | agricultural_mask | barren_mask)
        if np.any(unclassified):
            # Use K-means clustering for remaining pixels
            unclassified_pixels = rgb_image[unclassified].reshape(-1, 3)
            if len(unclassified_pixels) > 100:
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(unclassified_pixels)
                
                # Assign clusters based on centroid colors
                for i, centroid in enumerate(kmeans.cluster_centers_):
                    cluster_mask = (clusters == i)
                    pixel_indices = np.where(unclassified)[0][cluster_mask]
                    
                    # Determine category based on centroid color
                    if centroid[2] > centroid[0] * 1.2:  # More blue
                        water_mask.flat[pixel_indices] = True
                    elif centroid[1] > centroid[0] * 1.1:  # More green
                        if np.mean(vvi.flat[pixel_indices]) > 0.1:
                            forest_mask.flat[pixel_indices] = True
                        else:
                            agricultural_mask.flat[pixel_indices] = True
                    elif np.std(centroid) < 20:  # Gray
                        urban_mask.flat[pixel_indices] = True
                    else:
                        barren_mask.flat[pixel_indices] = True
            else:
                # If too few pixels, assign to barren
                barren_mask[unclassified] = True
        
        # Calculate percentages
        forest_pixels = np.sum(forest_mask)
        water_pixels = np.sum(water_mask)
        urban_pixels = np.sum(urban_mask)
        agricultural_pixels = np.sum(agricultural_mask)
        barren_pixels = np.sum(barren_mask)
        
        # Normalize to ensure total is 100%
        total_classified = forest_pixels + water_pixels + urban_pixels + agricultural_pixels + barren_pixels
        if total_classified > 0:
            forest_pct = (forest_pixels / total_classified) * 100
            water_pct = (water_pixels / total_classified) * 100
            urban_pct = (urban_pixels / total_classified) * 100
            agricultural_pct = (agricultural_pixels / total_classified) * 100
            barren_pct = (barren_pixels / total_classified) * 100
        else:
            # Fallback if no pixels classified
            forest_pct = water_pct = urban_pct = agricultural_pct = barren_pct = 20.0
        
        return {
            'forest': float(forest_pct),
            'water': float(water_pct),
            'urban': float(urban_pct),
            'agricultural': float(agricultural_pct),
            'barren': float(barren_pct)
        }
    
    def extract_advanced_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract advanced features for enhanced classification precision
        """
        features = []
        
        # Color statistics
        rgb_mean = np.mean(image, axis=(0, 1))
        rgb_std = np.std(image, axis=(0, 1))
        features.extend(rgb_mean)
        features.extend(rgb_std)
        
        # Color space features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        hsv_mean = np.mean(hsv, axis=(0, 1))
        lab_mean = np.mean(lab, axis=(0, 1))
        features.extend(hsv_mean)
        features.extend(lab_mean)
        
        # Texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # Vegetation index
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        vvi = np.mean((g - r) / (g + r + 1e-10))
        features.append(vvi)
        
        # Local texture (using GLCM-like features)
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        features.append(np.mean(local_var))
        
        return np.array(features)
    
    def segment_image(self, image: np.ndarray, num_clusters: int = 5) -> np.ndarray:
        """
        Segment image using K-means clustering
        
        Args:
            image: Input satellite image
            num_clusters: Number of clusters for segmentation
            
        Returns:
            Segmented image
        """
        # Reshape image for clustering
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Reshape labels back to image shape
        segmented = labels.reshape(image.shape[:2])
        
        return segmented
    
    def calculate_ndvi(self, red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index (NDVI)
        
        Args:
            red_band: Red band of satellite image
            nir_band: Near-infrared band of satellite image
            
        Returns:
            NDVI values (-1 to 1)
        """
        # Normalize bands if needed
        if red_band.max() > 1.0:
            red_band = red_band.astype(np.float32) / 255.0
        if nir_band.max() > 1.0:
            nir_band = nir_band.astype(np.float32) / 255.0
        
        # Avoid division by zero
        denominator = nir_band.astype(np.float32) + red_band.astype(np.float32)
        denominator[denominator == 0] = 1e-10
        
        ndvi = (nir_band.astype(np.float32) - red_band.astype(np.float32)) / denominator
        
        # Clip values to valid range
        ndvi = np.clip(ndvi, -1, 1)
        
        return ndvi
    
    def calculate_ndvi_from_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate approximate NDVI from RGB image using Visible Vegetation Index
        This is used when NIR band is not available
        
        Args:
            image: RGB satellite image
            
        Returns:
            Approximate NDVI values
        """
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        
        # Normalize
        if r.max() > 1.0:
            r = r / 255.0
        if g.max() > 1.0:
            g = g / 255.0
        
        # Visible Vegetation Index (VVI) as NDVI approximation
        denominator = g + r + 1e-10
        vvi = (g - r) / denominator
        
        return np.clip(vvi, -1, 1)
    
    def analyze_vegetation_health(self, ndvi: np.ndarray) -> Dict[str, any]:
        """
        Analyze vegetation health based on NDVI values
        
        Args:
            ndvi: NDVI values
            
        Returns:
            Dictionary with vegetation health analysis
        """
        # Calculate statistics
        mean_ndvi = np.mean(ndvi)
        std_ndvi = np.std(ndvi)
        min_ndvi = np.min(ndvi)
        max_ndvi = np.max(ndvi)
        
        # Classify health levels
        excellent = np.sum((ndvi >= 0.6) & (ndvi <= 1.0))
        good = np.sum((ndvi >= 0.3) & (ndvi < 0.6))
        fair = np.sum((ndvi >= 0.1) & (ndvi < 0.3))
        poor = np.sum((ndvi >= -0.1) & (ndvi < 0.1))
        critical = np.sum(ndvi < -0.1)
        
        total_pixels = ndvi.size
        
        health_distribution = {
            'excellent': float((excellent / total_pixels) * 100) if total_pixels > 0 else 0.0,
            'good': float((good / total_pixels) * 100) if total_pixels > 0 else 0.0,
            'fair': float((fair / total_pixels) * 100) if total_pixels > 0 else 0.0,
            'poor': float((poor / total_pixels) * 100) if total_pixels > 0 else 0.0,
            'critical': float((critical / total_pixels) * 100) if total_pixels > 0 else 0.0
        }
        
        # Determine overall health
        if mean_ndvi >= 0.6:
            overall_health = 'excellent'
        elif mean_ndvi >= 0.3:
            overall_health = 'good'
        elif mean_ndvi >= 0.1:
            overall_health = 'fair'
        elif mean_ndvi >= -0.1:
            overall_health = 'poor'
        else:
            overall_health = 'critical'
        
        return {
            'mean_ndvi': float(mean_ndvi),
            'std_ndvi': float(std_ndvi),
            'min_ndvi': float(min_ndvi),
            'max_ndvi': float(max_ndvi),
            'overall_health': overall_health,
            'health_distribution': health_distribution
        }
    
    def detect_deforestation(self, current_image: np.ndarray, 
                           reference_image: np.ndarray, 
                           threshold: float = 0.2) -> Dict[str, any]:
        """
        Detect deforestation by comparing current and reference images
        
        Args:
            current_image: Current satellite image
            reference_image: Reference satellite image (older)
            threshold: Threshold for change detection
            
        Returns:
            Dictionary with deforestation analysis
        """
        # Calculate NDVI for both images
        current_ndvi = self.calculate_ndvi_from_rgb(current_image)
        reference_ndvi = self.calculate_ndvi_from_rgb(reference_image)
        
        # Calculate NDVI difference
        ndvi_diff = current_ndvi - reference_ndvi
        
        # Detect significant decreases (potential deforestation)
        deforestation_mask = ndvi_diff < -threshold
        
        # Calculate statistics
        deforestation_pixels = np.sum(deforestation_mask)
        total_pixels = deforestation_mask.size
        deforestation_percentage = (deforestation_pixels / total_pixels) * 100
        
        # Determine severity
        if deforestation_percentage > 20:
            severity = 'high'
        elif deforestation_percentage > 10:
            severity = 'medium'
        elif deforestation_percentage > 5:
            severity = 'low'
        else:
            severity = 'low'  # Changed from 'minimal' to 'low' for enum compatibility
        
        return {
            'deforestation_detected': deforestation_percentage > 5,
            'deforestation_percentage': float(deforestation_percentage),
            'severity': severity,
            'affected_pixels': int(deforestation_pixels),
            'ndvi_change': float(np.mean(ndvi_diff))
        }
    
    def assess_flood_risk(self, image: np.ndarray, 
                         elevation_data: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Assess flood risk based on water bodies and terrain
        
        Args:
            image: Satellite image
            elevation_data: Optional elevation data
            
        Returns:
            Dictionary with flood risk assessment
        """
        # Convert to HSV for better water detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define water color range in HSV
        lower_water = np.array([100, 50, 50])
        upper_water = np.array([130, 255, 255])
        
        # Create water mask
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        # Also detect water using blue channel
        b = image[:, :, 2].astype(np.float32)
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        
        blue_water_mask = (b > r * 1.2) & (b > g * 1.1)
        water_mask = water_mask | (blue_water_mask.astype(np.uint8) * 255)
        
        # Calculate water percentage
        water_pixels = np.sum(water_mask > 0)
        total_pixels = water_mask.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Assess risk based on water percentage
        if water_percentage > 30:
            risk_level = 'very_high'
        elif water_percentage > 20:
            risk_level = 'high'
        elif water_percentage > 10:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_factors = []
        if water_percentage > 15:
            risk_factors.append('high_water_coverage')
        if elevation_data is not None and np.mean(elevation_data) < 10:
            risk_factors.append('low_elevation')
        
        return {
            'risk_level': risk_level,
            'water_percentage': float(water_percentage),
            'risk_factors': risk_factors,
            'probability': float(min(water_percentage * 2, 100))  # Rough probability estimate
        }
    
    def assess_drought_risk(self, ndvi: np.ndarray, 
                          precipitation_data: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Assess drought risk based on vegetation health
        
        Args:
            ndvi: NDVI values
            precipitation_data: Optional precipitation data
            
        Returns:
            Dictionary with drought risk assessment
        """
        # Calculate vegetation stress indicators
        low_vegetation = np.sum(ndvi < 0.2)
        total_pixels = ndvi.size
        stress_percentage = (low_vegetation / total_pixels) * 100
        
        # Assess risk based on vegetation stress
        if stress_percentage > 60:
            risk_level = 'very_high'
        elif stress_percentage > 40:
            risk_level = 'high'
        elif stress_percentage > 20:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        risk_factors = []
        if stress_percentage > 30:
            risk_factors.append('vegetation_stress')
        if np.mean(ndvi) < 0.3:
            risk_factors.append('low_vegetation_health')
        if precipitation_data is not None and np.mean(precipitation_data) < 10:
            risk_factors.append('low_precipitation')
        
        return {
            'risk_level': risk_level,
            'stress_percentage': float(stress_percentage),
            'risk_factors': risk_factors,
            'probability': float(min(stress_percentage * 1.5, 100))  # Rough probability estimate
        }
    
    def generate_visualization(self, image: np.ndarray, 
                             classification_result: Dict[str, float],
                             save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate visualization of classification results
        
        Args:
            image: Original satellite image
            classification_result: Classification results
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Satellite Image')
        axes[0].axis('off')
        
        # Classification pie chart
        labels = list(classification_result.keys())
        sizes = list(classification_result.values())
        colors = [np.array(self.class_colors[label.title()])/255 for label in labels if label.title() in self.class_colors]
        
        axes[1].pie(sizes, labels=labels, colors=colors[:len(labels)], 
                   autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Land Classification')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return vis_image

def main():
    """Example usage of the LandClassificationModel"""
    # Initialize model
    model = LandClassificationModel()
    
    # Example: Load a satellite image (replace with actual image loading)
    # image = cv2.imread('satellite_image.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # For demonstration, create a sample image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Classify the image
    classification = model.classify_image(image)
    print("Land Classification Results:")
    for land_type, percentage in classification.items():
        print(f"{land_type.title()}: {percentage:.2f}%")
    
    # Calculate NDVI (example with mock bands)
    red_band = image[:, :, 2]
    nir_band = image[:, :, 0]  # Mock NIR band
    ndvi = model.calculate_ndvi(red_band, nir_band)
    
    # Analyze vegetation health
    health_analysis = model.analyze_vegetation_health(ndvi)
    print(f"\nVegetation Health: {health_analysis['overall_health']}")
    print(f"Mean NDVI: {health_analysis['mean_ndvi']:.3f}")

if __name__ == "__main__":
    main()
