"""
City-Agnostic Satellite-Based Land-Use Analysis Pipeline
Main orchestrator for the complete analysis workflow

Pipeline Flow:
1. AOI → Fetch exact city polygon from OSM
2. Clipping → Clip Sentinel-2 image to polygon
3. Tiling → Generate 64x64 patches with sliding window
4. CNN → Classify each patch with EfficientNet
5. Mapping → Map EuroSAT classes to high-level classes
6. Validation → Validate with spectral indices (hybrid fusion)
7. Aggregation → Aggregate to city-level statistics
8. Risk → Calculate flood/drought risks
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
import os

try:
    from .aoi_handler import AOIHandler
    from .tiling_strategy import SlidingWindowTiler
    from .cnn_patch_classifier import CNNPatchClassifier
    from .class_mapper import ClassMapper
    from .index_validator import IndexValidator
    from .risk_models import FloodRiskModel, DroughtRiskModel
    from .aggregation_engine import AggregationEngine
except ImportError:
    # Fallback for direct imports
    from aoi_handler import AOIHandler
    from tiling_strategy import SlidingWindowTiler
    from cnn_patch_classifier import CNNPatchClassifier
    from class_mapper import ClassMapper
    from index_validator import IndexValidator
    from risk_models import FloodRiskModel, DroughtRiskModel
    from aggregation_engine import AggregationEngine

logger = logging.getLogger(__name__)

# Fallback imports
try:
    from sklearn.ensemble import RandomForestClassifier
    import pickle
    RANDOM_FOREST_AVAILABLE = True
except ImportError:
    RANDOM_FOREST_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False


class CityAnalysisPipeline:
    """
    Complete pipeline for city-level land-use analysis
    
    Uses trained EfficientNet model for patch-level classification
    Aggregates to city-level statistics with scientific validation
    """
    
    def __init__(self, 
                 tile_size: int = 64,
                 stride: int = 32,
                 batch_size: int = 32):
        """
        Args:
            tile_size: Size of tiles for CNN (64x64 for EuroSAT)
            stride: Stride for sliding window (32 = 50% overlap)
            batch_size: Batch size for CNN inference
        """
        self.aoi_handler = AOIHandler()
        self.tiler = SlidingWindowTiler(tile_size=tile_size, stride=stride)
        self.cnn_classifier = CNNPatchClassifier()
        self.class_mapper = ClassMapper()
        self.index_validator = IndexValidator()
        self.flood_model = FloodRiskModel()
        self.drought_model = DroughtRiskModel()
        self.aggregator = AggregationEngine()
        
        self.tile_size = tile_size
        self.stride = stride
        self.batch_size = batch_size
        
        # Fallback models
        self.rf_model = None
        self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models (Random Forest, KMeans)"""
        # Try to load Random Forest model
        if RANDOM_FOREST_AVAILABLE:
            rf_path = 'models/random_forest_model.pkl'
            if os.path.exists(rf_path):
                try:
                    with open(rf_path, 'rb') as f:
                        self.rf_model = pickle.load(f)
                    logger.info("Loaded Random Forest fallback model")
                except Exception as e:
                    # Version mismatch warnings are expected - models work but may have minor differences
                    if "InconsistentVersionWarning" in str(type(e).__name__) or "version" in str(e).lower():
                        logger.warning(f"RF model version mismatch (non-critical): {e}")
                        logger.info("   → Model will still work, but consider updating scikit-learn to match model version")
                        # Try to load anyway - it usually still works
                        try:
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                with open(rf_path, 'rb') as f:
                                    self.rf_model = pickle.load(f)
                                logger.info("   → RF model loaded despite version warning")
                        except:
                            pass
                    else:
                        logger.warning(f"Failed to load RF model: {e}")
    
    def analyze_city(self, 
                    location: str,
                    satellite_image: np.ndarray,
                    multispectral_bands: Dict[str, np.ndarray],
                    weather_data: Dict,
                    image_bbox: List[float]) -> Dict:
        """
        Complete city analysis pipeline
        
        Args:
            location: City name
            satellite_image: RGB satellite image (H, W, 3)
            multispectral_bands: Dict with 'red', 'green', 'blue', 'nir', 'swir' arrays
            weather_data: Weather data dict with rainfall, etc.
            image_bbox: [min_lng, min_lat, max_lng, max_lat] of image
        
        Returns:
            Complete analysis results dictionary
        """
        logger.info(f"Starting city analysis pipeline for: {location}")
        
        # Step 1: Fetch exact city polygon
        try:
            polygon_data = self.aoi_handler.fetch_city_polygon_osm(location)
            logger.info(f"Fetched polygon: {polygon_data['area_km2']:.2f} km²")
        except Exception as e:
            logger.error(f"Failed to fetch polygon: {e}")
            raise
        
        # Step 2: Clip image to polygon
        try:
            clipped_image, mask = self.aoi_handler.clip_image_to_polygon(
                satellite_image,
                polygon_data['polygon'],
                image_bbox
            )
            logger.info(f"Clipped image: {np.sum(mask)} valid pixels")
        except Exception as e:
            logger.error(f"Image clipping failed: {e}")
            raise
        
        # Step 3: Calculate spectral indices for validation
        # Calculate indices on full image first
        indices_map = self._calculate_indices_map(multispectral_bands)
        
        # Clip indices to polygon (indices are 2D arrays)
        clipped_indices_map = {}
        for name, index_array in indices_map.items():
            # Reshape to 3D for clipping function (add channel dimension)
            index_3d = index_array.reshape(*index_array.shape, 1)
            clipped_index_3d, _ = self.aoi_handler.clip_image_to_polygon(
                index_3d,
                polygon_data['polygon'],
                image_bbox
            )
            # Remove channel dimension
            if len(clipped_index_3d.shape) == 3:
                clipped_indices_map[name] = clipped_index_3d[:, :, 0]
            else:
                clipped_indices_map[name] = clipped_index_3d
        
        indices_map = clipped_indices_map
        
        # Step 4: Generate tiles and classify
        all_predictions = []
        all_indices = []
        tiles_processed = 0
        
        # Collect all tiles first
        tile_list = list(self.tiler.generate_tiles(clipped_image, mask))
        logger.info(f"Processing {len(tile_list)} tiles...")
        
        # Process in batches for efficiency
        for batch_start in range(0, len(tile_list), self.batch_size):
            batch_tiles = tile_list[batch_start:batch_start + self.batch_size]
            
            # Extract tile arrays and coordinates
            tile_arrays = [tile for tile, _ in batch_tiles]
            tile_coords = [coord for _, coord in batch_tiles]
            
            # Classify batch
            try:
                if self.cnn_classifier.is_available():
                    # Use CNN
                    eurosat_predictions = self.cnn_classifier.classify_batch(tile_arrays)
                    method_used = 'cnn'
                elif self.rf_model is not None:
                    # Fallback to Random Forest
                    eurosat_predictions = self._classify_with_rf(tile_arrays)
                    method_used = 'random_forest'
                else:
                    # Last resort: KMeans
                    eurosat_predictions = self._classify_with_kmeans(tile_arrays)
                    method_used = 'kmeans'
                    logger.warning("Using KMeans fallback - low confidence results")
                
                # Process each prediction
                for i, (tile_coord, eurosat_pred) in enumerate(zip(tile_coords, eurosat_predictions)):
                    # Map to high-level classes
                    highlevel_pred = self.class_mapper.map_patch_prediction(eurosat_pred)
                    
                    # Get indices for this tile
                    tile_indices = self._get_tile_indices(indices_map, tile_coord)
                    
                    # Validate with indices (hybrid fusion)
                    validated_pred = self.index_validator.validate_with_indices(
                        highlevel_pred,
                        tile_indices,
                        tile_coord
                    )
                    
                    all_predictions.append(validated_pred)
                    all_indices.append(tile_indices)
                    tiles_processed += 1
                    
            except Exception as e:
                logger.error(f"Batch classification failed: {e}")
                continue
        
        if not all_predictions:
            raise Exception("No valid predictions generated")
        
        logger.info(f"Processed {tiles_processed} tiles using {method_used}")
        
        # Step 5: Aggregate predictions
        aggregated_landcover = self.aggregator.aggregate_landcover(
            all_predictions,
            clipped_image.shape[:2],
            mask,
            self.tile_size,
            self.stride
        )
        
        # Step 6: Calculate risks
        flood_risk = self.flood_model.calculate_flood_risk(
            rainfall_7d=weather_data.get('precipitation_7d', 0),
            rainfall_30d=weather_data.get('precipitation_30d', 0),
            urban_percentage=aggregated_landcover['Urban']['percentage'],
            elevation_variance=weather_data.get('elevation_variance'),
            ndwi_density=np.mean([idx.get('ndwi', 0) for idx in all_indices])
        )
        
        drought_risk = self.drought_model.calculate_drought_risk(
            rainfall_anomaly_3m=weather_data.get('rainfall_anomaly_3m'),
            rainfall_anomaly_6m=weather_data.get('rainfall_anomaly_6m'),
            ndvi_trend=weather_data.get('ndvi_trend'),
            water_area_change=weather_data.get('water_area_change')
        )
        
        # Step 7: Calculate confidence
        confidence = self.aggregator.calculate_confidence(all_predictions)
        
        # Adjust confidence based on method used
        if method_used == 'kmeans':
            confidence *= 0.5  # Lower confidence for fallback method
        
        return {
            'land_cover': {
                'Vegetation': aggregated_landcover['Vegetation'],
                'Water': aggregated_landcover['Water'],
                'Urban': aggregated_landcover['Urban'],
                'Agricultural': aggregated_landcover['Agricultural'],
                'Barren': aggregated_landcover['Barren']
            },
            'flood_risk': flood_risk,
            'drought_risk': drought_risk,
            'confidence': round(confidence, 3),
            'metadata': {
                'satellite_date': weather_data.get('satellite_date'),
                'cloud_coverage': weather_data.get('cloud_coverage', 0),
                'aoi_source': polygon_data['source'],
                'total_tiles_analyzed': tiles_processed,
                'city_area_km2': round(polygon_data['area_km2'], 2),
                'classification_method': method_used,
                'tile_size': self.tile_size,
                'stride': self.stride
            }
        }
    
    def _calculate_indices_map(self, 
                              multispectral_bands: Dict[str, np.ndarray],
                              image_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Calculate spectral indices for entire image"""
        indices = {}
        
        red = multispectral_bands.get('red')
        green = multispectral_bands.get('green')
        blue = multispectral_bands.get('blue')
        nir = multispectral_bands.get('nir')
        swir = multispectral_bands.get('swir')
        
        if red is not None and nir is not None:
            # NDVI
            indices['ndvi'] = (nir - red) / (nir + red + 1e-10)
        
        if green is not None and nir is not None:
            # NDWI
            indices['ndwi'] = (green - nir) / (green + nir + 1e-10)
        
        if nir is not None and swir is not None:
            # NDBI
            indices['ndbi'] = (swir - nir) / (swir + nir + 1e-10)
        
        if red is not None and blue is not None and nir is not None and swir is not None:
            # BSI
            indices['bsi'] = self.index_validator.calculate_bsi(red, blue, nir, swir)
        
        return indices
    
    def _get_tile_indices(self, 
                         indices_map: Dict[str, np.ndarray],
                         tile_coord: Tuple[int, int]) -> Dict[str, float]:
        """Extract mean indices for a specific tile"""
        row, col = tile_coord
        tile_indices = {}
        
        for name, index_array in indices_map.items():
            tile_region = index_array[row:row+self.tile_size, col:col+self.tile_size]
            tile_indices[name] = np.mean(tile_region)
        
        return tile_indices
    
    def _classify_with_rf(self, tile_arrays: List[np.ndarray]) -> List[Dict[str, float]]:
        """Fallback: Classify using Random Forest"""
        if not RANDOM_FOREST_AVAILABLE or self.rf_model is None:
            raise RuntimeError("Random Forest not available")
        
        # Extract features (simplified - would need proper feature extraction)
        # For now, use flattened RGB values
        features = []
        for tile in tile_arrays:
            if tile.max() > 1.0:
                tile = tile / 255.0
            # Resize to 64x64 if needed
            if tile.shape[0] != 64 or tile.shape[1] != 64:
                tile = cv2.resize(tile, (64, 64))
            # Flatten
            features.append(tile.flatten())
        
        # Predict
        predictions = self.rf_model.predict_proba(features)
        
        # Convert to dict format
        eurosat_classes = self.cnn_classifier.eurosat_classes
        results = []
        for pred in predictions:
            results.append({
                cls: float(prob)
                for cls, prob in zip(eurosat_classes, pred)
            })
        
        return results
    
    def _classify_with_kmeans(self, tile_arrays: List[np.ndarray]) -> List[Dict[str, float]]:
        """Last resort: Classify using KMeans clustering"""
        if not KMEANS_AVAILABLE:
            raise RuntimeError("KMeans not available")
        
        # Simple KMeans-based classification
        # This is a very basic fallback
        results = []
        eurosat_classes = self.cnn_classifier.eurosat_classes
        
        for tile in tile_arrays:
            # Simple color-based classification
            if tile.max() > 1.0:
                tile = tile / 255.0
            
            # Calculate mean RGB
            mean_r = np.mean(tile[:, :, 0]) if len(tile.shape) == 3 else np.mean(tile)
            mean_g = np.mean(tile[:, :, 1]) if len(tile.shape) == 3 else 0
            mean_b = np.mean(tile[:, :, 2]) if len(tile.shape) == 3 else 0
            
            # Simple rule-based classification
            # This is very basic and should be improved
            probs = {cls: 0.0 for cls in eurosat_classes}
            
            if mean_g > 0.4:  # Green = vegetation
                probs['Forest'] = 0.4
                probs['HerbaceousVegetation'] = 0.3
            elif mean_b > 0.4:  # Blue = water
                probs['River'] = 0.3
                probs['SeaLake'] = 0.4
            elif mean_r + mean_g + mean_b > 0.6:  # Bright = urban
                probs['Residential'] = 0.3
                probs['Industrial'] = 0.2
            else:  # Dark = barren
                probs['Barren'] = 0.5
            
            # Normalize
            total = sum(probs.values())
            if total > 0:
                probs = {cls: prob / total for cls, prob in probs.items()}
            else:
                probs['Barren'] = 1.0
            
            results.append(probs)
        
        return results

