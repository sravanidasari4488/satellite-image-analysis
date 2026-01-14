"""
Google Earth Engine Integration Module
Provides efficient access to satellite imagery and processing capabilities
"""

import ee
import numpy as np
from PIL import Image
import io
import logging
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GoogleEarthEngineIntegration:
    """
    Google Earth Engine integration for efficient satellite image processing
    """
    
    def __init__(self):
        """Initialize Google Earth Engine"""
        try:
            # Initialize Earth Engine
            # Check if already initialized
            try:
                ee.Number(1).getInfo()  # Test if initialized
                logger.info("Google Earth Engine already initialized")
                self.initialized = True
                return
            except Exception:
                pass  # Not initialized yet, continue with initialization
            
            # Try to authenticate using service account or user credentials
            credentials_path = os.getenv('GEE_CREDENTIALS_PATH')
            project_id = os.getenv('GEE_PROJECT_ID')
            
            if credentials_path and os.path.exists(credentials_path):
                # Use service account credentials file
                try:
                    # For service account, use credentials file directly
                    ee.Initialize(credentials=credentials_path, project=project_id)
                    logger.info("Google Earth Engine initialized with service account credentials")
                    self.initialized = True
                    return
                except Exception as e:
                    logger.warning(f"Service account initialization failed: {e}")
            
            # Try default initialization (requires earthengine authenticate)
            try:
                # Initialize with project if provided, otherwise without
                if project_id:
                    ee.Initialize(project=project_id)
                else:
                    ee.Initialize()
                logger.info("Google Earth Engine initialized (default/user credentials)")
                self.initialized = True
            except Exception as e:
                # If initialization fails, it means user needs to authenticate
                logger.warning(f"Google Earth Engine initialization failed: {e}")
                logger.warning("Run 'python -m earthengine authenticate' or set GEE_PROJECT_ID to set up credentials.")
                self.initialized = False
                
        except Exception as e:
            logger.warning(f"Google Earth Engine initialization error: {e}")
            logger.warning("Some features may not be available. Run 'earthengine authenticate' to set up credentials.")
            self.initialized = False
    
    def get_sentinel2_collection(self, start_date: str = None, end_date: str = None, 
                                 cloud_cover: int = 20) -> ee.ImageCollection:
        """
        Get Sentinel-2 image collection with cloud filtering
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format (default: 30 days ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
            cloud_cover: Maximum cloud cover percentage (default: 20)
        
        Returns:
            Filtered Sentinel-2 image collection
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover))
                     .sort('system:time_start', False))  # Most recent first
        
        return collection
    
    def get_median_image(self, collection: ee.ImageCollection) -> ee.Image:
        """
        Get median composite from image collection
        
        Args:
            collection: Earth Engine image collection
        
        Returns:
            Median composite image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        return collection.median()
    
    def get_least_cloudy_image(self, collection: ee.ImageCollection) -> ee.Image:
        """
        Get the least cloudy image from collection
        
        Args:
            collection: Earth Engine image collection
        
        Returns:
            Least cloudy image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        return ee.Image(collection.sort('CLOUDY_PIXEL_PERCENTAGE').first())
    
    def clip_to_bounds(self, image: ee.Image, bounds: List[float]) -> ee.Image:
        """
        Clip image to bounding box
        
        Args:
            image: Earth Engine image
            bounds: [min_lng, min_lat, max_lng, max_lat]
        
        Returns:
            Clipped image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        geometry = ee.Geometry.Rectangle(bounds)
        return image.clip(geometry)
    
    def get_rgb_image(self, image: ee.Image, bounds: List[float], 
                      scale: int = 10, max_pixels: int = 1e9) -> np.ndarray:
        """
        Get RGB image as numpy array
        
        Args:
            image: Earth Engine image
            bounds: [min_lng, min_lat, max_lng, max_lat]
            scale: Pixel scale in meters (default: 10 for Sentinel-2)
            max_pixels: Maximum pixels to process
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        # Select RGB bands (B4=Red, B3=Green, B2=Blue) and apply visualization
        rgb = image.select(['B4', 'B3', 'B2'])
        
        # Apply visualization parameters for better contrast
        rgb = rgb.visualize(
            min=0,
            max=3000,  # Typical Sentinel-2 reflectance range
            gamma=1.2
        )
        
        # Clip to bounds
        geometry = ee.Geometry.Rectangle(bounds)
        rgb = rgb.clip(geometry)
        
        # Calculate dimensions based on bounds and scale
        bbox_width = bounds[2] - bounds[0]
        bbox_height = bounds[3] - bounds[1]
        
        # Approximate pixel dimensions (1 degree ≈ 111 km)
        width_meters = bbox_width * 111000
        height_meters = bbox_height * 111000
        
        width_pixels = int(width_meters / scale)
        height_pixels = int(height_meters / scale)
        
        # Limit dimensions to prevent memory issues
        max_dimension = 2048
        if width_pixels > max_dimension or height_pixels > max_dimension:
            scale_factor = max(width_pixels, height_pixels) / max_dimension
            width_pixels = int(width_pixels / scale_factor)
            height_pixels = int(height_pixels / scale_factor)
            scale = scale * scale_factor
        
        # Get image URL
        try:
            url = rgb.getThumbURL({
                'region': geometry,
                'dimensions': [width_pixels, height_pixels],
                'format': 'png',
                'maxPixels': max_pixels
            })
            
            # Download the image
            import requests
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            img = Image.open(io.BytesIO(response.content))
            img_array = np.array(img)
            
            # Ensure RGB format
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array, img_array, img_array], axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # Remove alpha channel
            
            return img_array
        except Exception as e:
            logger.error(f"Error downloading RGB image: {e}")
            raise RuntimeError(f"Failed to download RGB image: {e}")
    
    def calculate_ndvi(self, image: ee.Image) -> ee.Image:
        """
        Calculate NDVI from Sentinel-2 image
        
        Args:
            image: Earth Engine image with NIR and Red bands
        
        Returns:
            NDVI image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return ndvi
    
    def calculate_ndwi(self, image: ee.Image) -> ee.Image:
        """
        Calculate NDWI (Normalized Difference Water Index) from Sentinel-2 image
        
        Args:
            image: Earth Engine image with Green and NIR bands
        
        Returns:
            NDWI image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        return ndwi
    
    def calculate_ndbi(self, image: ee.Image) -> ee.Image:
        """
        Calculate NDBI (Normalized Difference Built-up Index) from Sentinel-2 image
        
        Args:
            image: Earth Engine image with SWIR and NIR bands
        
        Returns:
            NDBI image
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        return ndbi
    
    def get_spectral_indices(self, image: ee.Image) -> ee.Image:
        """
        Calculate multiple spectral indices at once
        
        Args:
            image: Earth Engine image
        
        Returns:
            Image with NDVI, NDWI, and NDBI bands
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        ndvi = self.calculate_ndvi(image)
        ndwi = self.calculate_ndwi(image)
        ndbi = self.calculate_ndbi(image)
        
        return image.addBands([ndvi, ndwi, ndbi])
    
    def get_image_statistics(self, image: ee.Image, geometry: ee.Geometry) -> Dict[str, Any]:
        """
        Get statistics for an image region
        
        Args:
            image: Earth Engine image
            geometry: Region of interest
        
        Returns:
            Dictionary with statistics
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        stats = image.reduceRegion(
            reducer=ee.Reducer.minMax().combine(
                ee.Reducer.mean(),
                '', True
            ).combine(
                ee.Reducer.stdDev(),
                '', True
            ),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        )
        
        return stats.getInfo()
    
    def classify_land_cover(self, image: ee.Image, method: str = 'indices') -> ee.Image:
        """
        Classify land cover using spectral indices
        
        Args:
            image: Earth Engine image with spectral indices
            method: Classification method ('indices' or 'ml')
        
        Returns:
            Classified image with land cover classes
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        if method == 'indices':
            # Simple threshold-based classification
            ndvi = image.select('NDVI')
            ndwi = image.select('NDWI')
            ndbi = image.select('NDBI')
            
            # Classify: 0=Water, 1=Forest, 2=Urban, 3=Agricultural, 4=Barren
            water = ndwi.gt(0.3)
            forest = ndvi.gt(0.4).And(ndwi.lte(0.3))
            urban = ndbi.gt(0.1).Or(ndvi.lte(0.2))
            agricultural = ndvi.gt(0.2).And(ndvi.lte(0.4)).And(ndbi.lte(0.1))
            barren = ndvi.lte(0.2).And(ndwi.lte(0.3)).And(ndbi.lte(0.1))
            
            classified = (water.multiply(0)
                         .add(forest.multiply(1))
                         .add(urban.multiply(2))
                         .add(agricultural.multiply(3))
                         .add(barren.multiply(4)))
            
            return classified.rename('landcover')
        else:
            raise ValueError(f"Unknown classification method: {method}")
    
    def get_land_cover_statistics(self, image: ee.Image, bounds: List[float]) -> Dict[str, Any]:
        """
        Get land cover class statistics for a region
        
        Args:
            image: Earth Engine image with land cover classification
            bounds: [min_lng, min_lat, max_lng, max_lat]
        
        Returns:
            Dictionary with class areas and percentages
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        geometry = ee.Geometry.Rectangle(bounds)
        classified = self.classify_land_cover(image)
        classified = classified.clip(geometry)
        
        # Calculate histogram
        histogram = classified.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=30,
            maxPixels=1e9
        )
        
        hist_data = histogram.getInfo().get('landcover', {})
        
        # Calculate total pixels
        total = sum(hist_data.values())
        
        # Map classes
        class_names = {0: 'water', 1: 'forest', 2: 'urban', 3: 'agricultural', 4: 'barren'}
        
        class_areas = {}
        pixel_size_km2 = 0.0009  # 30m x 30m = 0.0009 km²
        
        for class_id, count in hist_data.items():
            class_name = class_names.get(int(float(class_id)), 'unknown')
            percentage = (count / total * 100) if total > 0 else 0
            area_km2 = count * pixel_size_km2
            
            class_areas[class_name] = {
                'pixels': count,
                'area_km2': area_km2,
                'percentage': percentage
            }
        
        return {
            'total_pixels': total,
            'total_area_km2': total * pixel_size_km2,
            'class_areas': class_areas
        }
    
    def get_image_for_analysis(self, location: str, bounds: List[float],
                               start_date: str = None, end_date: str = None,
                               cloud_cover: int = 20) -> Dict[str, Any]:
        """
        Get processed satellite image ready for analysis
        
        Args:
            location: Location name (for metadata)
            bounds: [min_lng, min_lat, max_lng, max_lat]
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            cloud_cover: Maximum cloud cover percentage
        
        Returns:
            Dictionary with image data and metadata
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        # Get image collection
        collection = self.get_sentinel2_collection(start_date, end_date, cloud_cover)
        
        # Get least cloudy image
        image = self.get_least_cloudy_image(collection)
        
        # Clip to bounds
        image = self.clip_to_bounds(image, bounds)
        
        # Calculate spectral indices
        image_with_indices = self.get_spectral_indices(image)
        
        # Get RGB image
        rgb_array = self.get_rgb_image(image, bounds, scale=10)
        
        # Get statistics
        geometry = ee.Geometry.Rectangle(bounds)
        stats = self.get_image_statistics(image_with_indices, geometry)
        
        # Get land cover statistics
        land_cover_stats = self.get_land_cover_statistics(image_with_indices, bounds)
        
        return {
            'rgb_image': rgb_array,
            'image': image_with_indices,
            'statistics': stats,
            'land_cover': land_cover_stats,
            'metadata': {
                'source': 'google-earth-engine',
                'collection': 'COPERNICUS/S2_SR_HARMONIZED',
                'location': location,
                'bounds': bounds,
                'start_date': start_date,
                'end_date': end_date,
                'cloud_cover': cloud_cover,
                'resolution': '10m',
                'bands': ['B2', 'B3', 'B4', 'B8', 'B11', 'NDVI', 'NDWI', 'NDBI']
            }
        }
    
    def export_image_to_asset(self, image: ee.Image, asset_path: str, 
                             bounds: List[float], scale: int = 10) -> str:
        """
        Export image to Earth Engine asset
        
        Args:
            image: Earth Engine image to export
            asset_path: Full asset path (e.g., 'users/username/image_name')
            bounds: [min_lng, min_lat, max_lng, max_lat]
            scale: Pixel scale in meters
        
        Returns:
            Task ID for the export
        """
        if not self.initialized:
            raise RuntimeError("Google Earth Engine not initialized")
        
        geometry = ee.Geometry.Rectangle(bounds)
        
        task = ee.batch.Export.image.toAsset(
            image=image,
            description='Satellite Image Export',
            assetId=asset_path,
            region=geometry,
            scale=scale,
            maxPixels=1e9
        )
        
        task.start()
        return task.id

