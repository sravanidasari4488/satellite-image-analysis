"""
Flask API service for satellite image analysis
Provides endpoints for land classification, NDVI analysis, and risk assessment
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import logging
from typing import Dict, Any
import os
from dotenv import load_dotenv

from land_classification import LandClassificationModel
from multispectral_analysis import MultispectralAnalyzer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize the AI models
model = LandClassificationModel()  # Legacy pixel-based model
multispectral_analyzer = MultispectralAnalyzer()  # New ML-driven multispectral analyzer

# Initialize city pipeline if available
city_pipeline = None
try:
    from city_analysis_pipeline import CityAnalysisPipeline
    city_pipeline = CityAnalysisPipeline()
    logger.info("City analysis pipeline initialized")
except ImportError as e:
    logger.warning(f"City analysis pipeline not available: {e}")
    city_pipeline = None
except Exception as e:
    logger.warning(f"Failed to initialize city pipeline: {e}")
    city_pipeline = None

def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(pil_image)
        
        return image
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise ValueError("Invalid image data")

def encode_image(image: np.ndarray) -> str:
    """Encode numpy array image to base64 string"""
    try:
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise ValueError("Error encoding image")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model.model is not None,
        'version': '1.0.0'
    })

@app.route('/classify', methods=['POST'])
def classify_land():
    """
    Classify land types in satellite image
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "include_visualization": true/false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode image
        image = decode_image(data['image'])
        
        # Classify land types
        classification = model.classify_image(image)
                # Calculate NDVI using RGB-based approximation
        ndvi_analysis = None
        try:
            ndvi = model.calculate_ndvi_from_rgb(image)
            ndvi_analysis = model.analyze_vegetation_health(ndvi)
        except Exception as e:
            logger.warning(f"NDVI calculation failed: {e}")
        
        # Generate visualization if requested
        visualization = None
        if data.get('include_visualization', False):
            try:
                vis_image = model.generate_visualization(image, classification)
                visualization = encode_image(vis_image)
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        
        response = {
            'success': True,
            'classification': classification,
            'ndvi_analysis': ndvi_analysis,
            'visualization': visualization
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-ndvi', methods=['POST'])
def analyze_ndvi():
    """
    Analyze NDVI from satellite image bands
    
    Expected JSON payload:
    {
        "red_band": "base64_encoded_red_band",
        "nir_band": "base64_encoded_nir_band"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'red_band' not in data or 'nir_band' not in data:
            return jsonify({'error': 'Red and NIR bands are required'}), 400
        
        # Decode bands
        red_band = decode_image(data['red_band'])
        nir_band = decode_image(data['nir_band'])
        
        # Convert to grayscale if necessary
        if len(red_band.shape) == 3:
            red_band = cv2.cvtColor(red_band, cv2.COLOR_RGB2GRAY)
        if len(nir_band.shape) == 3:
            nir_band = cv2.cvtColor(nir_band, cv2.COLOR_RGB2GRAY)
        
        # Calculate NDVI
        ndvi = model.calculate_ndvi(red_band, nir_band)
        
        # Analyze vegetation health
        health_analysis = model.analyze_vegetation_health(ndvi)
        
        # Generate NDVI visualization
        ndvi_normalized = ((ndvi + 1) * 127.5).astype(np.uint8)
        ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
        ndvi_visualization = encode_image(ndvi_colored)
        
        response = {
            'success': True,
            'ndvi_statistics': {
                'mean': float(np.mean(ndvi)),
                'std': float(np.std(ndvi)),
                'min': float(np.min(ndvi)),
                'max': float(np.max(ndvi))
            },
            'health_analysis': health_analysis,
            'ndvi_visualization': ndvi_visualization
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"NDVI analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/assess-risks', methods=['POST'])
def assess_risks():
    """
    Assess environmental risks (flood, drought, deforestation)
    
    Expected JSON payload:
    {
        "current_image": "base64_encoded_current_image",
        "reference_image": "base64_encoded_reference_image" (optional),
        "elevation_data": "base64_encoded_elevation_data" (optional),
        "precipitation_data": "base64_encoded_precipitation_data" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'current_image' not in data:
            return jsonify({'error': 'Current image is required'}), 400
        
        # Decode current image
        current_image = decode_image(data['current_image'])
        
        # Assess flood risk
        elevation_data = None
        if 'elevation_data' in data:
            elevation_data = decode_image(data['elevation_data'])
            if len(elevation_data.shape) == 3:
                elevation_data = cv2.cvtColor(elevation_data, cv2.COLOR_RGB2GRAY)
        
        flood_risk = model.assess_flood_risk(current_image, elevation_data)
        
        # Assess drought risk
        precipitation_data = None
        if 'precipitation_data' in data:
            precipitation_data = decode_image(data['precipitation_data'])
            if len(precipitation_data.shape) == 3:
                precipitation_data = cv2.cvtColor(precipitation_data, cv2.COLOR_RGB2GRAY)
        
        # Calculate NDVI for drought assessment
        ndvi = model.calculate_ndvi_from_rgb(current_image)
        drought_risk = model.assess_drought_risk(ndvi, precipitation_data)
        
        # Detect deforestation if reference image provided
        deforestation = None
        if 'reference_image' in data:
            reference_image = decode_image(data['reference_image'])
            deforestation = model.detect_deforestation(current_image, reference_image)
        
        response = {
            'success': True,
            'flood_risk': flood_risk,
            'drought_risk': drought_risk,
            'deforestation': deforestation
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/segment', methods=['POST'])
def segment_image():
    """
    Segment satellite image using clustering
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "num_clusters": 5 (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode image
        image = decode_image(data['image'])
        
        # Get number of clusters
        num_clusters = data.get('num_clusters', 5)
        
        # Segment image
        segmented = model.segment_image(image, num_clusters)
        
        # Create colored segmentation
        segmented_colored = cv2.applyColorMap(
            (segmented * 255 // num_clusters).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Encode result
        segmentation_visualization = encode_image(segmented_colored)
        
        response = {
            'success': True,
            'num_clusters': num_clusters,
            'segmentation_visualization': segmentation_visualization
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Perform comprehensive analysis on satellite image
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "include_visualization": true/false
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Request body is required', 'type': 'ValidationError'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'Image data is required', 'type': 'ValidationError'}), 400
        
        # Decode image
        try:
            image = decode_image(data['image'])
        except Exception as decode_error:
            logger.error(f"Image decoding failed: {decode_error}", exc_info=True)
            return jsonify({
                'error': f'Failed to decode image: {str(decode_error)}',
                'type': type(decode_error).__name__
            }), 400
        
        # Perform all analyses
        try:
            classification = model.classify_image(image)
        except Exception as e:
            logger.error(f"Land classification error in batch-analyze: {e}", exc_info=True)
            # Set default classification if it fails
            classification = {
                'forest': 0.0,
                'water': 0.0,
                'urban': 0.0,
                'agricultural': 0.0,
                'barren': 0.0
            }
        
        # NDVI analysis - use RGB-based approximation
        ndvi = None
        try:
            ndvi = model.calculate_ndvi_from_rgb(image)
            ndvi_analysis = model.analyze_vegetation_health(ndvi)
        except Exception as e:
            logger.error(f"NDVI calculation error in batch-analyze: {e}", exc_info=True)
            # Set default values if NDVI calculation fails
            ndvi_analysis = {
                'mean_ndvi': 0.0,
                'min_ndvi': -1.0,
                'max_ndvi': 1.0,
                'health_distribution': {
                    'excellent': 0,
                    'good': 0,
                    'fair': 0,
                    'poor': 0,
                    'critical': 0
                }
            }
            # Create a default NDVI array for risk assessment
            ndvi = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Risk assessments
        try:
            flood_risk = model.assess_flood_risk(image)
        except Exception as e:
            logger.error(f"Flood risk assessment error: {e}", exc_info=True)
            flood_risk = {'level': 'unknown', 'probability': 0.0, 'factors': []}
        
        try:
            drought_risk = model.assess_drought_risk(ndvi)
        except Exception as e:
            logger.error(f"Drought risk assessment error: {e}", exc_info=True)
            drought_risk = {'level': 'unknown', 'probability': 0.0, 'factors': []}
        
        # Generate visualizations
        visualizations = {}
        if data.get('include_visualization', False):
            try:
                # Classification visualization
                vis_image = model.generate_visualization(image, classification)
                visualizations['classification'] = encode_image(vis_image)
                
                # NDVI visualization
                ndvi_normalized = ((ndvi + 1) * 127.5).astype(np.uint8)
                ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
                visualizations['ndvi'] = encode_image(ndvi_colored)
                
                # Segmentation
                segmented = model.segment_image(image)
                segmented_colored = cv2.applyColorMap(
                    (segmented * 255 // 5).astype(np.uint8), 
                    cv2.COLORMAP_JET
                )
                visualizations['segmentation'] = encode_image(segmented_colored)
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
        
        response = {
            'success': True,
            'classification': classification,
            'ndvi_analysis': ndvi_analysis,
            'flood_risk': flood_risk,
            'drought_risk': drought_risk,
            'visualizations': visualizations
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Traceback: {error_details}")
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'details': error_details if os.environ.get('DEBUG', 'False').lower() == 'true' else None
        }), 500

@app.route('/multispectral-analyze', methods=['POST'])
def multispectral_analyze():
    """
    Perform ML-driven multispectral analysis using Sentinel-2 bands
    
    Expected JSON payload:
    {
        "image": "base64_encoded_image_data",
        "nir_band": "base64_encoded_nir_band" (optional),
        "swir_band": "base64_encoded_swir_band" (optional),
        "include_indices": true/false,
        "include_segmentation": true/false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        
        # Decode RGB image
        image = decode_image(data['image'])
        
        # Decode optional multispectral bands
        nir_band = None
        swir_band = None
        
        if 'nir_band' in data:
            nir_band = decode_image(data['nir_band'])
            if len(nir_band.shape) == 3:
                nir_band = cv2.cvtColor(nir_band, cv2.COLOR_RGB2GRAY)
        
        if 'swir_band' in data:
            swir_band = decode_image(data['swir_band'])
            if len(swir_band.shape) == 3:
                swir_band = cv2.cvtColor(swir_band, cv2.COLOR_RGB2GRAY)
        
        # Prepare multispectral image
        if nir_band is not None:
            # Combine RGB + NIR
            multispectral = np.dstack([image, nir_band])
        else:
            multispectral = image
        
        # Calculate spectral indices
        indices = {}
        if data.get('include_indices', True):
            indices = multispectral_analyzer.calculate_indices_from_rgb_nir(image, nir_band)
        
        # Perform land cover segmentation
        segmentation_result = {}
        if data.get('include_segmentation', True):
            segmentation_result = multispectral_analyzer.segment_landcover(multispectral)
        
        # Assess risks using indices
        flood_risk = {}
        drought_risk = {}
        
        if 'ndwi' in indices and 'ndvi' in indices:
            flood_risk = multispectral_analyzer.assess_flood_risk(
                indices['ndwi'], 
                indices['ndvi']
            )
            drought_risk = multispectral_analyzer.assess_drought_risk(
                indices['ndvi'],
                indices['ndwi']
            )
        
        response = {
            'success': True,
            'indices': {
                'ndvi': {
                    'mean': float(np.mean(indices.get('ndvi', 0))),
                    'min': float(np.min(indices.get('ndvi', 0))),
                    'max': float(np.max(indices.get('ndvi', 0)))
                } if 'ndvi' in indices else None,
                'ndwi': {
                    'mean': float(np.mean(indices.get('ndwi', 0))),
                    'min': float(np.min(indices.get('ndwi', 0))),
                    'max': float(np.max(indices.get('ndwi', 0)))
                } if 'ndwi' in indices else None,
                'ndbi': {
                    'mean': float(np.mean(indices.get('ndbi', 0))),
                    'min': float(np.min(indices.get('ndbi', 0))),
                    'max': float(np.max(indices.get('ndbi', 0)))
                } if 'ndbi' in indices else None
            },
            'landcover_segmentation': segmentation_result,
            'flood_risk': flood_risk,
            'drought_risk': drought_risk,
            'method': 'ml_driven_multispectral'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Multispectral analysis error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/calculate-indices', methods=['POST'])
def calculate_indices():
    """
    Calculate spectral indices (NDVI, NDWI, NDBI) from multispectral bands
    
    Expected JSON payload:
    {
        "red_band": "base64_encoded_red_band",
        "green_band": "base64_encoded_green_band" (optional),
        "blue_band": "base64_encoded_blue_band" (optional),
        "nir_band": "base64_encoded_nir_band",
        "swir_band": "base64_encoded_swir_band" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if 'red_band' not in data or 'nir_band' not in data:
            return jsonify({'error': 'Red and NIR bands are required'}), 400
        
        # Decode bands
        red = decode_image(data['red_band'])
        nir = decode_image(data['nir_band'])
        
        if len(red.shape) == 3:
            red = cv2.cvtColor(red, cv2.COLOR_RGB2GRAY)
        if len(nir.shape) == 3:
            nir = cv2.cvtColor(nir, cv2.COLOR_RGB2GRAY)
        
        # Calculate NDVI
        ndvi = multispectral_analyzer.calculate_ndvi(red, nir)
        
        indices = {
            'ndvi': {
                'mean': float(np.mean(ndvi)),
                'std': float(np.std(ndvi)),
                'min': float(np.min(ndvi)),
                'max': float(np.max(ndvi))
            }
        }
        
        # Calculate NDWI if green band provided
        if 'green_band' in data:
            green = decode_image(data['green_band'])
            if len(green.shape) == 3:
                green = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
            ndwi = multispectral_analyzer.calculate_ndwi(green, nir)
            indices['ndwi'] = {
                'mean': float(np.mean(ndwi)),
                'std': float(np.std(ndwi)),
                'min': float(np.min(ndwi)),
                'max': float(np.max(ndwi))
            }
        
        # Calculate NDBI if SWIR band provided
        if 'swir_band' in data:
            swir = decode_image(data['swir_band'])
            if len(swir.shape) == 3:
                swir = cv2.cvtColor(swir, cv2.COLOR_RGB2GRAY)
            ndbi = multispectral_analyzer.calculate_ndbi(nir, swir)
            indices['ndbi'] = {
                'mean': float(np.mean(ndbi)),
                'std': float(np.std(ndbi)),
                'min': float(np.min(ndbi)),
                'max': float(np.max(ndbi))
            }
        
        return jsonify({
            'success': True,
            'indices': indices
        })
        
    except Exception as e:
        logger.error(f"Index calculation error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    # Don't override specific error responses - let them pass through
    # This handler only catches unhandled exceptions
    logger.error(f"Unhandled error: {error}", exc_info=True)
    return jsonify({
        'error': str(error) if error else 'Internal server error',
        'type': type(error).__name__ if error else 'UnknownError'
    }), 500

@app.route('/analyze-city', methods=['POST'])
def analyze_city():
    """
    City-agnostic land-use analysis using exact administrative boundaries
    
    Uses:
    - Exact OSM polygon boundaries (not bbox)
    - 64x64 patch-level CNN classification
    - Spectral index validation
    - Multi-factor risk models
    
    Expected JSON payload:
    {
        "location": "City name",
        "image": "base64_encoded_RGB_image",
        "red_band": "base64_encoded_red_band" (optional),
        "green_band": "base64_encoded_green_band" (optional),
        "blue_band": "base64_encoded_blue_band" (optional),
        "nir_band": "base64_encoded_nir_band" (optional),
        "swir_band": "base64_encoded_swir_band" (optional),
        "image_bbox": [min_lng, min_lat, max_lng, max_lat],
        "weather_data": {
            "precipitation_7d": float,
            "precipitation_30d": float,
            "rainfall_anomaly_3m": float,
            "rainfall_anomaly_6m": float,
            "elevation_variance": float,
            "ndvi_trend": float,
            "water_area_change": float,
            "satellite_date": "ISO date string",
            "cloud_coverage": float
        }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'location' not in data:
            return jsonify({'error': 'Location is required'}), 400
        
        if 'image' not in data:
            return jsonify({'error': 'Image data is required'}), 400
        
        location = data['location']
        
        # Decode RGB image
        image = decode_image(data['image'])
        
        # Decode multispectral bands
        multispectral_bands = {}
        for band_name in ['red', 'green', 'blue', 'nir', 'swir']:
            if f'{band_name}_band' in data:
                band_data = decode_image(data[f'{band_name}_band'])
                if len(band_data.shape) == 3:
                    # Convert to grayscale if RGB
                    band_data = cv2.cvtColor(band_data, cv2.COLOR_RGB2GRAY)
                multispectral_bands[band_name] = band_data
        
        # Extract RGB from image if bands not provided
        if 'red' not in multispectral_bands:
            multispectral_bands['red'] = image[:, :, 0]
            multispectral_bands['green'] = image[:, :, 1]
            multispectral_bands['blue'] = image[:, :, 2]
        
        # Get image bbox
        image_bbox = data.get('image_bbox')
        if not image_bbox:
            return jsonify({'error': 'image_bbox is required'}), 400
        
        # Get weather data
        weather_data = data.get('weather_data', {})
        
        # Check if pipeline is available
        if not city_pipeline:
            return jsonify({
                'error': 'City analysis pipeline not available. Check dependencies.'
            }), 503
        
        # Run pipeline
        results = city_pipeline.analyze_city(
            location=location,
            satellite_image=image,
            multispectral_bands=multispectral_bands,
            weather_data=weather_data,
            image_bbox=image_bbox
        )
        
        return jsonify({
            'success': True,
            **results
        })
        
    except Exception as e:
        logger.error(f"City analysis error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting AI service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

