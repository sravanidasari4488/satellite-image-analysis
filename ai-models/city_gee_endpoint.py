"""
Flask endpoint for city-level GEE analysis
Integrates with backend to fetch city boundaries and perform analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

from city_level_gee_analysis import CityLevelGEEAnalysis

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize GEE analysis (lazy initialization)
gee_analyzer = None

def get_gee_analyzer():
    """Get or initialize GEE analyzer"""
    global gee_analyzer
    if gee_analyzer is None:
        try:
            gee_analyzer = CityLevelGEEAnalysis()
            logger.info("City-level GEE analyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize GEE analyzer: {e}")
            logger.error("City-level GEE analysis will not be available")
            gee_analyzer = None
    return gee_analyzer


def create_city_analysis_endpoint(app: Flask):
    """
    Create city-level analysis endpoint
    
    Expected JSON payload:
    {
        "city_name": "New York",
        "polygon_coords": [[lng1, lat1], [lng2, lat2], ...],
        "start_date": "2024-01-01" (optional),
        "end_date": "2024-01-31" (optional),
        "cloud_cover_threshold": 20 (optional)
    }
    """
    
    @app.route('/gee/analyze-city', methods=['POST'])
    def analyze_city_gee():
        """
        City-level land cover analysis using Google Earth Engine
        
        This endpoint performs:
        1. Region-based classification over exact city boundary
        2. Accurate area calculations (km²) using GEE reducers
        3. Returns percentages that sum to ~100%
        """
        try:
            # Get or initialize GEE analyzer
            analyzer = get_gee_analyzer()
            if not analyzer:
                return jsonify({
                    'error': 'Google Earth Engine not initialized. Please ensure GEE is set up correctly.',
                    'type': 'GEEInitializationError',
                    'hint': 'Run: python -c "import ee; ee.Authenticate(); ee.Initialize(project=\'your-project-id\')"'
                }), 503
            
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'Request body is required'}), 400
            
            city_name = data.get('city_name')
            polygon_coords = data.get('polygon_coords')
            
            if not city_name:
                return jsonify({'error': 'city_name is required'}), 400
            
            if not polygon_coords:
                return jsonify({'error': 'polygon_coords is required'}), 400
            
            if not isinstance(polygon_coords, list) or len(polygon_coords) < 3:
                return jsonify({
                    'error': 'polygon_coords must be a list of at least 3 [lng, lat] coordinates'
                }), 400
            
            # Validate coordinate format
            for coord in polygon_coords:
                if not isinstance(coord, list) or len(coord) != 2:
                    return jsonify({
                        'error': 'Each coordinate must be [lng, lat]'
                    }), 400
            
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            cloud_cover = data.get('cloud_cover_threshold', 20)
            include_rgb = data.get('include_rgb_image', False)  # Default False to avoid memory issues
            
            # Perform analysis (all processing in GEE - no large image downloads)
            result = analyzer.analyze_city(
                city_name=city_name,
                polygon_coords=polygon_coords,
                start_date=start_date,
                end_date=end_date,
                cloud_cover_threshold=cloud_cover,
                include_rgb_image=include_rgb
            )
            
            return jsonify({
                'success': True,
                **result
            })
            
        except Exception as e:
            logger.error(f"City-level GEE analysis error: {e}", exc_info=True)
            return jsonify({
                'error': str(e),
                'type': type(e).__name__
            }), 500

