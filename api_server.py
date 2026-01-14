"""
Flask API server for Geospatial Intelligence System
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from geospatial_intelligence import GeospatialIntelligenceSystem
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Cache for locality lists (7 days TTL)
# Structure: { city_name: { 'localities': [...], 'cached_at': datetime } }
app.locality_list_cache = {}

# Initialize system
opencage_key = os.getenv('OPENCAGE_API_KEY')
openweather_key = os.getenv('OPENWEATHER_API_KEY')
earthengine_project = os.getenv('EARTHENGINE_PROJECT')  # Optional
supabase_url = os.getenv('SUPABASE_URL')  # Optional
supabase_key = os.getenv('SUPABASE_KEY')  # Optional (service role key)

if not opencage_key or not openweather_key:
    raise ValueError("API keys must be set in environment variables")

system = GeospatialIntelligenceSystem(
    opencage_key, 
    openweather_key, 
    earthengine_project,
    supabase_url,
    supabase_key
)


@app.route('/localities', methods=['POST'])
def get_localities():
    """Get list of localities for a city (fast, names only)"""
    try:
        data = request.get_json()
        
        if not data or 'city' not in data:
            return jsonify({'error': 'City parameter is required'}), 400
        
        city_name = data['city'].strip()
        radius_km = data.get('radius_km', 20)  # Default 20km
        
        # Check cache first (7 days TTL)
        if city_name in app.locality_list_cache:
            cached_data = app.locality_list_cache[city_name]
            cached_at = cached_data.get('cached_at')
            if cached_at:
                age = datetime.now() - cached_at
                if age.days < 7:
                    # Return cached data
                    return jsonify({
                        'city': city_name,
                        'localities': cached_data['localities'],
                        'cached': True
                    }), 200
        
        # Fetch localities (names and centers only, no polygons)
        localities = system.geocoding.get_localities(city_name, radius_km)
        
        # Build response with names only
        locality_list = []
        for loc in localities:
            locality_list.append({
                'name': loc['name'],
                'lat': loc.get('lat'),
                'lon': loc.get('lon'),
                'place_type': loc.get('place_type', 'unknown')
            })
        
        # Cache the locality list for 7 days
        app.locality_list_cache[city_name] = {
            'localities': locality_list,
            'raw_data': localities,  # Store raw data for geometry fetching
            'cached_at': datetime.now()
        }
        
        return jsonify({
            'city': city_name,
            'localities': locality_list,
            'cached': False
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze a locality (fetches geometry on-demand)"""
    try:
        data = request.get_json()
        
        if not data or 'city' not in data or 'locality' not in data:
            return jsonify({'error': 'City and locality parameters are required'}), 400
        
        city_name = data['city'].strip()
        locality_name = data['locality'].strip()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Step 1: Find locality in cached list
        locality_info = None
        if city_name in app.locality_list_cache:
            cached_data = app.locality_list_cache[city_name]
            raw_data = cached_data.get('raw_data', [])
            
            # Find matching locality (case-insensitive)
            for loc in raw_data:
                if loc['name'].strip().lower() == locality_name.lower():
                    locality_info = loc
                    break
        
        if not locality_info:
            # Locality not found in cache - user needs to fetch localities first
            available = []
            if city_name in app.locality_list_cache:
                cached_data = app.locality_list_cache[city_name]
                available = [loc['name'] for loc in cached_data.get('localities', [])]
            
            error_msg = f'Locality "{locality_name}" not found. Please fetch localities first by clicking "Find Localities".'
            if available:
                error_msg += f' Available localities: {", ".join(available[:10])}...'
            return jsonify({'error': error_msg}), 404
        
        # Step 2: Fetch geometry on-demand for this ONE locality
        locality_polygon, locality_bbox = system.geocoding.get_locality_geometry(
            locality_info['name'],
            locality_info.get('lat', 0),
            locality_info.get('lon', 0)
        )
        
        # Step 3: Run analysis
        result = system.analyze_locality(
            city_name, locality_name, locality_polygon, locality_bbox,
            start_date, end_date
        )
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

