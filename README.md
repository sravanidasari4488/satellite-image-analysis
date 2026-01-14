# Satellite-based Geospatial Intelligence System

A real-time satellite-based geospatial intelligence system that performs land cover classification and climate risk assessment using Sentinel-2 satellite imagery, Google Earth Engine, and weather data.

## Features

1. **Geocoding**: Convert city names or coordinates to bounding boxes using OpenCage API
2. **Satellite Imagery**: Fetch real Sentinel-2 imagery from Google Earth Engine
3. **Land Cover Classification**: Classify pixels into 5 categories:
   - Urban
   - Forest
   - Vegetation
   - Water
   - Bare land
4. **Percentage Calculation**: Compute exact pixel-based percentages for each land cover type
5. **Weather Data**: Fetch real-time weather data from OpenWeather API
6. **Climate Risk Assessment**: Calculate flood, heat, and drought risks using scientific rules

## Prerequisites

1. **Google Earth Engine Account**: 
   - Sign up at https://earthengine.google.com/
   - Authenticate using: `earthengine authenticate`

2. **OpenCage API Key**: 
   - Sign up at https://opencagedata.com/
   - Get your API key

3. **OpenWeather API Key**:
   - Sign up at https://openweathermap.org/api
   - Get your API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Authenticate Google Earth Engine:
```bash
python authenticate_earth_engine.py
```

Or manually:
```bash
python -c "import ee; ee.Authenticate()"
```

3. Set up Google Cloud Project:
   - Visit https://code.earthengine.google.com/ to see your project ID
   - Or create a new project at https://console.cloud.google.com/

4. Set environment variables:
```bash
# Windows PowerShell
$env:OPENCAGE_API_KEY="your_opencage_key"
$env:OPENWEATHER_API_KEY="your_openweather_key"
$env:EARTHENGINE_PROJECT="your-project-id"  # Optional, but recommended

# Linux/Mac
export OPENCAGE_API_KEY="your_opencage_key"
export OPENWEATHER_API_KEY="your_openweather_key"
export EARTHENGINE_PROJECT="your-project-id"  # Optional, but recommended
```

Or create a `.env` file:
```
OPENCAGE_API_KEY=your_opencage_key
OPENWEATHER_API_KEY=your_openweather_key
EARTHENGINE_PROJECT=your-project-id
```

**Note:** If you get a "no project found" error, you must set the `EARTHENGINE_PROJECT` environment variable. See `EARTH_ENGINE_SETUP.md` for detailed instructions.

## Usage

### Command Line

```bash
python geospatial_intelligence.py "Paris, France"
```

Or with coordinates:
```bash
python geospatial_intelligence.py "48.8566,2.3522"
```

### Web Interface (React.js)

1. Install frontend dependencies:
```bash
cd frontend
npm install
```

2. Start the Flask backend server:
```bash
python api_server.py
```

3. In a new terminal, start the React frontend:
```bash
cd frontend
npm start
```

The web interface will open at `http://localhost:3000`

### API Server (Direct API Access)

Start the Flask server:
```bash
python api_server.py
```

Send POST request:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"location": "Paris, France"}'
```

With date range:
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"location": "Paris, France", "start_date": "2024-01-01", "end_date": "2024-01-31"}'
```

## Output Format

```json
{
  "location": "Paris, France",
  "bounding_box": {
    "min_lon": 2.2241,
    "min_lat": 48.8156,
    "max_lon": 2.4699,
    "max_lat": 48.9022
  },
  "land_cover": {
    "urban": 45.23,
    "forest": 12.45,
    "vegetation": 28.67,
    "water": 8.90,
    "bare_land": 4.75,
    "total_pixels": 1234567
  },
  "weather": {
    "temperature": 18.5,
    "humidity": 65.0,
    "precipitation": 2.3,
    "wind_speed": 12.5,
    "pressure": 1013.25
  },
  "climate_risks": {
    "flood": 25.5,
    "heat": 35.2,
    "drought": 15.8
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## Technical Details

- **Satellite Data**: Uses Sentinel-2 Level-2A (Surface Reflectance) from Google Earth Engine
- **Classification Method**: Rule-based classification using spectral indices (NDVI, NDWI, EVI, SAVI)
- **Pixel Resolution**: 30 meters (configurable)
- **Cloud Filtering**: Automatically filters images with <20% cloud coverage
- **Climate Risk Calculation**: Based on scientific rules combining weather patterns and land cover characteristics

## Error Handling

The system will throw errors if:
- API keys are missing or invalid
- Google Earth Engine is not authenticated
- Location cannot be geocoded
- Satellite imagery is unavailable for the specified area/date
- Weather data cannot be retrieved

## License

MIT License

