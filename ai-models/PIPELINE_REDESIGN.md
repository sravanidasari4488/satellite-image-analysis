# City-Agnostic Analysis Pipeline Redesign

## Overview

Complete redesign of the satellite image analysis pipeline for scientifically-valid, city-agnostic land-use statistics and risk assessment.

## Architecture

### Pipeline Flow

```
1. AOI Handler → Fetch exact city polygon from OSM
2. Image Clipping → Clip Sentinel-2 image to polygon boundary
3. Tiling Strategy → Generate 64×64 patches with sliding window
4. CNN Classifier → Classify each patch with EfficientNet
5. Class Mapper → Map EuroSAT 10-class → 5 high-level classes
6. Index Validator → Validate with spectral indices (hybrid fusion)
7. Aggregation Engine → Aggregate to city-level statistics
8. Risk Models → Calculate multi-factor flood/drought risks
```

## Key Components

### 1. AOI Handler (`aoi_handler.py`)
- Fetches exact administrative boundaries from OpenStreetMap
- Uses Nominatim for place search, Overpass API for boundaries
- Clips images to polygon (not bbox)
- Calculates actual city area in km²

### 2. Tiling Strategy (`tiling_strategy.py`)
- Sliding window: 64×64 tiles, configurable stride (default: 32 = 50% overlap)
- Handles masked pixels (outside polygon)
- Provides coverage statistics

### 3. CNN Patch Classifier (`cnn_patch_classifier.py`)
- Uses existing EfficientNet model (trained on EuroSAT)
- Classifies 64×64 patches (not full images)
- Batch processing for efficiency
- Fallback to Random Forest or KMeans if CNN unavailable

### 4. Class Mapper (`class_mapper.py`)
- Maps EuroSAT 10 classes → 5 high-level classes:
  - Forest, HerbaceousVegetation → Vegetation
  - River, SeaLake → Water
  - Residential, Industrial, Highway → Urban
  - AnnualCrop, PermanentCrop, Pasture → Agricultural
  - Low confidence / remaining → Barren
- Uses softmax averaging (not hard voting)

### 5. Index Validator (`index_validator.py`)
- Hybrid fusion: Validates CNN predictions with spectral indices
- NDVI → vegetation confidence
- NDWI → water validation
- NDBI → urban confirmation
- BSI → barren land
- Down-weights conflicting predictions

### 6. Aggregation Engine (`aggregation_engine.py`)
- Aggregates patch predictions statistically
- Converts pixel counts → km² → percentages
- Ensures percentages sum to 100% (±1%)
- Calculates confidence based on prediction consistency

### 7. Risk Models (`risk_models.py`)

#### Flood Risk
Multi-factor model:
```
FloodScore = 0.4 × RainfallIndex + 
             0.3 × UrbanIndex + 
             0.2 × LowElevationIndex + 
             0.1 × WaterProximityIndex
```

#### Drought Risk
Long-term indicators:
- 3-6 month rainfall anomaly
- NDVI trend deviation
- Water body surface area change

### 8. Main Pipeline (`city_analysis_pipeline.py`)
- Orchestrates entire workflow
- Handles fallback logic (CNN → RF → KMeans)
- Returns structured results

## API Endpoint

### `/analyze-city` (POST)

**Request:**
```json
{
  "location": "New York City",
  "image": "base64_encoded_RGB_image",
  "red_band": "base64_encoded_red_band",
  "green_band": "base64_encoded_green_band",
  "blue_band": "base64_encoded_blue_band",
  "nir_band": "base64_encoded_nir_band",
  "swir_band": "base64_encoded_swir_band",
  "image_bbox": [min_lng, min_lat, max_lng, max_lat],
  "weather_data": {
    "precipitation_7d": 50.0,
    "precipitation_30d": 200.0,
    "rainfall_anomaly_3m": -20.0,
    "rainfall_anomaly_6m": -15.0,
    "elevation_variance": 25.0,
    "ndvi_trend": -0.05,
    "water_area_change": -0.1,
    "satellite_date": "2024-01-15T00:00:00Z",
    "cloud_coverage": 5.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "land_cover": {
    "Vegetation": {"percentage": 25.5, "area_km2": 199.8},
    "Water": {"percentage": 15.2, "area_km2": 119.1},
    "Urban": {"percentage": 45.3, "area_km2": 355.1},
    "Agricultural": {"percentage": 8.5, "area_km2": 66.6},
    "Barren": {"percentage": 5.5, "area_km2": 43.1}
  },
  "flood_risk": {
    "level": "moderate",
    "score": 0.52,
    "components": {...},
    "metadata": {...}
  },
  "drought_risk": {
    "level": "low",
    "score": 0.28,
    "components": {...},
    "metadata": {...}
  },
  "confidence": 0.85,
  "metadata": {
    "satellite_date": "2024-01-15T00:00:00Z",
    "cloud_coverage": 5.0,
    "aoi_source": "osm_nominatim",
    "total_tiles_analyzed": 1247,
    "city_area_km2": 783.7,
    "classification_method": "cnn",
    "tile_size": 64,
    "stride": 32
  }
}
```

## Scientific Validation

### Why This Approach is Valid

1. **Exact Boundaries**: Uses administrative polygons, not approximations
2. **Patch-Level Classification**: CNN trained on 64×64 patches, used correctly
3. **Statistical Aggregation**: Proper averaging across patches
4. **Hybrid Fusion**: Combines ML and physics-based methods
5. **Multi-Factor Risks**: Uses multiple indicators, not single factors
6. **Normalized Scores**: Allows comparison across cities
7. **Confidence Metrics**: Provides uncertainty quantification

## Dependencies

New dependencies added:
- `shapely>=2.0.0` - Polygon operations
- `pyproj>=3.6.0` - Coordinate transformations

## Migration Guide

### For Backend Integration

Update the `/analyze` endpoint to call `/analyze-city`:

```javascript
// In backend/routes/satellite.js
const analysisResult = await callAIService('/analyze-city', {
  location: location,
  image: imageBase64,
  red_band: redBandBase64,
  green_band: greenBandBase64,
  blue_band: blueBandBase64,
  nir_band: nirBandBase64,
  swir_band: swirBandBase64,
  image_bbox: [min_lng, min_lat, max_lng, max_lat],
  weather_data: {
    precipitation_7d: weatherData.precipitation?.current || 0,
    precipitation_30d: weatherData.precipitation?.accumulated_30d || 0,
    // ... other weather data
  }
});
```

## Testing

Test with various cities:
- Small cities (< 100 km²)
- Large cities (> 1000 km²)
- Coastal cities (water bodies)
- Desert cities (barren land)
- Agricultural regions

## Performance

- Typical city (500 km²): ~1000-2000 tiles
- Processing time: 2-5 minutes (depends on city size)
- Batch size: 32 patches (configurable)

## Future Improvements

1. Parallel tile processing
2. GPU acceleration for CNN
3. Caching of OSM polygons
4. Incremental processing for very large cities
5. Multi-temporal analysis






