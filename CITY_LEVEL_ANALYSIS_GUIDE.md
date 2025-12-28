# City-Level Land Cover Analysis - Complete Guide

## Overview

This implementation provides **city-level** land cover analysis using Google Earth Engine with:
- ✅ Exact city boundary polygons (not bounding boxes)
- ✅ Region-based classification over full city area
- ✅ Accurate area calculations in km² using GEE reducers
- ✅ Percentages that sum to ~100%
- ✅ Scientifically-justified classification thresholds
- ✅ Hackathon-ready, modular code

## Architecture

### Components

1. **`city_level_gee_analysis.py`** - Core GEE analysis engine
   - Handles polygon-to-geometry conversion
   - Sentinel-2 image loading with cloud masking
   - Spectral index calculation (NDVI, NDWI, NDBI, EVI)
   - ML-based land cover classification
   - Accurate area statistics using GEE reducers

2. **`city_gee_endpoint.py`** - Flask API endpoint
   - `/gee/analyze-city` endpoint
   - Validates input and calls analysis engine

3. **`backend/routes/city-gee.js`** - Backend route
   - `/api/city-gee/analyze` endpoint
   - Fetches city boundaries from OpenCage
   - Calls Python GEE service
   - Returns formatted results

## Methodology

### Classification Rules (Scientifically Justified)

Based on peer-reviewed literature:

1. **Water**: NDWI > 0.3 (McFeeters, 1996)
2. **Forest/Vegetation**: NDVI > 0.4 AND NDWI ≤ 0.3 (Tucker, 1979)
3. **Urban/Built-up**: NDBI > 0.1 OR (NDVI < 0.2 AND NDWI < 0.1) (Zha et al., 2003)
4. **Agriculture**: 0.2 < NDVI ≤ 0.4 AND NDBI ≤ 0.1
5. **Barren**: NDVI ≤ 0.2 AND NDWI ≤ 0.1 AND NDBI ≤ 0.1

### Area Calculation

- Uses GEE `reduceRegion` with `frequencyHistogram` reducer
- Pixel scale: 10m (Sentinel-2 resolution)
- Pixel area: 10m × 10m = 100 m² = 0.0001 km²
- Total area = pixel_count × pixel_area_km²
- Percentage = (class_pixels / total_pixels) × 100

### Data Source

- **Collection**: `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Surface Reflectance)
- **Cloud masking**: QA60 band (cloud and cirrus bit masks)
- **Temporal composite**: Median composite (reduces cloud contamination)
- **Cloud cover threshold**: Default 20% (configurable)

## API Usage

### Endpoint: `POST /api/city-gee/analyze`

**Request:**
```json
{
  "location": "New York",
  "start_date": "2024-01-01",  // Optional
  "end_date": "2024-01-31",    // Optional
  "cloud_cover_threshold": 20,  // Optional, default: 20
  "title": "Analysis Title"     // Optional
}
```

**Response:**
```json
{
  "success": true,
  "city_name": "New York",
  "location": {
    "address": "New York, NY, USA",
    "coordinates": { "latitude": 40.7128, "longitude": -74.0060 },
    "polygon": [[lng1, lat1], [lng2, lat2], ...],
    "bounds": { ... }
  },
  "land_classification": {
    "water": {
      "percentage": 15.23,
      "areaKm2": 45.67,
      "pixels": 456700
    },
    "forest": {
      "percentage": 25.45,
      "areaKm2": 76.34,
      "pixels": 763400
    },
    "urban": {
      "percentage": 40.12,
      "areaKm2": 120.36,
      "pixels": 1203600
    },
    "agricultural": {
      "percentage": 15.20,
      "areaKm2": 45.60,
      "pixels": 456000
    },
    "barren": {
      "percentage": 4.00,
      "areaKm2": 12.00,
      "pixels": 120000
    },
    "_method": "gee_city_level",
    "_totalAreaKm2": 299.97,
    "_percentageSum": 100.00
  },
  "summary": {
    "total_area_km2": 299.97,
    "total_pixels": 2999700,
    "percentage_sum": 100.00
  },
  "methodology": {
    "classification_method": "ML-based with scientific thresholds",
    "data_source": "Sentinel-2 Surface Reflectance",
    "cloud_cover_threshold": 20,
    "pixel_scale_m": 10,
    "indices_used": ["NDVI", "NDWI", "NDBI", "EVI"]
  }
}
```

## Python Backend Endpoint

### Endpoint: `POST /gee/analyze-city`

**Request:**
```json
{
  "city_name": "New York",
  "polygon_coords": [[-74.1, 40.6], [-73.9, 40.6], [-73.9, 40.8], [-74.1, 40.8], [-74.1, 40.6]],
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "cloud_cover_threshold": 20
}
```

**Note**: `polygon_coords` must be in `[lng, lat]` format (GeoJSON convention).

## Code Structure

### Key Functions

1. **`polygon_to_ee_geometry()`** - Converts polygon coordinates to GEE Geometry
2. **`get_sentinel2_image()`** - Loads and masks Sentinel-2 imagery
3. **`calculate_spectral_indices()`** - Computes NDVI, NDWI, NDBI, EVI
4. **`classify_land_cover_ml()`** - ML-based classification with thresholds
5. **`calculate_area_statistics()`** - Accurate area calculation using reducers
6. **`analyze_city()`** - Complete pipeline orchestration

### Error Handling

- Validates polygon coordinates format
- Checks GEE initialization
- Handles cloud-free image availability
- Provides clear error messages

## Testing

### Test a City Analysis

```bash
# Using curl
curl -X POST http://localhost:5000/api/city-gee/analyze \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "location": "New York"
  }'
```

### Verify Results

- Check `percentage_sum` ≈ 100%
- Verify `total_area_km2` is reasonable for the city
- Ensure all classes have non-negative values
- Validate percentages sum correctly

## Performance Considerations

- **Processing time**: 30-120 seconds per city (depends on size)
- **Timeout**: 2 minutes for backend requests
- **Scale**: 10m resolution (Sentinel-2 native)
- **Max pixels**: 1e9 (handles large cities)

## Limitations

1. **Polygon availability**: Requires OpenCage to return polygon (not all locations)
2. **Cloud cover**: May need to adjust date range for cloud-free imagery
3. **Temporal resolution**: Uses median composite over date range
4. **Classification**: Threshold-based (can be enhanced with ML models)

## Future Enhancements

1. **ML Model Integration**: Use pre-trained Random Forest in GEE
2. **Temporal Analysis**: Multi-date comparison
3. **Custom Classifications**: User-defined class thresholds
4. **Export Visualization**: Generate classified map images
5. **Batch Processing**: Analyze multiple cities

## Scientific References

- McFeeters, S. K. (1996). The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.
- Tucker, C. J. (1979). Red and photographic infrared linear combinations for monitoring vegetation.
- Zha, Y., et al. (2003). Use of normalized difference built-up index in automatically mapping urban areas from TM imagery.

## Support

For issues or questions:
1. Check GEE initialization: `python ai-models/test_gee_integration.py`
2. Verify polygon format: Must be `[lng, lat]` pairs
3. Check cloud coverage: Adjust date range if needed
4. Review logs: Check Flask and Node.js console output

