# Google Earth Engine Setup Guide

This guide will help you set up Google Earth Engine (GEE) for more efficient satellite image processing in the Satellite Image Analysis application.

## Why Google Earth Engine?

Google Earth Engine provides several advantages over other satellite data sources:

1. **Free Access**: Access to petabytes of satellite imagery at no cost
2. **Efficient Processing**: Built-in cloud processing capabilities
3. **Multiple Datasets**: Access to Sentinel-2, Landsat, MODIS, and more
4. **Direct Index Calculation**: Calculate spectral indices (NDVI, NDWI, NDBI) directly without downloading full images
5. **Large-Scale Analysis**: Process large areas efficiently using Google's infrastructure

## Prerequisites

1. A Google account
2. Python 3.7 or higher
3. Internet connection

## Setup Steps

### 1. Install Dependencies

The `earthengine-api` package has already been added to `requirements.txt`. Install it:

```bash
cd ai-models
pip install -r requirements.txt
```

### 2. Get a Google Cloud Project ID

**Important**: Google Earth Engine requires a Google Cloud Project ID to work.

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Note your Project ID (not the project name)
4. Enable the Earth Engine API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Earth Engine API"
   - Click "Enable"

### 3. Authenticate Google Earth Engine

You have two options for authentication:

#### Option A: User Credentials (Recommended for Development)

1. Run the authentication command:
```bash
python -m earthengine authenticate
```

**Note**: If `earthengine` command is not found, use:
```bash
python -c "import ee; ee.Authenticate()"
```

2. This will open a browser window asking you to sign in with your Google account
3. Grant the necessary permissions
4. The credentials will be saved locally

#### Option B: Service Account (Recommended for Production)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Earth Engine API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Earth Engine API"
   - Click "Enable"
4. Create a service account:
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "Service Account"
   - Fill in the details and create
5. Download the service account key (JSON file)
6. Set environment variables:
```bash
export GEE_SERVICE_ACCOUNT="sravani-dasari@satellitanalysis.iam.gserviceaccount.com"
export GEE_CREDENTIALS_PATH="/path/to/service-account-key.json"
export GEE_PROJECT_ID="satellitanalysis"
```

### 4. Configure Environment Variables

Add the following to your `.env` file in the `ai-models` directory:

```env
# Google Earth Engine Configuration
GEE_PROJECT_ID=your-project-id  # REQUIRED - Your Google Cloud Project ID
GEE_SERVICE_ACCOUNT=your-service-account@project-id.iam.gserviceaccount.com  # Optional, for service account
GEE_CREDENTIALS_PATH=/path/to/credentials.json  # Optional, for service account
```

**Important**: `GEE_PROJECT_ID` is **REQUIRED** for GEE to work. You can find your project ID in the Google Cloud Console.

### 5. Verify Installation

Test the setup by running a simple Python script:

```python
import ee

try:
    ee.Initialize()
    print("✅ Google Earth Engine initialized successfully!")
    
    # Test by getting a simple image
    image = ee.Image('COPERNICUS/S2_SR_HARMONIZED/20220101T000241_20220101T000238_T60HVE')
    print(f"✅ Successfully accessed Sentinel-2 image: {image.get('system:id').getInfo()}")
except Exception as e:
    print(f"❌ Error: {e}")
```

## Usage

### Using GEE in the Application

The application now supports Google Earth Engine as the primary data source. The backend will automatically try to use GEE first, falling back to SentinelHub if GEE is not available.

### API Endpoints

The following new endpoints are available in the AI service:

1. **`POST /gee/fetch-image`**: Fetch satellite image using GEE
   ```json
   {
     "location": "New York",
     "bounds": [-74.1, 40.6, -73.9, 40.8],
     "start_date": "2024-01-01",
     "end_date": "2024-01-31",
     "cloud_cover": 20
   }
   ```

2. **`POST /gee/calculate-indices`**: Calculate spectral indices (NDVI, NDWI, NDBI)
   ```json
   {
     "bounds": [-74.1, 40.6, -73.9, 40.8],
     "start_date": "2024-01-01",
     "end_date": "2024-01-31",
     "cloud_cover": 20
   }
   ```

3. **`POST /gee/land-cover`**: Get land cover classification
   ```json
   {
     "bounds": [-74.1, 40.6, -73.9, 40.8],
     "start_date": "2024-01-01",
     "end_date": "2024-01-31",
     "cloud_cover": 20
   }
   ```

4. **`POST /gee/analyze`**: Comprehensive analysis using GEE
   ```json
   {
     "location": "New York",
     "bounds": [-74.1, 40.6, -73.9, 40.8],
     "start_date": "2024-01-01",
     "end_date": "2024-01-31",
     "cloud_cover": 20,
     "include_visualization": true
   }
   ```

### Backend Integration

The backend routes have been updated to automatically use GEE when available:

- **`POST /api/satellite/fetch`**: Now supports `use_gee` parameter (default: `true`)
- **`POST /api/satellite/analyze`**: Automatically tries GEE first, falls back to SentinelHub

## Benefits

1. **Cost Efficiency**: No API costs for satellite data access
2. **Faster Processing**: Cloud-based processing reduces local computation
3. **Better Scalability**: Handle larger areas and more requests
4. **More Datasets**: Access to multiple satellite sources (Sentinel-2, Landsat, MODIS)
5. **Direct Index Calculation**: Calculate indices server-side without downloading images

## Troubleshooting

### Error: "Google Earth Engine not initialized"

**Solution**: Run `earthengine authenticate` to set up credentials.

### Error: "Earth Engine API not enabled"

**Solution**: Enable the Earth Engine API in Google Cloud Console.

### Error: "Quota exceeded"

**Solution**: 
- Check your Google Cloud project quotas
- For production, consider using a service account with appropriate quotas
- Reduce the number of concurrent requests

### Performance Issues

**Solutions**:
- Use appropriate `scale` parameters (10m for Sentinel-2)
- Limit `maxPixels` for very large areas
- Use date ranges to limit the collection size
- Consider using median composites instead of single images

## Additional Resources

- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Earth Engine Python API](https://github.com/google/earthengine-api)
- [Sentinel-2 Dataset](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- [Earth Engine Code Editor](https://code.earthengine.google.com/) - For testing and visualization

## Migration from SentinelHub

The application automatically falls back to SentinelHub if GEE is not available. To fully migrate:

1. Complete the GEE setup above
2. Test the GEE endpoints
3. Monitor performance and costs
4. Once stable, you can disable SentinelHub (optional)

The system is designed to work with both services, so you can use them in parallel or switch between them as needed.

