# ML-Driven Multispectral Analysis Upgrade Guide

## Overview

This upgrade transforms the satellite-analysis project from pixel-based methods to ML-driven supervised learning using Sentinel-2 multispectral bands. The system now supports:

- **10-class land cover classification** (EuroSAT format)
- **Proper spectral indices**: NDVI, NDWI, NDBI from multispectral bands
- **CNN and Random Forest models** for classification
- **OpenWeather integration** for risk validation
- **OpenCage geolocation** for enhanced location services

## New Features

### 1. Multispectral Analysis Module (`multispectral_analysis.py`)

Provides ML-driven analysis using Sentinel-2 bands:

- **Spectral Indices Calculation**:
  - NDVI (Normalized Difference Vegetation Index)
  - NDWI (Normalized Difference Water Index) 
  - NDBI (Normalized Difference Built-up Index)

- **10-Class Land Cover Segmentation**:
  - AnnualCrop
  - Forest
  - HerbaceousVegetation
  - Highway
  - Industrial
  - Pasture
  - PermanentCrop
  - Residential
  - River
  - SeaLake

- **Risk Assessment**:
  - Flood risk using NDWI and elevation data
  - Drought risk using NDVI and precipitation data

### 2. Training Scripts

#### EuroSAT CNN Training (`train_eurosat.py`)

Train CNN models on EuroSAT dataset:

```bash
cd ai-models
python train_eurosat.py --data-dir data/eurosat --epochs 50 --batch-size 32
```

**Features**:
- EfficientNetB0 or ResNet50 transfer learning
- Data augmentation
- Early stopping and learning rate scheduling
- Saves best model automatically

#### Random Forest Training (`train_random_forest.py`)

Alternative classifier for faster training:

```bash
cd ai-models
python train_random_forest.py --data-dir data/eurosat --n-estimators 100
```

**Features**:
- Uses spectral indices and color features
- Faster training than CNNs
- Good for comparison and baseline

### 3. New API Endpoints

#### `/multispectral-analyze` (Flask AI Service)

Perform ML-driven multispectral analysis:

```json
POST /multispectral-analyze
{
  "image": "base64_encoded_rgb_image",
  "nir_band": "base64_encoded_nir_band" (optional),
  "swir_band": "base64_encoded_swir_band" (optional),
  "include_indices": true,
  "include_segmentation": true
}
```

**Response**:
```json
{
  "success": true,
  "indices": {
    "ndvi": {"mean": 0.45, "min": -0.1, "max": 0.8},
    "ndwi": {"mean": 0.2, "min": -0.3, "max": 0.6},
    "ndbi": {"mean": 0.1, "min": -0.2, "max": 0.4}
  },
  "landcover_segmentation": {
    "segmentation": [...],
    "statistics": {...}
  },
  "flood_risk": {...},
  "drought_risk": {...}
}
```

#### `/calculate-indices` (Flask AI Service)

Calculate spectral indices from bands:

```json
POST /calculate-indices
{
  "red_band": "base64_encoded_red",
  "green_band": "base64_encoded_green",
  "nir_band": "base64_encoded_nir",
  "swir_band": "base64_encoded_swir" (optional)
}
```

#### `/analyze-multispectral` (Backend API)

Enhanced analysis with OpenWeather validation:

```json
POST /api/satellite/analyze-multispectral
{
  "location": "Mumbai, India",
  "image": "base64_encoded_image" (optional),
  "nir_band": "base64_encoded_nir" (optional),
  "swir_band": "base64_encoded_swir" (optional)
}
```

**Features**:
- Automatic geocoding via OpenCage
- Weather data validation via OpenWeather
- Risk assessment with weather context
- Automatic report saving

## Setup Instructions

### 1. Install Dependencies

```bash
cd ai-models
.\venv\Scripts\Activate.ps1
pip install tensorflow scikit-learn opencv-python numpy pillow
```

### 2. Download EuroSAT Dataset

**Option A: Direct Download**
```bash
# Download from GitHub
git clone https://github.com/phelber/eurosat.git
# Or download zip from: https://madm.dfki.de/files/sentinel/EuroSAT.zip
```

**Option B: Using Python**
```python
# The dataset will be downloaded automatically during training
# Or use the download instructions in train_eurosat.py
```

**Expected Structure**:
```
data/eurosat/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

### 3. Train Models

**Train CNN Model**:
```bash
cd ai-models
python train_eurosat.py --data-dir data/eurosat --epochs 50
```

**Train Random Forest**:
```bash
cd ai-models
python train_random_forest.py --data-dir data/eurosat --n-estimators 100
```

### 4. Configure Environment Variables

Ensure these are set in `backend/.env`:

```env
# OpenWeather API (for risk validation)
OPENWEATHER_API_KEY=your_openweather_api_key

# OpenCage Geocoding (for location services)
OPENCAGE_API_KEY=your_opencage_api_key

# AI Service URL
AI_SERVICE_URL=http://localhost:5001
```

## Usage Examples

### Example 1: Basic Multispectral Analysis

```python
import requests
import base64

# Load image
with open('satellite_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Call API
response = requests.post('http://localhost:5001/multispectral-analyze', json={
    'image': image_data,
    'include_indices': True,
    'include_segmentation': True
})

result = response.json()
print(f"NDVI Mean: {result['indices']['ndvi']['mean']}")
print(f"Flood Risk: {result['flood_risk']['level']}")
```

### Example 2: Enhanced Analysis with Weather Validation

```javascript
// Frontend/Backend call
const response = await fetch('/api/satellite/analyze-multispectral', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    location: 'Mumbai, India',
    image: base64ImageData
  })
});

const analysis = await response.json();
console.log('Weather Validated Risk:', analysis.risk_assessment);
```

## Model Performance

### Expected Accuracy

- **CNN (EfficientNetB0)**: 85-95% accuracy on EuroSAT
- **Random Forest**: 75-85% accuracy on EuroSAT
- **Pixel-based (fallback)**: 70-80% accuracy

### Training Time

- **CNN**: 30-60 minutes (depending on hardware)
- **Random Forest**: 5-15 minutes

## Integration Points

### OpenWeather Integration

Weather data is used to validate and adjust risk assessments:

- **Flood Risk**: Enhanced with precipitation data
- **Drought Risk**: Enhanced with humidity and temperature
- **Validation**: Real-time weather conditions validate satellite-derived risks

### OpenCage Integration

Geolocation services provide:

- **Forward Geocoding**: Location names → Coordinates
- **Reverse Geocoding**: Coordinates → Address
- **Enhanced Metadata**: Country, region, administrative boundaries

## Migration from Pixel-Based

The system maintains backward compatibility:

1. **Legacy endpoints** (`/classify`, `/batch-analyze`) still work
2. **Automatic fallback** to pixel-based if ML models unavailable
3. **Gradual migration** - use new endpoints alongside old ones

## API Comparison

| Feature | Pixel-Based | ML-Driven Multispectral |
|---------|-------------|-------------------------|
| Classes | 5 | 10 (EuroSAT) |
| NDVI | Approximated | Real from NIR |
| NDWI | No | Yes |
| NDBI | No | Yes |
| Accuracy | 70-80% | 85-95% |
| Weather Validation | No | Yes |
| Model Training | Not required | Required |

## Troubleshooting

### Model Not Loading

- Check model file exists: `models/multispectral_landcover_model.h5`
- Verify TensorFlow is installed: `pip install tensorflow`
- Check model compatibility with TensorFlow version

### Dataset Issues

- Verify EuroSAT dataset structure matches expected format
- Check image formats (JPG/PNG supported)
- Ensure sufficient disk space (EuroSAT is ~2.5GB)

### API Errors

- Verify AI service is running on port 5001
- Check environment variables are set correctly
- Review logs for detailed error messages

## Next Steps

1. **Train on Real Data**: Replace synthetic data with labeled satellite imagery
2. **Fine-tune Models**: Adjust hyperparameters for your specific use case
3. **Add More Indices**: Implement additional spectral indices (EVI, SAVI, etc.)
4. **Ensemble Methods**: Combine CNN and Random Forest predictions
5. **Real-time Processing**: Optimize for production deployment

## References

- **EuroSAT Dataset**: https://github.com/phelber/eurosat
- **Sentinel-2 Bands**: https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi
- **Spectral Indices**: https://www.usgs.gov/landsat-missions/landsat-normalized-difference-vegetation-index






