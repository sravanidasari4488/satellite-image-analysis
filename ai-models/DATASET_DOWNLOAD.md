# EuroSAT Dataset Download Guide

## Quick Download

The EuroSAT dataset is required for training the ML models. Follow these steps:

### Option 1: Direct Download (Recommended)

1. **Download the dataset:**
   - Direct link: https://madm.dfki.de/files/sentinel/EuroSAT.zip
   - File size: ~2.5 GB
   - Contains: 27,000 labeled Sentinel-2 images across 10 land cover classes

2. **Extract the zip file:**
   ```powershell
   # Extract EuroSAT.zip to a temporary location first
   Expand-Archive -Path EuroSAT.zip -DestinationPath temp_extract
   ```

3. **Copy class folders to the correct location:**
   ```powershell
   # Navigate to the extracted folder
   cd temp_extract\EuroSAT
   
   # Copy all class folders to ai-models/data/eurosat/
   Copy-Item -Path AnnualCrop,Forest,HerbaceousVegetation,Highway,Industrial,Pasture,PermanentCrop,Residential,River,SeaLake -Destination ..\..\ai-models\data\eurosat\ -Recurse
   ```

### Option 2: Using Python Script

Create a simple download script:

```python
import urllib.request
import zipfile
import os
from pathlib import Path

# Download URL
url = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
data_dir = Path("data/eurosat")
data_dir.mkdir(parents=True, exist_ok=True)

# Download
print("Downloading EuroSAT dataset (2.5 GB)...")
zip_path = data_dir / "EuroSAT.zip"
urllib.request.urlretrieve(url, zip_path)

# Extract
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir.parent)

# Clean up zip file
zip_path.unlink()
print("Done! Dataset ready at:", data_dir)
```

### Option 3: Manual Download from GitHub

1. Visit: https://github.com/phelber/eurosat
2. Follow the download instructions on the repository
3. Extract to `ai-models/data/eurosat/`

## Expected Directory Structure

After downloading, your directory should look like this:

```
ai-models/
└── data/
    └── eurosat/
        ├── AnnualCrop/
        │   ├── AnnualCrop_1.jpg
        │   ├── AnnualCrop_2.jpg
        │   └── ... (3000 images)
        ├── Forest/
        │   ├── Forest_1.jpg
        │   └── ... (3000 images)
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/
```

## Verify Dataset

After downloading, verify the dataset structure:

```powershell
cd ai-models
python -c "from pathlib import Path; import os; data_dir = Path('data/eurosat'); classes = ['AnnualCrop','Forest','HerbaceousVegetation','Highway','Industrial','Pasture','PermanentCrop','Residential','River','SeaLake']; found = [(c, len(list((data_dir/c).glob('*.jpg')))) for c in classes if (data_dir/c).exists()]; print(f'Found {len(found)} classes:'); [print(f'  {c}: {count} images') for c, count in found]"
```

Or run the training script - it will now validate the dataset and show what classes are found:

```powershell
python train_eurosat.py --data-dir data/eurosat
```

## Dataset Information

- **Total Images**: ~27,000
- **Classes**: 10 land cover types
- **Image Size**: 64x64 pixels
- **Format**: RGB JPEG images
- **Source**: Sentinel-2 satellite imagery
- **License**: See LICENSE file in dataset

## Classes

1. **AnnualCrop** - Annual crop fields
2. **Forest** - Forest areas
3. **HerbaceousVegetation** - Grasslands and herbaceous vegetation
4. **Highway** - Highway infrastructure
5. **Industrial** - Industrial areas
6. **Pasture** - Pasture land
7. **PermanentCrop** - Permanent crop fields (orchards, vineyards)
8. **Residential** - Residential areas
9. **River** - River water bodies
10. **SeaLake** - Sea and lake water bodies

## Troubleshooting

### Issue: "No images found"

**Solution**: Make sure the class folders are directly inside `data/eurosat/`, not nested in another folder.

### Issue: Download is slow

**Solution**: The dataset is 2.5 GB. Use a stable internet connection. You can also use a download manager.

### Issue: Extraction fails

**Solution**: Make sure you have enough disk space (at least 5 GB free). The extracted dataset is larger than the zip file.

### Issue: Wrong folder structure

**Solution**: The training script expects:
```
data/eurosat/ClassName/image.jpg
```

NOT:
```
data/eurosat/EuroSAT/ClassName/image.jpg
```

If you have the nested structure, move the class folders up one level.

## After Download

Once the dataset is downloaded and verified, you can train the model:

```powershell
cd ai-models
.\venv\Scripts\Activate.ps1
python train_eurosat.py --data-dir data/eurosat --epochs 50
```

The training will:
- Load images from all 10 classes
- Split into train/validation sets (80/20)
- Train EfficientNetB0 model
- Save best model to `models/multispectral_landcover_model.h5`






