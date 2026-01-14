# Model Training Guide

## Overview
This system uses a hybrid approach combining:
1. **Trained ML Model** (EfficientNetB0 transfer learning) - For high-level classification
2. **Pixel-based Analysis** - For precise per-pixel classification
3. **Advanced Feature Extraction** - Texture, edges, color statistics

## Training the Model

### Quick Start
```bash
cd ai-models
python train_model.py
```

This will:
- Generate synthetic training data (2000 samples)
- Train an EfficientNetB0-based model with transfer learning
- Save the model to `models/trained_land_classification.h5`

### Using Real Training Data

For production, replace synthetic data with real labeled satellite imagery:

1. **Prepare your dataset:**
   - Organize images by class: `data/train/Forest/`, `data/train/Water/`, etc.
   - Use high-quality satellite imagery (Sentinel-2, Landsat, etc.)
   - Ensure balanced classes

2. **Update training script:**
   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
   train_datagen = ImageDataGenerator(
       rescale=1./255,
       rotation_range=15,
       width_shift_range=0.1,
       height_shift_range=0.1,
       horizontal_flip=True,
       zoom_range=0.1
   )
   
   train_generator = train_datagen.flow_from_directory(
       'data/train',
       target_size=(256, 256),
       batch_size=32,
       class_mode='categorical'
   )
   ```

3. **Train with real data:**
   ```python
   trainer.train(
       X_train, y_train,
       X_val, y_val,
       epochs=100,  # More epochs for real data
       batch_size=32
   )
   ```

## Model Architecture

### EfficientNetB0 Transfer Learning
- **Base**: Pre-trained on ImageNet (1.4M images, 1000 classes)
- **Fine-tuning**: Last 30 layers are trainable
- **Custom Head**: 
  - Global Average Pooling
  - Batch Normalization
  - Dropout (0.5, 0.3, 0.2)
  - Dense layers (512, 256, 5)

### U-Net Segmentation (Optional)
For pixel-level precision:
```python
trainer.create_segmentation_model()
trainer.train(..., use_segmentation=True)
```

## Enhanced Features

The system extracts:
- **Color Features**: RGB, HSV, LAB statistics
- **Texture Features**: Edge density, gradient magnitude
- **Vegetation Index**: VVI (Visible Vegetation Index)
- **Local Variance**: GLCM-like texture features

## Precision Improvements

1. **Hybrid Classification**: 70% ML model + 30% pixel-based
2. **Multi-scale Analysis**: Analyzes image at multiple resolutions
3. **Ensemble Methods**: Can combine multiple models
4. **Post-processing**: Refines results with domain knowledge

## Model Evaluation

After training, evaluate on test set:
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2%}")
```

## Production Deployment

1. Train model on real data
2. Save model: `trainer.save_model('models/production_model.h5')`
3. Update `land_classification.py` to load the model
4. The system will automatically use the trained model for enhanced precision

## Recommended Datasets

- **EuroSAT**: 27,000 labeled Sentinel-2 images
- **BigEarthNet**: 590,326 Sentinel-2 patches
- **UC Merced Land Use**: 21 classes, 100 images each
- **DeepGlobe Land Cover**: 1,146 satellite images

## Training Tips

1. **Data Augmentation**: Essential for generalization
2. **Learning Rate**: Start with 0.001, use ReduceLROnPlateau
3. **Early Stopping**: Prevents overfitting
4. **Class Balance**: Ensure balanced training data
5. **Transfer Learning**: Always use pre-trained models for satellite imagery

## Performance Metrics

- **Accuracy**: Overall classification accuracy
- **Per-class Precision/Recall**: For each land type
- **F1-Score**: Harmonic mean of precision and recall
- **IoU (Intersection over Union)**: For segmentation tasks





