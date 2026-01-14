"""
Training script for EuroSAT dataset
Trains CNN model on 10-class land cover classification using Sentinel-2 imagery
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import logging
from pathlib import Path
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EuroSAT classes
EUROSAT_CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

class EuroSATTrainer:
    """
    Trainer for EuroSAT 10-class land cover classification
    """
    
    def __init__(self, data_dir: str = 'data/eurosat', input_shape=(64, 64, 3)):
        """
        Initialize trainer
        
        Args:
            data_dir: Directory containing EuroSAT dataset
            input_shape: Input image shape (EuroSAT uses 64x64)
        """
        self.data_dir = Path(data_dir)
        self.input_shape = input_shape
        self.num_classes = len(EUROSAT_CLASSES)
        self.class_names = EUROSAT_CLASSES
        self.model = None
        self.history = None
        
    def create_model(self, base_model_type: str = 'efficientnet', 
                    use_transfer_learning: bool = True):
        """
        Create CNN model with transfer learning
        
        Args:
            base_model_type: 'efficientnet' or 'resnet'
            use_transfer_learning: Whether to use pre-trained weights
        """
        logger.info(f"Creating {base_model_type} model with transfer learning={use_transfer_learning}")
        
        if base_model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet' if use_transfer_learning else None,
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            base_model = ResNet50(
                weights='imagenet' if use_transfer_learning else None,
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze early layers, fine-tune later layers
        if use_transfer_learning:
            base_model.trainable = True
            # Fine-tune last 30 layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
        
        # Add custom classification head
        inputs = base_model.input
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        logger.info("Model created successfully")
        self.model.summary()
    
    def prepare_data_generators(self, train_dir: str, val_dir: Optional[str] = None,
                               batch_size: int = 32) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Prepare data generators with augmentation
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory (optional)
            batch_size: Batch size
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        if val_dir:
            val_generator = val_datagen.flow_from_directory(
                val_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )
        else:
            val_generator = None
        
        if train_generator.samples == 0:
            raise ValueError(
                f"No images found in {train_dir}. "
                "Please ensure the directory contains class subdirectories with images."
            )
        
        logger.info(f"Training samples: {train_generator.samples}")
        if val_generator:
            logger.info(f"Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator
    
    def train(self, train_generator: ImageDataGenerator,
             val_generator: Optional[ImageDataGenerator] = None,
             epochs: int = 50,
             steps_per_epoch: Optional[int] = None,
             validation_steps: Optional[int] = None):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs: Number of epochs
            steps_per_epoch: Steps per epoch (default: samples // batch_size)
            validation_steps: Validation steps
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy' if val_generator else 'accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_generator else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                'models/best_eurosat_model.h5',
                monitor='val_accuracy' if val_generator else 'accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.CSVLogger('models/training_log.csv')
        ]
        
        # Calculate steps
        if steps_per_epoch is None:
            steps_per_epoch = train_generator.samples // train_generator.batch_size
        
        if val_generator and validation_steps is None:
            validation_steps = val_generator.samples // val_generator.batch_size
        
        logger.info("Starting training...")
        logger.info(f"Epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
        
        # Train
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Training completed!")
        return self.history
    
    def evaluate(self, test_generator: ImageDataGenerator):
        """Evaluate model on test set"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        logger.info("Evaluating model...")
        results = self.model.evaluate(test_generator, verbose=1)
        
        logger.info(f"Test Loss: {results[0]:.4f}")
        logger.info(f"Test Accuracy: {results[1]:.4%}")
        if len(results) > 2:
            logger.info(f"Top-K Accuracy: {results[2]:.4%}")
        
        return results
    
    def save_model(self, filepath: str = 'models/multispectral_landcover_model.h5'):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

def validate_dataset(data_dir: Path) -> bool:
    """
    Validate that dataset directory contains required class folders with images
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        True if dataset is valid, False otherwise
    """
    if not data_dir.exists():
        return False
    
    # Check if at least some class folders exist
    found_classes = []
    for class_name in EUROSAT_CLASSES:
        class_dir = data_dir / class_name
        if class_dir.exists() and class_dir.is_dir():
            # Check if folder contains image files
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.tif'))
            if len(image_files) > 0:
                found_classes.append((class_name, len(image_files)))
    
    if len(found_classes) == 0:
        return False
    
    logger.info(f"Found {len(found_classes)} classes with images:")
    for class_name, count in found_classes:
        logger.info(f"  - {class_name}: {count} images")
    
    return True

def download_eurosat_dataset(data_dir: str = 'data/eurosat'):
    """
    Instructions for downloading EuroSAT dataset
    
    Args:
        data_dir: Directory to save dataset
    """
    logger.error("=" * 60)
    logger.error("EuroSAT Dataset Not Found!")
    logger.error("=" * 60)
    logger.error("The dataset directory exists but doesn't contain the required class folders.")
    logger.error("")
    logger.error("Download Instructions:")
    logger.error("1. Download from: https://github.com/phelber/eurosat")
    logger.error("2. Or use direct link: https://madm.dfki.de/files/sentinel/EuroSAT.zip")
    logger.error("3. Extract the zip file")
    logger.error("4. Copy the class folders to: " + str(Path(data_dir).absolute()))
    logger.error("")
    logger.error("Expected structure:")
    logger.error("   data/eurosat/")
    logger.error("   ├── AnnualCrop/")
    logger.error("   │   ├── image1.jpg")
    logger.error("   │   └── ...")
    logger.error("   ├── Forest/")
    logger.error("   ├── HerbaceousVegetation/")
    logger.error("   ├── Highway/")
    logger.error("   ├── Industrial/")
    logger.error("   ├── Pasture/")
    logger.error("   ├── PermanentCrop/")
    logger.error("   ├── Residential/")
    logger.error("   ├── River/")
    logger.error("   └── SeaLake/")
    logger.error("")
    logger.error("After downloading, run the training script again.")
    logger.error("=" * 60)

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EuroSAT land cover classification model')
    parser.add_argument('--data-dir', type=str, default='data/eurosat',
                       help='Path to EuroSAT dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--base-model', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet'],
                       help='Base model architecture')
    parser.add_argument('--no-transfer-learning', action='store_true',
                       help='Train from scratch (no transfer learning)')
    
    args = parser.parse_args()
    
    # Check if dataset exists and is valid
    data_path = Path(args.data_dir)
    if not data_path.exists():
        logger.error(f"Dataset directory not found at {args.data_dir}")
        download_eurosat_dataset(args.data_dir)
        return
    
    # Validate dataset structure
    if not validate_dataset(data_path):
        logger.error(f"Dataset directory exists but doesn't contain valid class folders with images.")
        download_eurosat_dataset(args.data_dir)
        return
    
    # Initialize trainer
    trainer = EuroSATTrainer(data_dir=str(data_path))
    
    # Create model
    trainer.create_model(
        base_model_type=args.base_model,
        use_transfer_learning=not args.no_transfer_learning
    )
    
    # Split data (80% train, 20% val)
    # For EuroSAT, you can manually split or use the full dataset
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    # If split directories don't exist, use full dataset for training
    if not train_dir.exists():
        train_dir = data_path
        val_dir = None
        logger.warning("Using full dataset for training (no validation split)")
    
    # Prepare data generators
    train_gen, val_gen = trainer.prepare_data_generators(
        str(train_dir),
        str(val_dir) if val_dir and val_dir.exists() else None,
        batch_size=args.batch_size
    )
    
    # Validate generators have data
    if train_gen.samples == 0:
        logger.error("No training images found! Please check your dataset structure.")
        download_eurosat_dataset(args.data_dir)
        return
    
    # Train
    trainer.train(
        train_gen,
        val_gen,
        epochs=args.epochs
    )
    
    # Save final model
    trainer.save_model()
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    if trainer.history:
        final_acc = trainer.history.history['accuracy'][-1]
        logger.info(f"Final Training Accuracy: {final_acc:.2%}")
        if val_gen:
            final_val_acc = trainer.history.history['val_accuracy'][-1]
            logger.info(f"Final Validation Accuracy: {final_val_acc:.2%}")
    logger.info("Model saved to: models/multispectral_landcover_model.h5")

if __name__ == "__main__":
    main()

