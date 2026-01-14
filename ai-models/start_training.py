#!/usr/bin/env python3
"""
Quick start script to train the land classification model
Run this to train the model for enhanced precision
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_model import LandClassificationTrainer, generate_synthetic_training_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Land Classification Model Training")
    logger.info("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize trainer
    trainer = LandClassificationTrainer(input_shape=(256, 256, 3), num_classes=5)
    
    # Generate training data
    logger.info("\nStep 1: Generating training data...")
    logger.info("Note: For production, replace with real labeled satellite imagery")
    images, labels = generate_synthetic_training_data(num_samples=2000)
    logger.info(f"Generated {len(images)} training samples")
    
    # Split data
    split_idx = int(len(images) * 0.8)
    X_train = images[:split_idx]
    y_train = labels[:split_idx]
    X_val = images[split_idx:]
    y_val = labels[split_idx:]
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    
    # Prepare data
    logger.info("\nStep 2: Preparing training data with augmentation...")
    X_train_prep, y_train_prep = trainer.prepare_training_data(X_train, y_train, augment=True)
    X_val_prep, y_val_prep = trainer.prepare_training_data(X_val, y_val, augment=False)
    logger.info("Data preparation complete")
    
    # Create model
    logger.info("\nStep 3: Creating advanced model with transfer learning...")
    trainer.create_advanced_model(use_transfer_learning=True)
    logger.info("Model architecture ready")
    
    # Train
    logger.info("\nStep 4: Starting training...")
    logger.info("This may take 30-60 minutes depending on your hardware")
    logger.info("Model will be saved automatically when validation accuracy improves")
    
    history = trainer.train(
        X_train_prep, y_train_prep,
        X_val_prep, y_val_prep,
        epochs=30,
        batch_size=16
    )
    
    # Save final model
    logger.info("\nStep 5: Saving trained model...")
    trainer.save_model('models/trained_land_classification.h5')
    
    # Print training summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2%}")
    logger.info(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    logger.info(f"\nModel saved to: models/trained_land_classification.h5")
    logger.info("\nThe model will now be automatically used for enhanced precision!")
    logger.info("For production, train on real labeled satellite imagery datasets.")
    logger.info("See README_TRAINING.md for details.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)





