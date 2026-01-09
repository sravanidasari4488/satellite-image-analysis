"""
Training script for satellite image land classification model
Uses transfer learning and advanced CNN architecture for precise classification
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json
import logging
from typing import Dict, Tuple, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LandClassificationTrainer:
    """
    Trainer for land classification model using transfer learning
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.class_names = ['Forest', 'Water', 'Urban', 'Agricultural', 'Barren']
        self.model = None
        self.history = None
        
    def create_advanced_model(self, use_transfer_learning=True):
        """
        Create advanced CNN model with transfer learning for better accuracy
        """
        logger.info("Creating advanced model architecture...")
        
        if use_transfer_learning:
            # Use EfficientNetB0 as base (pre-trained on ImageNet)
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze early layers, fine-tune later layers
            base_model.trainable = True
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
            
        else:
            inputs = layers.Input(shape=self.input_shape)
            # Custom deep CNN architecture
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Use learning rate schedule for better training
        initial_learning_rate = 0.001
        lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        logger.info(f"Model created with {model.count_params()} parameters")
        return model
    
    def create_segmentation_model(self):
        """
        Create U-Net style segmentation model for pixel-level classification
        This provides more precise per-pixel classification
        """
        logger.info("Creating U-Net segmentation model...")
        
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (downsampling path)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        c4 = layers.Dropout(0.5)(c4)
        
        # Decoder (upsampling path)
        u5 = layers.UpSampling2D((2, 2))(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
        
        u6 = layers.UpSampling2D((2, 2))(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.UpSampling2D((2, 2))(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
        
        # Output layer - pixel-level classification
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(c7)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Segmentation model created with {model.count_params()} parameters")
        return model
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract advanced features from image for better classification
        Includes texture, edges, color statistics, etc.
        """
        features = []
        
        # Color features
        rgb_mean = np.mean(image, axis=(0, 1))
        rgb_std = np.std(image, axis=(0, 1))
        features.extend(rgb_mean)
        features.extend(rgb_std)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        hsv_mean = np.mean(hsv, axis=(0, 1))
        lab_mean = np.mean(lab, axis=(0, 1))
        features.extend(hsv_mean)
        features.extend(lab_mean)
        
        # Texture features using Local Binary Patterns
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        
        # Vegetation index (approximate NDVI from RGB)
        r = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        vvi = np.mean((g - r) / (g + r + 1e-10))
        features.append(vvi)
        
        return np.array(features)
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to increase training data diversity
        """
        # Random rotation
        angle = np.random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-10, 10)   # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        return image
    
    def prepare_training_data(self, images: list, labels: list, augment: bool = True):
        """
        Prepare training data with augmentation
        """
        X = []
        y = []
        
        for img, label in zip(images, labels):
            # Resize to model input size
            img_resized = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
            img_resized = img_resized.astype(np.float32) / 255.0
            
            # Convert label to one-hot
            label_onehot = tf.keras.utils.to_categorical(label, self.num_classes)
            
            X.append(img_resized)
            y.append(label_onehot)
            
            # Augment if requested
            if augment:
                img_aug = self.augment_image(img)
                img_aug_resized = cv2.resize(img_aug, (self.input_shape[0], self.input_shape[1]))
                img_aug_resized = img_aug_resized.astype(np.float32) / 255.0
                X.append(img_aug_resized)
                y.append(label_onehot)
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, use_segmentation=False):
        """
        Train the model
        """
        if self.model is None:
            if use_segmentation:
                self.create_segmentation_model()
            else:
                self.create_advanced_model(use_transfer_learning=True)
        
        # Callbacks for training
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Data augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            fill_mode='reflect'
        )
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.history = history
        logger.info("Training completed!")
        
        return history
    
    def save_model(self, filepath: str):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

def generate_synthetic_training_data(num_samples=1000):
    """
    Generate synthetic training data for initial model training
    In production, replace with real labeled satellite imagery
    """
    images = []
    labels = []
    
    # Class-specific color patterns
    class_patterns = {
        0: {'color': [34, 139, 34], 'variance': 30},  # Forest - Green
        1: {'color': [0, 100, 200], 'variance': 20},  # Water - Blue
        2: {'color': [128, 128, 128], 'variance': 25}, # Urban - Gray
        3: {'color': [255, 215, 0], 'variance': 40},  # Agricultural - Gold/Yellow
        4: {'color': [139, 69, 19], 'variance': 35}    # Barren - Brown
    }
    
    for _ in range(num_samples):
        class_idx = np.random.randint(0, 5)
        pattern = class_patterns[class_idx]
        
        # Create image with class-specific characteristics
        img = np.random.normal(
            pattern['color'],
            pattern['variance'],
            (256, 256, 3)
        ).astype(np.uint8)
        
        # Add texture
        noise = np.random.normal(0, 10, (256, 256, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add some structure
        if class_idx == 0:  # Forest - add tree-like patterns
            for _ in range(20):
                x, y = np.random.randint(0, 256, 2)
                cv2.circle(img, (x, y), np.random.randint(5, 15), 
                          tuple(pattern['color']), -1)
        elif class_idx == 1:  # Water - smoother texture
            img = cv2.GaussianBlur(img, (15, 15), 0)
        elif class_idx == 2:  # Urban - add rectangular structures
            for _ in range(10):
                x, y = np.random.randint(0, 200, 2)
                w, h = np.random.randint(20, 60, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), 
                             tuple(pattern['color']), -1)
        
        images.append(img)
        labels.append(class_idx)
    
    return images, labels

if __name__ == "__main__":
    # Initialize trainer
    trainer = LandClassificationTrainer(input_shape=(256, 256, 3), num_classes=5)
    
    # Generate or load training data
    logger.info("Generating training data...")
    images, labels = generate_synthetic_training_data(num_samples=2000)
    
    # Split data
    split_idx = int(len(images) * 0.8)
    X_train = images[:split_idx]
    y_train = labels[:split_idx]
    X_val = images[split_idx:]
    y_val = labels[split_idx:]
    
    # Prepare data
    X_train_prep, y_train_prep = trainer.prepare_training_data(X_train, y_train, augment=True)
    X_val_prep, y_val_prep = trainer.prepare_training_data(X_val, y_val, augment=False)
    
    # Create and train model
    trainer.create_advanced_model(use_transfer_learning=True)
    
    # Train
    history = trainer.train(
        X_train_prep, y_train_prep,
        X_val_prep, y_val_prep,
        epochs=100,
        batch_size=16
    )
    
    # Save model
    trainer.save_model('models/trained_land_classification.h5')
    
    logger.info("Training complete! Model saved to models/trained_land_classification.h5")
    logger.info("For production, train on real labeled satellite imagery datasets")

