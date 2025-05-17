"""
Script to train the damage detection model.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_mask_from_image(image):
    """
    Create a damage mask from the image.
    For now, we'll use a simple thresholding approach.
    Later, you can replace this with your actual mask creation logic.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply adaptive thresholding
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def load_and_preprocess_data(image_paths, target_size=(256, 256)):
    """
    Load and preprocess images and create masks.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing
        
    Returns:
        Tuple of (images, masks) as numpy arrays
    """
    images = []
    masks = []
    
    for img_path in image_paths:
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Create mask
        mask = create_mask_from_image(img)
        mask = cv2.resize(mask, target_size)
        mask = (mask > 127).astype(np.float32)  # Binarize
        
        # Normalize image
        img = img / 255.0
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def build_unet_model(input_shape=(256, 256, 3), num_classes=1):
    """
    Build U-Net model for damage detection.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    def conv_block(inputs, filters, kernel_size=3):
        x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def encoder_block(inputs, filters):
        x = conv_block(inputs, filters)
        p = layers.MaxPooling2D((2, 2))(x)
        return x, p
    
    def decoder_block(inputs, skip_features, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = conv_block(x, filters)
        return x
    
    # Input
    inputs = layers.Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bridge
    b1 = conv_block(p4, 1024)
    
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Output
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def train_model():
    """Train the damage detection model."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Data paths
    data_dir = "data/test_multi_label"
    
    # Get image paths
    image_paths = sorted(glob(os.path.join(data_dir, "*.jpg")))
    print(f"Found {len(image_paths)} images")
    
    # Split data
    train_img_paths, val_img_paths = train_test_split(
        image_paths, test_size=0.2, random_state=42
    )
    
    # Load and preprocess data
    print("Loading training data...")
    X_train, y_train = load_and_preprocess_data(train_img_paths)
    print("Loading validation data...")
    X_val, y_val = load_and_preprocess_data(val_img_paths)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Build model
    model = build_unet_model()
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            "models/damage_detection_model.h5",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        validation_data=(X_val, y_val),
        epochs=10,
        steps_per_epoch=len(X_train) // 8,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("\nTraining completed!")
    print("Model saved to models/damage_detection_model.h5")

if __name__ == "__main__":
    train_model() 