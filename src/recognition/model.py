"""
Monument recognition module using CNN for classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, List, Dict
import numpy as np
import os

# Monument class names from the notebook
MONUMENT_CLASS_NAMES = [
    'Ajanta Caves', 'Charar-E- Sharif', 'Chhota_Imambara', 'Ellora Caves', 'Fatehpur Sikri', 'Gateway of India',
    'Humayun_s Tomb', 'India gate pics', 'Khajuraho', 'Sun Temple Konark', 'alai_darwaza', 'alai_minar',
    'basilica_of_bom_jesus', 'charminar', 'golden temple', 'hawa mahal pics', 'iron_pillar', 'jamali_kamali_tomb',
    'lotus_temple', 'mysore_palace', 'qutub_minar', 'tajmahal', 'tanjavur temple', 'victoria memorial'
]

class MonumentRecognizer:
    def __init__(
        self,
        class_names: list = MONUMENT_CLASS_NAMES,
        input_shape: Tuple[int, int, int] = (300, 300, 3),
        model_path: str = None
    ):
        """
        Initialize the monument recognition model.
        
        Args:
            class_names: List of monument class names
            input_shape: Input image shape (height, width, channels)
            model_path: Path to pretrained model (optional)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.input_shape = input_shape
        
        print(f"Loading model from: {model_path}")
        try:
            if model_path and os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print("Model loaded successfully")
                print(f"Model input shape: {self.model.input_shape}")
                
                # Ensure we have valid input dimensions
                if hasattr(self.model, 'input_shape') and len(self.model.input_shape) >= 3:
                    h, w = self.model.input_shape[1:3]
                    if h and w:  # Check if dimensions are not None
                        self.input_shape = (int(h), int(w), 3)
                        print(f"Updated input shape to: {self.input_shape}")
                    else:
                        print("Warning: Invalid model input shape, using default")
                else:
                    print("Warning: Could not get model input shape, using default")
            else:
                print(f"Model file not found at {model_path}, building new model")
                self.model = self._build_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def _build_model(self) -> Model:
        """
        Build a simple CNN model for monument recognition.
        
        Returns:
            Compiled Keras model
        """
        model = tf.keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for model inference.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Ensure image is not None and has valid shape
            if image is None or len(image.shape) != 3:
                raise ValueError("Invalid image input")
            
            print(f"Input image shape: {image.shape}")
            
            # Ensure we have valid target dimensions
            target_height, target_width = self.input_shape[:2]
            if not target_height or not target_width:
                raise ValueError(f"Invalid target dimensions: {self.input_shape}")
            
            print(f"Target dimensions: {target_height}x{target_width}")
            
            # Calculate aspect ratio
            h, w = image.shape[:2]
            if h == 0 or w == 0:
                raise ValueError("Invalid image dimensions")
                
            aspect = w / h
            print(f"Image aspect ratio: {aspect}")
            
            # Resize while maintaining aspect ratio
            if aspect > 1:  # Wider than tall
                new_width = target_width
                new_height = int(target_width / aspect)
            else:  # Taller than wide
                new_height = target_height
                new_width = int(target_height * aspect)
            
            print(f"Resizing to: {new_height}x{new_width}")
            
            # Resize image
            image = tf.image.resize(image, (new_height, new_width))
            
            # Pad to target size
            image = tf.image.resize_with_pad(
                image,
                target_height,
                target_width,
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            # Normalize pixel values
            image = image / 255.0
            
            print(f"Final preprocessed image shape: {image.shape}")
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict monument class probabilities.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of class probabilities (monument name -> probability)
        """
        try:
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            print(f"Input tensor shape: {image.shape}")
            
            # Get predictions
            predictions = self.model.predict(image, verbose=0)[0]
            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions: {predictions}")
            
            # Get top 3 predictions
            top_indices = np.argsort(predictions)[-3:][::-1]
            print("\nTop 3 predictions:")
            for idx in top_indices:
                print(f"{self.class_names[idx]}: {predictions[idx]:.2%}")
            
            return {
                self.class_names[i]: float(prob)
                for i, prob in enumerate(predictions)
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise  # Re-raise to see the full error
    
    def fine_tune(self, train_data, val_data, epochs: int = 10):
        """
        Fine-tune the model on monument dataset.
        
        Args:
            train_data: Training data generator
            val_data: Validation data generator
            epochs: Number of training epochs
        """
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-20:]:
            layer.trainable = True
        
        # Recompile model with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs
        )
        
        return history 