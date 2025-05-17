import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_monument_recognition_model(num_classes=24, input_shape=(300, 300, 3)):
    """
    Creates an improved model for monument recognition using transfer learning.
    
    Args:
        num_classes (int): Number of monument classes to classify
        input_shape (tuple): Input image shape (height, width, channels)
    
    Returns:
        tf.keras.Model: Compiled model ready for training
    """
    # Load pre-trained ResNet50V2 model without top layers
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create the model
    model = models.Sequential([
        # Data augmentation layers
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomFlip("horizontal"),
        
        # Base model
        base_model,
        
        # Additional layers
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks():
    """
    Returns a list of callbacks for model training.
    """
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

def fine_tune_model(model, train_data, validation_data, epochs=20):
    """
    Fine-tunes the model by unfreezing some layers of the base model.
    
    Args:
        model (tf.keras.Model): The trained model
        train_data: Training data generator
        validation_data: Validation data generator
        epochs (int): Number of epochs for fine-tuning
    
    Returns:
        tf.keras.callbacks.History: Training history
    """
    # Find the ResNet50V2 base model in the Sequential model (robust search)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        print("Model layers:")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} ({layer.__class__.__name__})")
        raise ValueError("ResNet50V2 base model not found in the model.")
    base_model.trainable = True
    # Freeze the bottom layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Fine-tune the model
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=get_callbacks()
    )
    return history 