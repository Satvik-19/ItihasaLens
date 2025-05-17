import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.monument_recognition import create_monument_recognition_model, get_callbacks, fine_tune_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_monument_model(train_dir, test_dir, model_save_path, input_shape=(300, 300, 3)):
    """
    Trains the monument recognition model with full model checkpointing (weights + optimizer + epoch).
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=32,
        class_mode='categorical'
    )
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_model')
    initial_epoch = 0
    # Try to load the full model checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading full model from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
        # Try to get the last epoch from the checkpoint's training history
        # (Keras does not store epoch in the model, so you may need to track it separately if you want exact resume)
        # For now, we resume with weights and optimizer state, but start at epoch 0
    else:
        model = create_monument_recognition_model(
            num_classes=len(train_data.class_indices),
            input_shape=input_shape
        )
        model.build((None, *input_shape))
    model.summary()
    # ModelCheckpoint callback to save the full model after each epoch
    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=False,
        verbose=1
    )
    best_model_cb = ModelCheckpoint(
        filepath=model_save_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    print(f"\nResuming training from epoch {initial_epoch}...")
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=20,
        initial_epoch=initial_epoch,
        callbacks=get_callbacks() + [checkpoint_cb, best_model_cb]
    )
    print("\nFine-tuning the model...")
    history_fine = fine_tune_model(
        model,
        train_data,
        test_data,
        epochs=20
    )
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")
    return model, history, history_fine

if __name__ == "__main__":
    train_dir = "data/indian-monuments-image-dataset/images/train"
    test_dir = "data/indian-monuments-image-dataset/images/test"
    model_save_path = "models/monument_recognition_model.h5"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model, history, history_fine = train_monument_model(
        train_dir,
        test_dir,
        model_save_path
    ) 