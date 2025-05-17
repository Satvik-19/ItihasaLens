"""
Script to test the damage detection model.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image."""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = img.shape[:2]
    
    # Resize for model
    img_resized = cv2.resize(img, target_size)
    
    # Normalize
    img_normalized = img_resized / 255.0
    
    return img, img_normalized, original_size

def predict_damage(model, image_path):
    """Predict damage in an image."""
    # Load and preprocess image
    original_img, img, original_size = load_and_preprocess_image(image_path)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    # Predict
    prediction = model.predict(img)[0]
    
    # Resize prediction to original size
    prediction = cv2.resize(prediction, (original_size[1], original_size[0]))
    
    # Create binary mask
    mask = (prediction > 0.5).astype(np.uint8) * 255
    
    # Create colored overlay
    overlay = original_img.copy()
    overlay[mask == 255] = [255, 0, 0]  # Red for damage
    
    # Blend original and overlay
    alpha = 0.5
    output = cv2.addWeighted(original_img, 1-alpha, overlay, alpha, 0)
    
    return original_img, mask, output

def test_model():
    """Test the model on sample images."""
    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model("models/damage_detection_model.h5")
    
    # Get test images
    test_dir = "data/test_multi_label"
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]  # Test first 5 images
    
    # Create output directory
    os.makedirs("test_results", exist_ok=True)
    
    # Process each image
    for img_name in test_images:
        print(f"\nProcessing {img_name}...")
        img_path = os.path.join(test_dir, img_name)
        
        # Get predictions
        original, mask, overlay = predict_damage(model, img_path)
        
        # Save results
        base_name = os.path.splitext(img_name)[0]
        cv2.imwrite(f"test_results/{base_name}_original.jpg", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"test_results/{base_name}_mask.jpg", mask)
        cv2.imwrite(f"test_results/{base_name}_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Display results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(original)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(mask, cmap='gray')
        plt.title('Damage Mask')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(overlay)
        plt.title('Damage Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"test_results/{base_name}_combined.png")
        plt.close()
    
    print("\nTesting completed! Results saved in test_results/ directory.")

if __name__ == "__main__":
    test_model() 