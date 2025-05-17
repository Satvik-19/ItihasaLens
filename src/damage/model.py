"""
Damage detection module for the ItihasaLens application.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from pathlib import Path

class DamageDetector:
    """Class for detecting damage in monument images."""
    
    def __init__(self, model_path="models/damage_detection_model.h5"):
        """
        Initialize the damage detector.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = load_model(model_path)
        self.target_size = (256, 256)
        
    def preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.target_size)
        
        # Normalize
        image_normalized = image_resized / 255.0
        
        return image_normalized
    
    def detect_damage(self, image, threshold=0.5):
        """
        Detect damage in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            threshold: Threshold for damage detection (default: 0.5)
            
        Returns:
            Tuple of (damage_mask, damage_overlay)
            - damage_mask: Binary mask of detected damage
            - damage_overlay: Original image with damage highlighted
        """
        # Store original size
        original_size = image.shape[:2]
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Add batch dimension
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Get prediction
        prediction = self.model.predict(processed_image)[0]
        
        # Resize prediction to original size
        prediction = cv2.resize(prediction, (original_size[1], original_size[0]))
        
        # Create binary mask
        damage_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # Find contours of damage
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create overlay
        overlay = image.copy()
        
        # Draw filled contours with semi-transparent red
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
        
        # Draw contour outlines in bright red
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        # Blend original and overlay
        alpha = 0.4  # Reduced transparency for better visibility
        damage_overlay = cv2.addWeighted(image, 1-alpha, overlay, alpha, 0)
        
        return damage_mask, damage_overlay
    
    def analyze_damage(self, image):
        """
        Analyze damage in an image and return detailed results.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Dictionary containing:
            - damage_mask: Binary mask of detected damage
            - damage_overlay: Original image with damage highlighted
            - damage_percentage: Percentage of image showing damage
            - damage_severity: Qualitative assessment of damage severity
        """
        # Get damage detection results
        damage_mask, damage_overlay = self.detect_damage(image)
        
        # Calculate damage percentage
        total_pixels = image.shape[0] * image.shape[1]
        damage_pixels = np.sum(damage_mask == 255)
        damage_percentage = (damage_pixels / total_pixels) * 100
        
        # Assess damage severity
        if damage_percentage < 5:
            severity = "Minimal"
        elif damage_percentage < 15:
            severity = "Moderate"
        elif damage_percentage < 30:
            severity = "Significant"
        else:
            severity = "Severe"
        
        return {
            "damage_mask": damage_mask,
            "damage_overlay": damage_overlay,
            "damage_percentage": damage_percentage,
            "damage_severity": severity
        }
    
    def save_results(self, image_path, output_dir="damage_results"):
        """
        Process an image and save the results.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing paths to saved results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Get base filename
        base_name = Path(image_path).stem
        
        # Analyze damage
        results = self.analyze_damage(image)
        
        # Save results
        cv2.imwrite(f"{output_dir}/{base_name}_original.jpg", image)
        cv2.imwrite(f"{output_dir}/{base_name}_mask.jpg", results["damage_mask"])
        cv2.imwrite(f"{output_dir}/{base_name}_overlay.jpg", results["damage_overlay"])
        
        # Create and save visualization
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(results["damage_mask"], cmap='gray')
        plt.title('Damage Mask')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(cv2.cvtColor(results["damage_overlay"], cv2.COLOR_BGR2RGB))
        plt.title(f'Damage Overlay\nSeverity: {results["damage_severity"]}\nDamage: {results["damage_percentage"]:.1f}%')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_analysis.png")
        plt.close()
        
        return {
            "original": f"{output_dir}/{base_name}_original.jpg",
            "mask": f"{output_dir}/{base_name}_mask.jpg",
            "overlay": f"{output_dir}/{base_name}_overlay.jpg",
            "analysis": f"{output_dir}/{base_name}_analysis.png",
            "damage_percentage": results["damage_percentage"],
            "damage_severity": results["damage_severity"]
        }

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = DamageDetector()
    
    # Test on a sample image
    test_image = "data/test_multi_label/sans-0_0_0_0- (1).jpg"
    results = detector.save_results(test_image)
    
    print(f"Damage Analysis Results:")
    print(f"Damage Percentage: {results['damage_percentage']:.1f}%")
    print(f"Damage Severity: {results['damage_severity']}")
    print(f"Results saved in: damage_results/") 