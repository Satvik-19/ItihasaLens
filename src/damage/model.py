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
    
    def detect_damage(self, image, threshold=0.65):  # Lowered threshold for better sensitivity
        """
        Detect damage in an image.
        
        Args:
            image: Input image (BGR format from OpenCV)
            threshold: Threshold for damage detection (default: 0.65)
            
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
        
        # Apply more sophisticated thresholding
        prediction = (prediction > threshold).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)  # Reduced kernel size for better detail preservation
        damage_mask = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, kernel)
        damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of damage
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise) with reduced minimum area
        min_contour_area = 50  # Reduced minimum area for better detection of small damage
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        
        # Create overlay
        overlay = image.copy()
        
        # Draw filled contours with semi-transparent red
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
        
        # Draw contour outlines in bright red
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        
        # Blend original and overlay
        alpha = 0.5  # Increased visibility of damage
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
            - damage_patterns: List of detected damage patterns
            - preventive_measures: List of recommended preventive measures
        """
        # Get damage detection results
        damage_mask, damage_overlay = self.detect_damage(image)
        
        # Calculate damage percentage
        total_pixels = image.shape[0] * image.shape[1]
        damage_pixels = np.sum(damage_mask == 255)
        damage_percentage = (damage_pixels / total_pixels) * 100
        
        # Find contours for more detailed analysis
        contours, _ = cv2.findContours(damage_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate average damage area and distribution
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            avg_area = np.mean(areas)
            max_area = max(areas)
            area_std = np.std(areas)  # Standard deviation of damage areas
            
            # Analyze damage patterns
            damage_patterns = self._analyze_damage_patterns(contours, areas, image)
        else:
            avg_area = 0
            max_area = 0
            area_std = 0
            damage_patterns = []
        
        # More sophisticated severity assessment
        if damage_percentage < 0.5 or (avg_area < 150 and max_area < 400):  # Minimal damage
            severity = "Minimal"
        elif damage_percentage < 3 or (avg_area < 400 and max_area < 1000):  # Moderate damage
            severity = "Moderate"
        elif damage_percentage < 10 or (avg_area < 1000 and max_area < 2000):  # Significant damage
            severity = "Significant"
        else:  # Severe damage
            severity = "Severe"
        
        # Adjust severity based on damage distribution
        if area_std > 1000 and len(contours) > 5:  # Highly concentrated damage
            severity = "Severe"
        
        # Get preventive measures based on damage patterns and severity
        preventive_measures = self._get_preventive_measures(damage_patterns, severity, damage_percentage)
        
        return {
            "damage_mask": damage_mask,
            "damage_overlay": damage_overlay,
            "damage_percentage": damage_percentage,
            "damage_severity": severity,
            "avg_damage_area": avg_area,
            "max_damage_area": max_area,
            "damage_distribution": area_std,
            "num_damage_regions": len(contours),
            "damage_patterns": damage_patterns,
            "preventive_measures": preventive_measures
        }
    
    def _analyze_damage_patterns(self, contours, areas, image):
        """Analyze damage patterns from contours and areas."""
        patterns = []
        
        # Calculate shape features for each contour
        for i, cnt in enumerate(contours):
            area = areas[i]
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            # Calculate shape features
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            
            # Analyze pattern based on features with adjusted thresholds
            if circularity > 0.7:  # Lowered threshold for circular damage
                patterns.append("circular_damage")
            elif aspect_ratio > 3:  # Lowered threshold for linear cracks
                patterns.append("linear_crack")
            elif area > 500:  # Lowered threshold for large damage areas
                patterns.append("large_damage_area")
            elif area < 200:  # Adjusted threshold for small damage spots
                patterns.append("small_damage_spot")
            else:
                patterns.append("irregular_damage")
            
            # Additional pattern analysis
            if len(cnt) > 4:  # Check if contour has enough points for ellipse fitting
                ellipse = cv2.fitEllipse(cnt)
                if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                    ellipse_ratio = max(ellipse[1]) / (min(ellipse[1]) + 1e-6)
                    if ellipse_ratio > 2:
                        patterns.append("elongated_damage")
            
            # Check for complex patterns
            if len(cnt) > 6:  # Complex contours
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / (hull_area + 1e-6)
                if solidity < 0.7:
                    patterns.append("complex_damage")
        
        return list(set(patterns))  # Remove duplicates
    
    def _get_preventive_measures(self, damage_patterns, severity, damage_percentage):
        """Get preventive measures based on damage patterns and severity."""
        measures = []
        
        # Severity-based measures
        if severity == "Severe":
            measures.extend([
                "Immediate structural assessment required",
                "Install temporary supports if needed",
                "Implement emergency protection measures",
                "Schedule comprehensive restoration work",
                "Document all damage patterns in detail",
                "Monitor structural stability continuously"
            ])
        elif severity == "Significant":
            measures.extend([
                "Regular structural monitoring",
                "Document damage progression",
                "Plan restoration work",
                "Implement protective measures",
                "Schedule detailed condition assessment",
                "Monitor environmental factors"
            ])
        elif severity == "Moderate":
            measures.extend([
                "Monthly condition assessment",
                "Regular cleaning and maintenance",
                "Monitor environmental factors",
                "Document any changes",
                "Implement preventive treatments",
                "Schedule regular inspections"
            ])
        else:  # Minimal
            measures.extend([
                "Regular visual inspection",
                "Basic maintenance",
                "Document condition",
                "Monitor for changes",
                "Implement preventive measures",
                "Schedule periodic assessments"
            ])
        
        # Pattern-specific measures
        if "circular_damage" in damage_patterns:
            measures.extend([
                "Check for water seepage",
                "Inspect drainage systems",
                "Monitor moisture levels",
                "Apply water-repellent treatment if needed",
                "Check for salt crystallization",
                "Monitor humidity levels"
            ])
        
        if "linear_crack" in damage_patterns:
            measures.extend([
                "Monitor crack width",
                "Check for structural movement",
                "Document crack patterns",
                "Consider crack filling if stable",
                "Install crack monitoring devices",
                "Check for underlying causes"
            ])
        
        if "large_damage_area" in damage_patterns:
            measures.extend([
                "Comprehensive surface assessment",
                "Check for underlying causes",
                "Plan major restoration",
                "Implement temporary protection",
                "Document damage extent",
                "Monitor progression"
            ])
        
        if "small_damage_spot" in damage_patterns:
            measures.extend([
                "Regular cleaning",
                "Monitor spot progression",
                "Document changes",
                "Consider spot treatment",
                "Check for surface deterioration",
                "Implement local protection"
            ])
        
        if "irregular_damage" in damage_patterns:
            measures.extend([
                "Detailed condition assessment",
                "Document damage patterns",
                "Monitor progression",
                "Plan targeted treatment",
                "Check for multiple damage types",
                "Implement comprehensive protection"
            ])
        
        if "elongated_damage" in damage_patterns:
            measures.extend([
                "Check for structural stress",
                "Monitor elongation progression",
                "Document stress patterns",
                "Consider structural reinforcement",
                "Check for foundation issues",
                "Implement stress monitoring"
            ])
        
        if "complex_damage" in damage_patterns:
            measures.extend([
                "Detailed pattern analysis",
                "Document complex damage",
                "Monitor multiple aspects",
                "Plan comprehensive treatment",
                "Check for multiple causes",
                "Implement multi-faceted protection"
            ])
        
        # Percentage-based additional measures
        if damage_percentage > 5:
            measures.extend([
                "Comprehensive condition survey",
                "Detailed documentation",
                "Regular monitoring schedule",
                "Emergency response plan",
                "Implement extensive protection",
                "Schedule major restoration"
            ])
        
        return list(set(measures))  # Remove duplicates
    
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
        
        # Create visualization
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
            "damage_severity": results["damage_severity"],
            "damage_distribution": results["damage_distribution"],
            "num_damage_regions": results["num_damage_regions"],
            "damage_patterns": results["damage_patterns"],
            "preventive_measures": results["preventive_measures"]
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