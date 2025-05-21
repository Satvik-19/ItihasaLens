"""
Test script to evaluate improvements in damage detection and material analysis.
"""

import os
import cv2
import numpy as np
from damage.model import DamageDetector
from preservation.suggestions import PreservationAdvisor
import matplotlib.pyplot as plt
from pathlib import Path

def test_damage_detection(image_path, output_dir="test_results"):
    """Test damage detection on a single image."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = DamageDetector()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get base filename
    base_name = Path(image_path).stem
    
    # Analyze damage
    results = detector.analyze_damage(image)
    
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
    plt.title(f'Damage Analysis\nSeverity: {results["damage_severity"]}\nDamage: {results["damage_percentage"]:.1f}%\nRegions: {results["num_damage_regions"]}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{base_name}_analysis.png")
    plt.close()
    
    # Save damage analysis details
    with open(f"{output_dir}/{base_name}_damage_analysis.txt", "w") as f:
        f.write(f"Damage Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Damage Percentage: {results['damage_percentage']:.1f}%\n")
        f.write(f"Damage Severity: {results['damage_severity']}\n")
        f.write(f"Number of Damage Regions: {results['num_damage_regions']}\n")
        f.write(f"Damage Distribution: {results['damage_distribution']:.1f}\n\n")
        
        f.write("Detected Damage Patterns:\n")
        for pattern in results['damage_patterns']:
            f.write(f"- {pattern}\n")
        
        f.write("\nRecommended Preventive Measures:\n")
        for measure in results['preventive_measures']:
            f.write(f"- {measure}\n")
    
    return results

def test_material_analysis(image_path, monument_name=None, output_dir="test_results"):
    """Test material analysis on a single image."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize advisor
    advisor = PreservationAdvisor()
    
    # Get base filename
    base_name = Path(image_path).stem
    
    # Analyze material
    analysis = advisor.detect_material(image_path, monument_name)
    
    # Save results
    with open(f"{output_dir}/{base_name}_material_analysis.txt", "w") as f:
        f.write(f"Material Analysis Results for {monument_name if monument_name else 'Unknown Monument'}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Detected Material: {analysis.material.value}\n\n")
        f.write("Characteristics:\n")
        for char in analysis.characteristics:
            f.write(f"- {char}\n")
        f.write("\nPreservation Notes:\n")
        f.write(analysis.preservation_notes + "\n\n")
        f.write("Historical Usage:\n")
        f.write(analysis.historical_usage + "\n\n")
        f.write("Traditional Techniques:\n")
        for tech in analysis.traditional_techniques:
            f.write(f"- {tech}\n")
        f.write("\nSimilar Monuments:\n")
        for monument in analysis.similar_monuments:
            f.write(f"- {monument}\n")
    
    return analysis

def main():
    """Main test function."""
    # Test images
    test_images = [
        "data/test_multi_label/sans-0_0_0_0- (1).jpg",
        "data/test_multi_label/sans-0_0_0_0- (2).jpg",
        "data/test_multi_label/sans-0_0_0_0- (3).jpg"
    ]
    
    # Test monuments
    test_monuments = [
        "Taj Mahal",
        "Humayun's Tomb",
        "Red Fort"
    ]
    
    # Run tests
    for i, (image_path, monument_name) in enumerate(zip(test_images, test_monuments)):
        print(f"\nTesting image {i+1}: {image_path}")
        print(f"Monument: {monument_name}")
        
        # Test damage detection
        damage_results = test_damage_detection(image_path)
        print("\nDamage Detection Results:")
        print(f"Damage Percentage: {damage_results['damage_percentage']:.1f}%")
        print(f"Damage Severity: {damage_results['damage_severity']}")
        print(f"Number of Damage Regions: {damage_results['num_damage_regions']}")
        print(f"Damage Distribution: {damage_results['damage_distribution']:.1f}")
        print("\nDetected Damage Patterns:")
        for pattern in damage_results['damage_patterns']:
            print(f"- {pattern}")
        print("\nRecommended Preventive Measures:")
        for measure in damage_results['preventive_measures']:
            print(f"- {measure}")
        
        # Test material analysis
        material_results = test_material_analysis(image_path, monument_name)
        print("\nMaterial Analysis Results:")
        print(f"Detected Material: {material_results.material.value}")
        print(f"Number of Characteristics: {len(material_results.characteristics)}")
        print(f"Number of Traditional Techniques: {len(material_results.traditional_techniques)}")
        print(f"Number of Similar Monuments: {len(material_results.similar_monuments)}")
        
        print("\nResults saved in test_results directory")

if __name__ == "__main__":
    main() 