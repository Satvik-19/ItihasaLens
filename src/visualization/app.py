"""
Streamlit web interface for ItihāsaLens.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.model import MonumentRecognizer, MONUMENT_CLASS_NAMES
from damage.model import DamageDetector
from preservation.suggestions import PreservationAdvisor, DamageType, Severity, Material

# Initialize models
@st.cache_resource
def load_models():
    """Load and cache the ML models."""
    try:
        print("Loading monument recognition model...")
        recognizer = MonumentRecognizer(
            class_names=MONUMENT_CLASS_NAMES,
            model_path="models/monument_recognition_model.h5"
        )
        print("Monument recognition model loaded successfully")
        
        print("\nLoading damage detection model...")
        detector = DamageDetector()
        print("Damage detection model initialized")
        
        print("\nLoading preservation advisor...")
        advisor = PreservationAdvisor()
        print("Preservation advisor loaded successfully")
        
        return recognizer, detector, advisor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise

def process_image(image):
    """Process uploaded image for model input."""
    # Convert to numpy array
    image = np.array(image)
    
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image

def map_damage_severity(detector_severity: str) -> Severity:
    """Map damage detector severity to preservation advisor severity."""
    severity_map = {
        "minimal": Severity.LOW,
        "moderate": Severity.MEDIUM,
        "significant": Severity.HIGH,
        "severe": Severity.HIGH
    }
    return severity_map.get(detector_severity.lower(), Severity.MEDIUM)

def main():
    st.set_page_config(
        page_title="ItihāsaLens",
        page_icon="🏛️",
        layout="wide"
    )
    
    st.title("ItihāsaLens: AI-Powered Heritage Monument Analysis")
    st.markdown("""
    Upload an image of an Indian heritage monument to:
    - Identify the monument and its architectural style
    - Detect and analyze damage
    - Get preservation recommendations
    """)
    
    # Load models
    recognizer, detector, advisor = load_models()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    image = None
    image_path = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Save uploaded file temporarily
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        image_path = temp_path

    if image is not None:
        # Process image
        processed_image = process_image(image)
        
        # Create three columns for results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Monument Recognition")
            # Get monument predictions
            predictions = recognizer.predict(processed_image)
            
            # Display top predictions
            st.write("Top Predictions:")
            for monument_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.write(f"- {monument_name}: {prob:.2%}")
        
        with col2:
            st.subheader("Damage Analysis")
            # Get damage analysis
            damage_results = detector.analyze_damage(processed_image)
            
            # Display damage overlay
            st.image(cv2.cvtColor(damage_results["damage_overlay"], cv2.COLOR_BGR2RGB), 
                    caption="Damage Detection", use_column_width=True)
            
            # Display damage statistics
            st.write(f"Damage Percentage: {damage_results['damage_percentage']:.1f}%")
            st.write(f"Damage Severity: {damage_results['damage_severity']}")
        
        with col3:
            st.subheader("Material Analysis")
            if image_path:
                # Get material analysis
                material_analysis = advisor.detect_material(image_path)
                if material_analysis:
                    st.write(f"**Detected Material:** {material_analysis.material.value}")
                    
                    st.write("**Characteristics:**")
                    for char in material_analysis.characteristics:
                        st.write(f"- {char}")
                    
                    st.write("**Preservation Notes:**")
                    st.write(material_analysis.preservation_notes)
                else:
                    st.write("Material analysis not available")
        
        # Detailed Analysis Section
        st.subheader("Detailed Preservation Analysis")
        if image_path:
            # Get detailed analysis
            analysis = advisor.get_detailed_analysis(
                image_path,
                DamageType.CRACK,  # This should be determined by the damage detector
                map_damage_severity(damage_results['damage_severity'])
            )
            
            # Display analysis in expandable sections
            with st.expander("Damage Assessment", expanded=True):
                if isinstance(analysis["damage_analysis"], str):
                    st.write(analysis["damage_analysis"])
                else:
                    st.write("Error in damage analysis. Using basic assessment.")
                    st.write(f"Damage Type: {DamageType.CRACK.value}")
                    st.write(f"Severity: {damage_results['damage_severity']}")
            
            with st.expander("Preservation Recommendations", expanded=True):
                for treatment in analysis["recommendations"]:
                    st.write(f"**{treatment.name}** (Priority: {treatment.priority})")
                    st.write(f"Description: {treatment.description}")
                    st.write(f"Estimated Cost: {treatment.estimated_cost}")
                    st.write(f"Time Frame: {treatment.time_frame}")
                    st.write("Requirements:")
                    for req in treatment.requirements:
                        st.write(f"- {req}")
                    st.write("---")
        
        # Clean up temporary file
        if uploaded_file and os.path.exists("temp_upload.jpg"):
            os.remove("temp_upload.jpg")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ for Indian Heritage Preservation</p>
        <p>Powered by AI and Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 