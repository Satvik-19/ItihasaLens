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
from preservation.suggestions import (
    PreservationAdvisor, DamageType, Severity, Material,
    EnvironmentalContext
)

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

def get_monument_location(monument_name: str) -> str:
    """Get location of monument from its name."""
    # Common monument locations with variations
    monument_locations = {
        # Agra monuments
        "taj mahal": "Agra, India",
        "taj": "Agra, India",
        "fatehpur sikri": "Agra, India",
        "agra fort": "Agra, India",
        "itmad-ud-daulah": "Agra, India",
        "itmad ud daulah": "Agra, India",
        
        # Delhi monuments
        "red fort": "Delhi, India",
        "lal qila": "Delhi, India",
        "humayun's tomb": "Delhi, India",
        "humayun tomb": "Delhi, India",
        "qutub minar": "Delhi, India",
        "qutb minar": "Delhi, India",
        "jama masjid": "Delhi, India",
        "jama mosque": "Delhi, India",
        "purana qila": "Delhi, India",
        "purana fort": "Delhi, India",
        
        # Jaipur monuments
        "hawa mahal": "Jaipur, India",
        "palace of winds": "Jaipur, India",
        "amber fort": "Jaipur, India",
        "amer fort": "Jaipur, India",
        "city palace": "Jaipur, India",
        "jal mahal": "Jaipur, India",
        
        # Other major monuments
        "jaisalmer fort": "Jaisalmer, India",
        "sonar qila": "Jaisalmer, India",
        "konark sun temple": "Puri, India",
        "khajuraho": "Khajuraho, India",
        "khajuraho temples": "Khajuraho, India",
        "ajanta caves": "Aurangabad, India",
        "ellora caves": "Aurangabad, India",
        "hampi": "Hampi, India",
        "mahabalipuram": "Mahabalipuram, India",
        "brihadeeswarar": "Thanjavur, India",
        "brihadeeswarar temple": "Thanjavur, India",
        "meenakshi temple": "Madurai, India",
        "meenakshi": "Madurai, India",
        "golden temple": "Amritsar, India",
        "harmandir sahib": "Amritsar, India",
        "somnath temple": "Somnath, India",
        "somnath": "Somnath, India",
        "sanchi stupa": "Sanchi, India",
        "sanchi": "Sanchi, India",
        "elephanta caves": "Mumbai, India",
        "gateway of india": "Mumbai, India",
        "victoria memorial": "Kolkata, India",
        "howrah bridge": "Kolkata, India",
        "charminar": "Hyderabad, India",
        "golconda fort": "Hyderabad, India"
    }
    
    # Clean and normalize monument name
    monument_name = monument_name.lower().strip()
    
    # Try exact match first
    if monument_name in monument_locations:
        return monument_locations[monument_name]
    
    # Try partial matches
    for key, location in monument_locations.items():
        # Check if monument name contains key or key contains monument name
        if key in monument_name or monument_name in key:
            return location
        
        # Check for common variations
        if any(variation in monument_name for variation in [
            key.replace(" ", ""),
            key.replace("-", " "),
            key.replace("'", ""),
            key.replace("s", ""),
            key.replace("temple", "").strip(),
            key.replace("fort", "").strip(),
            key.replace("palace", "").strip()
        ]):
            return location
    
    # If no match found, try to extract city from monument name
    common_cities = {
        "agra": "Agra, India",
        "delhi": "Delhi, India",
        "jaipur": "Jaipur, India",
        "mumbai": "Mumbai, India",
        "kolkata": "Kolkata, India",
        "hyderabad": "Hyderabad, India",
        "chennai": "Chennai, India",
        "bangalore": "Bangalore, India",
        "pune": "Pune, India",
        "ahmedabad": "Ahmedabad, India"
    }
    
    for city in common_cities:
        if city in monument_name:
            return common_cities[city]
    
    return "Location not found"

def display_environmental_context(env_context: EnvironmentalContext):
    """Display environmental context using Streamlit-native elements."""
    st.markdown("### 🌍 Environmental Context")
    st.markdown("---")
    
    # Safely get values with defaults
    location = getattr(env_context, 'location', 'Location not found')
    historical_climate = getattr(env_context, 'historical_climate', 'No climate data available')
    known_risks = getattr(env_context, 'known_risks', []) or []
    preservation_challenges = getattr(env_context, 'preservation_challenges', []) or []
    
    # Location
    st.markdown(f"📍 **Location:** {location}")
    
    # Historical Climate
    st.markdown(f"🌡️ **Historical Climate:** {historical_climate}")
    
    # Known Risks
    if known_risks:
        st.markdown("⚠️ **Known Risks:**")
        for risk in known_risks:
            st.markdown(f"• {risk}")
    
    # Preservation Challenges
    if preservation_challenges:
        st.markdown("🏛️ **Preservation Challenges:**")
        for challenge in preservation_challenges:
            st.markdown(f"• {challenge}")

def display_material_analysis(material_analysis):
    """Display material analysis using Streamlit-native elements."""
    st.markdown("### 🧱 Material Analysis")
    st.markdown("---")
    
    # Safely get values with defaults
    material_type = getattr(material_analysis.material, 'value', 'Unknown')
    historical_usage = getattr(material_analysis, 'historical_usage', 'No historical usage data available')
    characteristics = getattr(material_analysis, 'characteristics', []) or ['No characteristics available']
    traditional_techniques = getattr(material_analysis, 'traditional_techniques', []) or []
    similar_monuments = getattr(material_analysis, 'similar_monuments', []) or []
    
    # Material Type
    st.markdown(f"**Material Type:** {material_type}")
    
    # Historical Usage
    st.markdown(f"🏛️ **Historical Usage:** {historical_usage}")
    
    # Characteristics
    st.markdown("📋 **Characteristics:**")
    for char in characteristics:
        st.markdown(f"• {char}")
    
    # Traditional Techniques
    if traditional_techniques:
        st.markdown("🔧 **Traditional Techniques:**")
        for tech in traditional_techniques:
            st.markdown(f"• {tech}")
    
    # Similar Monuments
    if similar_monuments:
        st.markdown("🏛️ **Similar Monuments:**")
        for monument in similar_monuments:
            st.markdown(f"• {monument}")

def get_priority_class(priority):
    """Get CSS class for priority badge."""
    if isinstance(priority, str):
        priority = priority.lower()
    elif isinstance(priority, int):
        if priority >= 8:
            priority = 'critical'
        elif priority >= 5:
            priority = 'moderate'
        else:
            priority = 'minor'
    else:
        priority = 'moderate'
    return priority

def display_treatment_plan(treatment_plan):
    """Display treatment plan using Streamlit-native elements."""
    st.markdown("### 🛠️ Treatment Plan")
    st.markdown("---")
    
    if not treatment_plan:
        st.markdown("No treatment recommendations available.")
        return
    
    # Handle treatments
    treatments = treatment_plan.get("treatments", [])
    if treatments:
        for treatment in treatments:
            # Display treatment details
            st.markdown(f"#### {treatment.name}")
            st.markdown(f"**Description:** {treatment.description}")
            st.markdown(f"**Priority Level:** {treatment.priority}")
            st.markdown(f"**Estimated Cost:** {treatment.estimated_cost}")
            st.markdown(f"**Time Frame:** {treatment.time_frame}")
            
            # Requirements
            if hasattr(treatment, 'requirements') and treatment.requirements:
                st.markdown("**Requirements:**")
                for req in treatment.requirements:
                    st.markdown(f"- {req}")
            
            # Steps
            if hasattr(treatment, 'steps') and treatment.steps:
                st.markdown("**Treatment Steps:**")
                for step in treatment.steps:
                    st.markdown(f"- {step}")
            
            # Preventive Measures
            if hasattr(treatment, 'preventive_measures') and treatment.preventive_measures:
                st.markdown("**Preventive Measures:**")
                for measure in treatment.preventive_measures:
                    st.markdown(f"- {measure}")
            
            # Historical Context
            if hasattr(treatment, 'historical_context') and treatment.historical_context:
                st.markdown("**Historical Context:**")
                st.markdown(f"{treatment.historical_context}")
            
            # Similar Cases
            if hasattr(treatment, 'similar_cases') and treatment.similar_cases:
                st.markdown("**Similar Cases:**")
                for case in treatment.similar_cases:
                    st.markdown(f"- {case}")
            
            st.markdown("---")
    
    # Display general preventive measures
    preventive_measures = treatment_plan.get("preventive_measures", [])
    if preventive_measures:
        st.markdown("### 🛡️ General Preventive Measures")
        st.markdown("---")
        for measure in preventive_measures:
            st.markdown(f"- {measure}")
    
    # Display monitoring schedule
    monitoring_schedule = treatment_plan.get("monitoring_schedule", {})
    if monitoring_schedule:
        st.markdown("### 📅 Monitoring Schedule")
        st.markdown("---")
        for frequency, tasks in monitoring_schedule.items():
            st.markdown(f"**{frequency.capitalize()}:**")
            for task in tasks:
                st.markdown(f"- {task}")

def main():
    # Set page config
    st.set_page_config(
        page_title="ItihāsaLens",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional dark theme
    st.markdown("""
    <style>
        /* Base Styles */
        .stMarkdown {
            color: #E0E0E0;
            margin-bottom: 0.25rem !important;
            line-height: 1.4 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #E0E0E0 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0.75rem !important;
            line-height: 1.3 !important;
        }
        
        /* Lists */
        ul, ol {
            margin: 0.25rem 0 !important;
            padding-left: 1.25rem !important;
        }
        
        li {
            margin-bottom: 0.15rem !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #1E1E1E !important;
            border-radius: 0.25rem !important;
            padding: 0.5rem !important;
            margin-bottom: 0.25rem !important;
        }
        
        .streamlit-expanderContent {
            padding: 0.5rem 0 !important;
        }
        
        /* Priority Badges */
        .priority-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
            font-weight: 500;
            margin-left: 0.5rem;
        }
        
        .priority-critical {
            background-color: #DC2626;
            color: white;
        }
        
        .priority-moderate {
            background-color: #F59E0B;
            color: black;
        }
        
        .priority-minor {
            background-color: #10B981;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("ItihāsaLens")
    st.markdown("### Heritage Monument Analysis System")
    
    # Introduction
    st.markdown("""
    Upload an image of an Indian heritage monument to perform comprehensive analysis:
    - Monument identification and architectural classification
    - Structural damage assessment and analysis
    - Preservation recommendations and treatment plans
    - Historical context and comparative analysis
    - Monitoring and maintenance scheduling
    """)
    
    # Load models
    recognizer, detector, advisor = load_models()
    
    # File uploader
    st.markdown("### Image Upload")
    uploaded_file = st.file_uploader(
        "Select monument image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an Indian heritage monument"
    )

    image = None
    image_path = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        image_path = temp_path

    if image is not None:
        # Process image
        processed_image = process_image(image)
        
        # Analysis Results
        st.markdown("### Analysis Results")
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Monument Recognition")
            predictions = recognizer.predict(processed_image)
            
            for monument_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.markdown(f"**{monument_name}:** {prob:.2%}")
            
            top_monument = next(iter(sorted(predictions.items(), key=lambda x: x[1], reverse=True)))
            location = get_monument_location(top_monument[0])
            
            with st.expander("Location Information", expanded=False):
                st.markdown(f"**Detected Location:** {location}")
                custom_location = st.text_input(
                    "Specify location (city, country):",
                    value=location if location != "Location not found" else "",
                    help="Enter city and country (e.g., 'Delhi, India')"
                )
                if custom_location:
                    location = custom_location
                elif location == "Location not found":
                    st.warning("Location detection failed. Please specify manually.")
        
        with col2:
            st.markdown("#### Damage Analysis")
            damage_results = detector.analyze_damage(processed_image)
            
            st.image(cv2.cvtColor(damage_results["damage_overlay"], cv2.COLOR_BGR2RGB), 
                    caption="Damage Detection Overlay", use_column_width=True)
            
            # Determine severity class and progress
            damage_percentage = damage_results['damage_percentage']
            severity_class = "progress-minor"
            if damage_percentage > 50:
                severity_class = "progress-critical"
            elif damage_percentage > 20:
                severity_class = "progress-moderate"
            
            st.markdown(f"**Damage Percentage:** {damage_percentage:.1f}%")
            st.markdown(f"**Severity Level:** {damage_results['damage_severity']}")
        
        with col3:
            st.markdown("#### Material Analysis")
            if image_path:
                material_analysis = advisor.detect_material(image_path)
                if material_analysis:
                    display_material_analysis(material_analysis)
                else:
                    st.warning("""
                    Material analysis could not be performed due to:
                    - API configuration issues
                    - Network connectivity problems
                    - Image quality limitations
                    
                    Please verify API configuration and try again.
                    """)
        
        st.markdown("---")
        
        # Enhanced Analysis
        st.markdown("### Preservation Analysis")
        st.markdown("---")
        if image_path:
            treatment_plan = advisor.get_enhanced_treatment_plan(
                [(DamageType.CRACK, map_damage_severity(damage_results['damage_severity']))],
                material_analysis.material if material_analysis else Material.SANDSTONE,
                location
            )
            
            if treatment_plan:
                # Display environmental context if it exists
                if isinstance(treatment_plan, dict) and "environmental_context" in treatment_plan:
                    display_environmental_context(treatment_plan["environmental_context"])
                    st.markdown("---")
                
                # Display treatment plan
                display_treatment_plan(treatment_plan)
            else:
                st.warning("""
                Enhanced preservation analysis could not be performed due to:
                - API configuration issues
                - Network connectivity problems
                - Insufficient monument information
                
                Please verify API configuration and try again.
                """)
        
        # Clean up
        if uploaded_file and os.path.exists("temp_upload.jpg"):
            os.remove("temp_upload.jpg")

if __name__ == "__main__":
    main() 