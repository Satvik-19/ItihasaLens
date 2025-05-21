"""
Preservation suggestion module for recommending conservation treatments.
"""

from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
from enum import Enum
import sys
import subprocess
import base64
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Robust import handling for optional dependencies
def try_import(module_name, package_name=None):
    try:
        return __import__(module_name)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name or module_name])
            return __import__(module_name)
        except Exception as e:
            print(f"Failed to install {module_name}: {e}")
            return None

requests = try_import('requests')
HAS_REQUESTS = requests is not None
cv2 = try_import('cv2')
HAS_CV2 = cv2 is not None
np = try_import('numpy')
HAS_NUMPY = np is not None
try:
    from datetime import datetime
    import pytz
    HAS_PYTZ = True
except ImportError:
    pytz = try_import('pytz')
    HAS_PYTZ = pytz is not None
    from datetime import datetime
geopy_mod = try_import('geopy')
if geopy_mod:
    try:
        from geopy.geocoders import Nominatim
        HAS_GEOPY = True
    except ImportError:
        HAS_GEOPY = False
else:
    HAS_GEOPY = False

from typing import Optional

class DamageType(Enum):
    CRACK = "crack"
    EROSION = "erosion"
    DISCOLORATION = "discoloration"
    STRUCTURAL = "structural"
    BIOLOGICAL = "biological"

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Material(Enum):
    SANDSTONE = "sandstone"
    MARBLE = "marble"
    GRANITE = "granite"
    LIMESTONE = "limestone"
    BRICK = "brick"

@dataclass
class Treatment:
    name: str
    description: str
    priority: int
    estimated_cost: str
    time_frame: str
    requirements: List[str]
    steps: List[str] = None
    preventive_measures: List[str] = None
    historical_context: str = None
    similar_cases: List[str] = None

    def __post_init__(self):
        # Set default values if None
        if self.steps is None:
            self.steps = []
        if self.preventive_measures is None:
            self.preventive_measures = []
        if self.historical_context is None:
            self.historical_context = ""
        if self.similar_cases is None:
            self.similar_cases = []

@dataclass
class MaterialAnalysis:
    material: Material
    characteristics: List[str]
    preservation_notes: str
    historical_usage: str
    traditional_techniques: List[str]
    similar_monuments: List[str]

@dataclass
class EnvironmentalContext:
    location: str
    historical_climate: str
    known_risks: List[str]
    preservation_challenges: List[str]

# Add this at the top of the file, after the imports
MONUMENT_MATERIAL_MAPPING = {
    # Marble Monuments
    "marble": [
        "taj mahal", "taj", "victoria memorial", "dilwara", "itmad",
        "itmad-ud-daulah", "itmad ud daulah", "buland darwaza", "buland darwaza",
        "akbar's tomb", "akbar tomb", "sikandra", "sikandra tomb",
        "motilal nehru memorial", "motilal nehru", "rashtrapati bhavan",
        "rashtrapati", "president house", "president's house"
    ],
    
    # Sandstone Monuments
    "sandstone": [
        "red fort", "lal qila", "humayun", "humayun's tomb", "humayun tomb",
        "fatehpur sikri", "hawa mahal", "palace of winds", "jaisalmer fort",
        "sonar qila", "india gate", "qutub minar", "qutb minar", "purana qila",
        "purana fort", "amber fort", "amer fort", "jantar mantar", "jantar",
        "hazrat nizamuddin", "nizamuddin", "safdarjung tomb", "safdarjung",
        "tughlaqabad fort", "tughlaqabad", "firoz shah kotla", "firoz shah",
        "salimgarh fort", "salimgarh", "old fort", "old fort delhi",
        "jama masjid", "jama mosque", "delhi gate", "ajmeri gate",
        "kashmiri gate", "turkman gate", "lahori gate", "delhi darwaza"
    ],
    
    # Granite Monuments
    "granite": [
        "brihadeeswarar", "brihadeeswarar temple", "meenakshi temple",
        "meenakshi", "somnath temple", "somnath", "virupaksha temple",
        "virupaksha", "hoysaleswara temple", "hoysaleswara", "belur",
        "belur temple", "halebidu", "halebidu temple", "badami caves",
        "badami", "aihole", "aihole temples", "pattadakal",
        "pattadakal temples", "mahabalipuram", "shore temple",
        "shore temple mahabalipuram", "kailasa temple", "kailasa",
        "kailasanatha", "kailasanatha temple"
    ],
    
    # Limestone Monuments
    "limestone": [
        "sanchi stupa", "sanchi", "ajanta caves", "ajanta",
        "ellora caves", "ellora", "elephanta caves", "elephanta",
        "karla caves", "karla", "bhaja caves", "bhaja",
        "kanheri caves", "kanheri", "nashik caves", "nashik",
        "pandavleni", "pandavleni caves", "jogeshwari caves", "jogeshwari"
    ],
    
    # Brick Monuments
    "brick": [
        "nalanda", "nalanda university", "nalanda ruins", "vikramshila",
        "vikramshila university", "vikramshila ruins", "odantapuri",
        "odantapuri university", "odantapuri ruins", "somapura",
        "somapura mahavihara", "somapura ruins", "jagaddala",
        "jagaddala mahavihara", "jagaddala ruins", "telhara",
        "telhara university", "telhara ruins"
    ]
}

class PreservationAdvisor:
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the preservation advisor.
        
        Args:
            knowledge_base_path: Path to JSON knowledge base file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.hf_api_key = os.getenv("HF_API_KEY", "")
        if HAS_GEOPY:
            self.geolocator = Nominatim(user_agent="itihasalens")
        else:
            self.geolocator = None

    def _load_knowledge_base(self, path: str = None) -> Dict:
        """
        Load treatment knowledge base from JSON file.
        
        Args:
            path: Path to knowledge base file
            
        Returns:
            Dictionary containing treatment rules
        """
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default knowledge base with complete treatment information
        return {
            "sandstone": {
                "crack": {
                    "low": [
                        {
                            "name": "Visual Monitoring and Documentation",
                            "description": "Regular visual inspection and documentation of crack patterns",
                            "priority": 1,
                            "estimated_cost": "₹5,000 - ₹10,000 per inspection",
                            "time_frame": "Monthly inspections for 3 months",
                            "requirements": ["Digital camera", "Measurement tools", "Documentation forms"],
                            "steps": [
                                "Photograph crack patterns",
                                "Measure crack widths",
                                "Document environmental conditions",
                                "Create detailed report"
                            ],
                            "preventive_measures": [
                                "Regular cleaning",
                                "Monitor water drainage",
                                "Check for vegetation growth"
                            ],
                            "historical_context": "Traditional sandstone monuments often develop hairline cracks that require careful monitoring",
                            "similar_cases": ["Red Fort, Delhi", "Jaisalmer Fort"]
                        }
                    ],
                    "medium": [
                        {
                            "name": "Crack Filling and Consolidation",
                            "description": "Fill cracks with compatible mortar and apply consolidating agent",
                            "priority": 2,
                            "estimated_cost": "₹50,000 - ₹1,00,000 per square meter",
                            "time_frame": "2-3 weeks",
                            "requirements": [
                                "Lime-based mortar",
                                "Consolidating agent",
                                "Expert conservation team",
                                "Scaffolding"
                            ],
                            "steps": [
                                "Clean crack surfaces",
                                "Prepare compatible mortar",
                                "Fill cracks carefully",
                                "Apply consolidating agent",
                                "Monitor for 1 month"
                            ],
                            "preventive_measures": [
                                "Regular inspection",
                                "Waterproofing treatment",
                                "Drainage maintenance"
                            ],
                            "historical_context": "Traditional lime-based mortars have been used for centuries in Indian monuments",
                            "similar_cases": ["Qutub Minar", "Hawa Mahal"]
                        }
                    ],
                    "high": [
                        {
                            "name": "Structural Reinforcement and Major Repair",
                            "description": "Install structural supports and repair major cracks with advanced techniques",
                            "priority": 3,
                            "estimated_cost": "₹2,00,000 - ₹5,00,000 per square meter",
                            "time_frame": "2-3 months",
                            "requirements": [
                                "Structural engineer",
                                "Conservation architect",
                                "Specialized equipment",
                                "High-strength materials"
                            ],
                            "steps": [
                                "Structural assessment",
                                "Install temporary supports",
                                "Remove damaged sections",
                                "Apply reinforcement",
                                "Restore surface",
                                "Monitor for 6 months"
                            ],
                            "preventive_measures": [
                                "Regular structural assessment",
                                "Environmental monitoring",
                                "Emergency response plan"
                            ],
                            "historical_context": "Major structural repairs require careful planning and traditional techniques",
                            "similar_cases": ["Konark Sun Temple", "Khajuraho Temples"]
                        }
                    ]
                },
                "erosion": {
                    "low": [
                        {
                            "name": "Surface Cleaning and Protection",
                            "description": "Gentle cleaning of eroded surfaces and application of protective coating",
                            "priority": 1,
                            "estimated_cost": "₹10,000 - ₹20,000 per square meter",
                            "time_frame": "1 week",
                            "requirements": [
                                "Soft brushes",
                                "Distilled water",
                                "Protective coating",
                                "Safety equipment"
                            ],
                            "steps": [
                                "Remove loose particles",
                                "Clean with distilled water",
                                "Apply protective coating",
                                "Document condition"
                            ],
                            "preventive_measures": [
                                "Regular cleaning",
                                "Monitor weather impact",
                                "Maintain drainage"
                            ],
                            "historical_context": "Traditional cleaning methods preserve the original surface",
                            "similar_cases": ["Fatehpur Sikri", "Amber Fort"]
                        }
                    ],
                    "medium": [
                        {
                            "name": "Surface Consolidation and Repair",
                            "description": "Consolidate eroded surfaces and repair damaged areas",
                            "priority": 2,
                            "estimated_cost": "₹75,000 - ₹1,50,000 per square meter",
                            "time_frame": "3-4 weeks",
                            "requirements": [
                                "Consolidating agents",
                                "Compatible repair materials",
                                "Expert team",
                                "Specialized tools"
                            ],
                            "steps": [
                                "Assess erosion extent",
                                "Apply consolidating agent",
                                "Fill eroded areas",
                                "Match surface texture",
                                "Apply protective coating"
                            ],
                            "preventive_measures": [
                                "Regular inspection",
                                "Environmental monitoring",
                                "Protective measures"
                            ],
                            "historical_context": "Surface consolidation techniques have evolved from traditional methods",
                            "similar_cases": ["Ellora Caves", "Ajanta Caves"]
                        }
                    ],
                    "high": [
                        {
                            "name": "Major Surface Restoration",
                            "description": "Complete surface restoration and protection system",
                            "priority": 3,
                            "estimated_cost": "₹2,50,000 - ₹5,00,000 per square meter",
                            "time_frame": "2-3 months",
                            "requirements": [
                                "Restoration team",
                                "Specialized materials",
                                "Advanced equipment",
                                "Conservation architect"
                            ],
                            "steps": [
                                "Detailed assessment",
                                "Remove damaged sections",
                                "Prepare new surface",
                                "Apply restoration materials",
                                "Match original texture",
                                "Install protection system"
                            ],
                            "preventive_measures": [
                                "Regular maintenance",
                                "Environmental control",
                                "Monitoring system"
                            ],
                            "historical_context": "Major restoration requires understanding of original construction techniques",
                            "similar_cases": ["Mahabalipuram", "Hampi"]
                        }
                    ]
                }
            }
        }
    
    def _get_material_details(self, material: Material, monument_name: str = None) -> MaterialAnalysis:
        """
        Get detailed material analysis with monument-specific information.
        
        Args:
            material: Detected material type
            monument_name: Optional name of the monument
            
        Returns:
            MaterialAnalysis object with detailed information
        """
        # Force material based on monument name
        if monument_name:
            monument_name = monument_name.lower().strip()
            
            # Force marble for specific monuments
            if any(name in monument_name for name in [
                "taj mahal", "taj", "victoria memorial", "dilwara", "itmad",
                "itmad-ud-daulah", "itmad ud daulah", "buland darwaza"
            ]):
                material = Material.MARBLE
            
            # Force sandstone for specific monuments
            elif any(name in monument_name for name in [
                "red fort", "lal qila", "humayun", "humayun's tomb", "humayun tomb",
                "fatehpur sikri", "hawa mahal", "palace of winds", "jaisalmer fort",
                "sonar qila", "india gate", "qutub minar", "qutb minar", "purana qila",
                "purana fort", "amber fort", "amer fort"
            ]):
                material = Material.SANDSTONE

        # Monument-specific checks
        if monument_name:
            monument_name = monument_name.lower()
            
            # Marble monuments
            if material == Material.MARBLE:
                return MaterialAnalysis(
                    material=Material.MARBLE,
                    characteristics=[
                        "Makrana white marble",
                        "Semi-translucent quality",
                        "Changes color with sunlight",
                        "Intricate inlay work"
                    ],
                    preservation_notes="The monument's marble requires special care due to its unique properties and intricate inlay work.",
                    historical_usage="Built using Makrana marble, known for its purity and beauty. The marble was transported from Rajasthan.",
                    traditional_techniques=[
                        "Traditional mud pack cleaning",
                        "Inlay work preservation",
                        "Marble surface protection",
                        "Regular cleaning with distilled water"
                    ],
                    similar_monuments=[
                        "Itmad-ud-Daulah's Tomb, Agra",
                        "Victoria Memorial, Kolkata",
                        "Dilwara Temples, Mount Abu"
                    ]
                )
            
            # Sandstone monuments
            elif material == Material.SANDSTONE:
                return MaterialAnalysis(
                    material=Material.SANDSTONE,
                    characteristics=[
                        "Red sandstone from Rajasthan",
                        "Durable and weather-resistant",
                        "Rich in iron oxide",
                        "Intricate carvings"
                    ],
                    preservation_notes="The sandstone requires careful water management and regular cleaning to prevent salt crystallization and biological growth.",
                    historical_usage="Constructed using red sandstone from Rajasthan, known for its durability and rich color.",
                    traditional_techniques=[
                        "Lime-based mortar repairs",
                        "Water-repellent treatments",
                        "Surface consolidation",
                        "Traditional cleaning methods"
                    ],
                    similar_monuments=[
                        "Red Fort, Delhi",
                        "Fatehpur Sikri, Agra",
                        "Hawa Mahal, Jaipur",
                        "Jaisalmer Fort, Jaisalmer"
                    ]
                )
        
        # Default material details if no monument-specific match
        material_details = {
            Material.MARBLE: MaterialAnalysis(
                material=Material.MARBLE,
                characteristics=[
                    "White or light-colored marble",
                    "Smooth surface texture",
                    "High reflectivity",
                    "Fine grain structure"
                ],
                preservation_notes="Marble requires regular cleaning and protection from acid rain and pollution.",
                historical_usage="Widely used in Mughal architecture for its beauty and durability.",
                traditional_techniques=[
                    "Traditional mud pack cleaning",
                    "Surface protection",
                    "Regular cleaning with distilled water",
                    "Joint preservation"
                ],
                similar_monuments=[
                    "Taj Mahal, Agra",
                    "Victoria Memorial, Kolkata",
                    "Dilwara Temples, Mount Abu"
                ]
            ),
            Material.SANDSTONE: MaterialAnalysis(
                material=Material.SANDSTONE,
                characteristics=[
                    "Red or yellow sandstone",
                    "Visible grain structure",
                    "Moderate porosity",
                    "Weather-resistant"
                ],
                preservation_notes="Sandstone requires careful water management and protection from salt crystallization.",
                historical_usage="Extensively used in Mughal and Rajput architecture for its durability and color.",
                traditional_techniques=[
                    "Lime-based mortar repairs",
                    "Water-repellent treatments",
                    "Surface consolidation",
                    "Traditional cleaning methods"
                ],
                similar_monuments=[
                    "Red Fort, Delhi",
                    "Fatehpur Sikri, Agra",
                    "Hawa Mahal, Jaipur",
                    "Jaisalmer Fort, Jaisalmer"
                ]
            )
        }
        
        return material_details.get(material, material_details[Material.SANDSTONE])

    def _get_enhanced_material_analysis(self, image) -> Material:
        """
        Enhanced material analysis using computer vision techniques.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Detected material type
        """
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate color statistics
            avg_color = np.mean(hsv, axis=(0,1))
            std_color = np.std(hsv, axis=(0,1))
            
            # Calculate texture metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            texture_magnitude = np.std(cv2.Laplacian(gray, cv2.CV_64F))
            
            # Calculate homogeneity
            glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
            homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
            
            # Calculate additional metrics
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            color_std = np.std(image, axis=(0,1))
            brightness = np.mean(gray)
            
            # Enhanced material detection logic with more precise thresholds
            
            # Marble detection (white/light colored, smooth, high homogeneity)
            if (avg_color[1] < 20 and  # Low saturation
                texture_magnitude < 10 and  # Smooth texture
                homogeneity > 0.92 and  # High homogeneity
                std_color[2] < 45 and  # Low value variation
                edge_density < 0.1 and  # Low edge density
                brightness > 150):  # High brightness
                return Material.MARBLE
            
            # Sandstone detection (reddish, visible grain, moderate texture)
            elif ((avg_color[0] > 5 and avg_color[0] < 35) and  # Reddish hue
                  avg_color[1] > 25 and  # Moderate saturation
                  texture_magnitude > 12 and  # Visible texture
                  homogeneity < 0.88 and  # Less homogeneous
                  edge_density > 0.1 and  # Moderate edge density
                  color_std[2] > 30):  # High value variation
                return Material.SANDSTONE
            
            # Granite detection (high texture, low homogeneity, dark)
            elif (texture_magnitude > 18 and  # High texture
                  homogeneity < 0.78 and  # Low homogeneity
                  std_color[2] > 45 and  # High value variation
                  edge_density > 0.15):  # High edge density
                return Material.GRANITE
            
            # Brick detection (red hue, high saturation, very high texture)
            elif (avg_color[0] > 0 and avg_color[0] < 15 and  # Red hue
                  avg_color[1] > 35 and  # High saturation
                  texture_magnitude > 22 and  # Very high texture
                  edge_density > 0.2):  # Very high edge density
                return Material.BRICK
            
            # If no clear match, default to sandstone for Indian monuments
            else:
                return Material.SANDSTONE
            
        except Exception as e:
            print(f"Error in material analysis: {str(e)}")
            # Default to sandstone for Indian monuments
            return Material.SANDSTONE

    def _process_ai_material_analysis(self, analysis: Dict, monument_name: str = None) -> Optional[MaterialAnalysis]:
        """Process AI analysis results into MaterialAnalysis object."""
        try:
            # Handle both list and dict responses
            if isinstance(analysis, list):
                analysis = analysis[0] if analysis else {}
            
            # Extract material from AI analysis
            material_text = analysis.get("material", "").lower()
            
            # Map AI-detected material to Material enum
            material_map = {
                "sandstone": Material.SANDSTONE,
                "marble": Material.MARBLE,
                "granite": Material.GRANITE,
                "limestone": Material.LIMESTONE,
                "brick": Material.BRICK,
                "makrana marble": Material.MARBLE,
                "red sandstone": Material.SANDSTONE,
                "agra marble": Material.MARBLE,
                "rajasthani sandstone": Material.SANDSTONE
            }
            
            # Get material type from AI analysis
            material = None
            for key, value in material_map.items():
                if key in material_text:
                    material = value
                    break
            
            if material is None:
                return None
            
            # Extract detailed information from AI analysis
            characteristics = analysis.get("characteristics", [])
            if isinstance(characteristics, str):
                characteristics = [c.strip() for c in characteristics.split(",")]
            
            historical_usage = analysis.get("historical_usage", "")
            traditional_techniques = analysis.get("traditional_techniques", [])
            if isinstance(traditional_techniques, str):
                traditional_techniques = [t.strip() for t in traditional_techniques.split(",")]
            
            similar_monuments = analysis.get("similar_monuments", [])
            if isinstance(similar_monuments, str):
                similar_monuments = [m.strip() for m in similar_monuments.split(",")]
            
            preservation_notes = analysis.get("preservation_notes", "")
            
            # If AI didn't provide enough information, return None
            if not all([characteristics, historical_usage, traditional_techniques, similar_monuments, preservation_notes]):
                return None
            
            return MaterialAnalysis(
                material=material,
                characteristics=characteristics,
                preservation_notes=preservation_notes,
                historical_usage=historical_usage,
                traditional_techniques=traditional_techniques,
                similar_monuments=similar_monuments
            )
            
        except Exception as e:
            print(f"Error processing AI analysis: {str(e)}")
            return None

    def _get_monument_material(self, monument_name: str) -> Optional[Material]:
        """
        Get material type based on monument name using the mapping dictionary.
        
        Args:
            monument_name: Name of the monument
            
        Returns:
            Material type if found, None otherwise
        """
        if not monument_name:
            return None
        
        monument_name = monument_name.lower().strip()
        
        # Check each material category
        for material, monuments in MONUMENT_MATERIAL_MAPPING.items():
            # Check for exact matches
            if monument_name in monuments:
                return Material(material)
            
            # Check for partial matches
            for known_monument in monuments:
                if (known_monument in monument_name or 
                    monument_name in known_monument or
                    any(variation in monument_name for variation in [
                        known_monument.replace(" ", ""),
                        known_monument.replace("-", " "),
                        known_monument.replace("'", ""),
                        known_monument.replace("s", ""),
                        known_monument.replace("temple", "").strip(),
                        known_monument.replace("fort", "").strip(),
                        known_monument.replace("palace", "").strip(),
                        known_monument.replace("tomb", "").strip(),
                        known_monument.replace("gate", "").strip(),
                        known_monument.replace("darwaza", "").strip()
                    ])):
                    return Material(material)
        
        return None

    def detect_material(self, image_path: str, monument_name: str = None) -> MaterialAnalysis:
        """
        Detect material using monument-specific knowledge and image analysis.
        
        Args:
            image_path: Path to the image file
            monument_name: Name of the monument (optional)
            
        Returns:
            MaterialAnalysis object with detected material and details
        """
        # Force material type based on monument name
        if monument_name:
            monument_name = monument_name.lower().strip()
            
            # Force marble for specific monuments
            if "taj" in monument_name or "taj mahal" in monument_name:
                return MaterialAnalysis(
                    material=Material.MARBLE,
                    characteristics=[
                        "Makrana white marble",
                        "Semi-translucent quality",
                        "Changes color with sunlight",
                        "Intricate inlay work"
                    ],
                    preservation_notes="The monument's marble requires special care due to its unique properties and intricate inlay work.",
                    historical_usage="Built using Makrana marble, known for its purity and beauty. The marble was transported from Rajasthan.",
                    traditional_techniques=[
                        "Traditional mud pack cleaning",
                        "Inlay work preservation",
                        "Marble surface protection",
                        "Regular cleaning with distilled water"
                    ],
                    similar_monuments=[
                        "Itmad-ud-Daulah's Tomb, Agra",
                        "Victoria Memorial, Kolkata",
                        "Dilwara Temples, Mount Abu"
                    ]
                )
            
            # Force sandstone for specific monuments
            elif any(name in monument_name for name in [
                "red fort", "lal qila", "humayun", "humayun's tomb", "humayun tomb",
                "fatehpur sikri", "hawa mahal", "palace of winds", "jaisalmer fort",
                "sonar qila", "india gate", "qutub minar", "qutb minar", "purana qila",
                "purana fort", "amber fort", "amer fort"
            ]):
                return MaterialAnalysis(
                    material=Material.SANDSTONE,
                    characteristics=[
                        "Red sandstone from Rajasthan",
                        "Durable and weather-resistant",
                        "Rich in iron oxide",
                        "Intricate carvings"
                    ],
                    preservation_notes="The sandstone requires careful water management and regular cleaning to prevent salt crystallization and biological growth.",
                    historical_usage="Constructed using red sandstone from Rajasthan, known for its durability and rich color.",
                    traditional_techniques=[
                        "Lime-based mortar repairs",
                        "Water-repellent treatments",
                        "Surface consolidation",
                        "Traditional cleaning methods"
                    ],
                    similar_monuments=[
                        "Red Fort, Delhi",
                        "Fatehpur Sikri, Agra",
                        "Hawa Mahal, Jaipur",
                        "Jaisalmer Fort, Jaisalmer"
                    ]
                )

        # For unknown monuments, use basic image analysis
        if not (HAS_CV2 and HAS_NUMPY):
            return self._get_material_details(Material.SANDSTONE, monument_name)

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Use enhanced image analysis
            material = self._get_enhanced_material_analysis(image)
            return self._get_material_details(material, monument_name)
                
        except Exception as e:
            print(f"Error in material detection: {str(e)}")
            # Default to sandstone for Indian monuments
            return self._get_material_details(Material.SANDSTONE, monument_name)

    def get_enhanced_treatment_plan(
        self,
        damages: List[Tuple[DamageType, Severity]],
        material: Material,
        location: str,
        monument_name: str = None
    ) -> Dict:
        """
        Get enhanced treatment plan with AI-powered analysis.
        
        Args:
            damages: List of (damage_type, severity) tuples
            material: Material type
            location: Location string
            monument_name: Name of the monument (optional)
            
        Returns:
            Dictionary containing comprehensive treatment plan
        """
        # Get AI analysis
        if self.hf_api_key and HAS_REQUESTS:
            try:
                API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
                headers = {"Authorization": f"Bearer {self.hf_api_key}"}
                
                # Prepare monument-specific prompt
                monument_context = f" for {monument_name}" if monument_name else ""
                damage_context = ", ".join([f"{d[0].value} ({d[1].value})" for d in damages])
                
                prompt = f"""
                Heritage Monument Analysis{monument_context}:
                Material: {material.value}
                Damage Types: {damage_context}
                Location: {location}
                
                Please provide monument-specific information about:
                1. Historical context and similar cases, focusing on this specific monument
                2. Traditional preservation techniques used for this monument
                3. Environmental challenges specific to this monument's location
                4. Preventive measures tailored to this monument's needs
                5. Step-by-step treatment recommendations for this specific monument
                6. Similar monuments with successful preservation cases
                
                Focus on providing monument-specific details rather than generic information.
                If this is a well-known monument, provide specific details about its unique preservation requirements.
                Consider the severity of damage: {damage_context}
                
                Include:
                - Traditional techniques used in this specific monument
                - Local environmental factors affecting this monument
                - Historical preservation methods used for this monument
                - Cost considerations specific to this monument
                - Timeline requirements for this monument's preservation
                - Resource availability in this monument's location
                - Emergency measures if damage is severe
                """
                
                response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
                if response.status_code == 200:
                    analysis = response.json()
                    
                    # Process AI analysis into treatment plan
                    treatments = []
                    for damage_type, severity in damages:
                        treatment_list = self.get_treatments(damage_type, severity, material)
                        if treatment_list:
                            treatment = treatment_list[0]
                            # Enhance treatment with AI analysis
                            if isinstance(analysis, list):
                                analysis = analysis[0]
                            if analysis.get("historical_context"):
                                treatment.historical_context = analysis["historical_context"]
                            if analysis.get("similar_cases"):
                                treatment.similar_cases = analysis["similar_cases"]
                            if analysis.get("treatment_steps"):
                                treatment.steps = analysis["treatment_steps"]
                            treatments.append(treatment)
                    
                    return {
                        "treatments": treatments,
                        "environmental_context": EnvironmentalContext(
                            location=location,
                            historical_climate=analysis.get("historical_climate", "Based on historical records"),
                            known_risks=analysis.get("environmental_challenges", ["Environmental factors", "Weather conditions"]),
                            preservation_challenges=analysis.get("preservation_challenges", ["Regular maintenance required", "Environmental protection needed"])
                        ),
                        "preventive_measures": analysis.get("preventive_measures", ["Regular inspection", "Documentation", "Maintenance"]),
                        "monitoring_schedule": self._get_monitoring_schedule(
                            material, damages, analysis
                        )
                    }
                
            except Exception as e:
                print(f"Error in AI analysis: {str(e)}")
        
        # Fallback to basic treatment plan only if AI analysis fails
        return self._get_basic_treatment_plan(damages, material, location, monument_name)

    def _get_basic_treatment_plan(
        self,
        damages: List[Tuple[DamageType, Severity]],
        material: Material,
        location: str,
        monument_name: str = None
    ) -> Dict:
        """Get basic treatment plan when AI analysis is not available."""
        # Try to get monument-specific details first
        if monument_name:
            monument_specific = self._get_monument_specific_details(monument_name, material)
            if monument_specific:
                return {
                    "treatments": [Treatment(
                        name="Monument-Specific Treatment",
                        description=monument_specific["preservation_notes"],
                        priority=1,
                        estimated_cost="Based on monument requirements",
                        time_frame="As per monument needs",
                        requirements=["Expert conservation team", "Specialized equipment"],
                        steps=monument_specific["traditional_techniques"],
                        preventive_measures=["Regular inspection", "Documentation", "Maintenance"],
                        historical_context=monument_specific["historical_usage"],
                        similar_cases=monument_specific["similar_monuments"]
                    )],
                    "environmental_context": EnvironmentalContext(
                        location=location,
                        historical_climate="Based on historical records",
                        known_risks=["Environmental factors", "Weather conditions"],
                        preservation_challenges=["Regular maintenance required", "Environmental protection needed"]
                    ),
                    "preventive_measures": monument_specific["traditional_techniques"],
                    "monitoring_schedule": {
                        "daily": ["Visual inspection"],
                        "weekly": ["Detailed documentation"],
                        "monthly": ["Comprehensive assessment"],
                        "quarterly": ["Maintenance check"],
                        "yearly": ["Full evaluation"]
                    }
                }
        
        # Fallback to generic treatment plan
        treatments = []
        for damage_type, severity in damages:
            treatment_list = self.get_treatments(damage_type, severity, material)
            if treatment_list:
                treatments.append(treatment_list[0])
        
        return {
            "treatments": treatments,
            "environmental_context": EnvironmentalContext(
                location=location,
                historical_climate="Based on historical records",
                known_risks=["Environmental factors", "Weather conditions"],
                preservation_challenges=["Regular maintenance required", "Environmental protection needed"]
            ),
            "preventive_measures": [
                "Regular inspection",
                "Documentation",
                "Maintenance",
                "Environmental monitoring"
            ],
            "monitoring_schedule": {
                "daily": ["Visual inspection"],
                "weekly": ["Detailed documentation"],
                "monthly": ["Comprehensive assessment"],
                "quarterly": ["Maintenance check"],
                "yearly": ["Full evaluation"]
            }
        }
    
    def get_treatments(
        self,
        damage_type: DamageType,
        severity: Severity,
        material: Material
    ) -> List[Treatment]:
        """
        Get treatment suggestions based on damage assessment.
        
        Args:
            damage_type: Type of damage detected
            severity: Severity level of damage
            material: Monument material
            
        Returns:
            List of recommended treatments
        """
        try:
            treatments = self.knowledge_base[material.value][damage_type.value][severity.value]
            return [Treatment(**t) for t in treatments]
        except KeyError:
            # Return severity-specific default treatment
            if severity == Severity.HIGH:
                return [Treatment(
                    name="Emergency Structural Intervention Required",
                    description="Severe damage detected. Immediate expert assessment and intervention needed.",
                    priority=3,
                    estimated_cost="₹5,00,000 - ₹10,00,000 for assessment and initial stabilization",
                    time_frame="Immediate action required, 2-3 months for complete treatment",
                    requirements=[
                        "Structural engineer",
                        "Conservation architect",
                        "Emergency response team",
                        "Specialized equipment",
                        "Temporary stabilization materials"
                    ],
                    steps=[
                        "Emergency structural assessment",
                        "Install temporary supports",
                        "Stabilize affected areas",
                        "Develop comprehensive treatment plan",
                        "Begin immediate conservation work"
                    ],
                    preventive_measures=[
                        "24/7 monitoring",
                        "Emergency response plan",
                        "Regular structural assessment",
                        "Environmental protection"
                    ],
                    historical_context="Severe damage requires immediate attention to prevent further deterioration",
                    similar_cases=["Various heritage sites requiring emergency intervention"]
                )]
            elif severity == Severity.MEDIUM:
                return [Treatment(
                    name="Comprehensive Conservation Required",
                    description="Moderate damage detected. Planned conservation work needed.",
                    priority=2,
                    estimated_cost="₹2,00,000 - ₹5,00,000 for assessment and treatment",
                    time_frame="1-2 months for complete treatment",
                    requirements=[
                        "Conservation expert",
                        "Specialized team",
                        "Conservation materials",
                        "Documentation equipment"
                    ],
                    steps=[
                        "Detailed condition assessment",
                        "Document damage patterns",
                        "Prepare conservation plan",
                        "Begin conservation work",
                        "Monitor progress"
                    ],
                    preventive_measures=[
                        "Weekly monitoring",
                        "Regular maintenance",
                        "Environmental control",
                        "Documentation"
                    ],
                    historical_context="Moderate damage requires careful planning and execution",
                    similar_cases=["Various heritage sites with moderate damage"]
                )]
            else:
                return [Treatment(
                    name="Regular Maintenance and Monitoring",
                    description="Minor damage detected. Regular maintenance and monitoring required.",
                    priority=1,
                    estimated_cost="₹25,000 - ₹50,000 for assessment and maintenance",
                    time_frame="2-4 weeks for maintenance work",
                    requirements=[
                        "Conservation expert",
                        "Maintenance team",
                        "Basic equipment",
                        "Documentation tools"
                    ],
                    steps=[
                        "Condition assessment",
                        "Document current state",
                        "Perform maintenance work",
                        "Monitor results"
                    ],
                    preventive_measures=[
                        "Monthly inspection",
                        "Regular cleaning",
                        "Basic maintenance",
                        "Documentation"
                    ],
                    historical_context="Regular maintenance is crucial for preventing further damage",
                    similar_cases=["Various heritage sites under regular maintenance"]
                )]
    
    def get_priority_treatment(
        self,
        damage_type: DamageType,
        severity: Severity,
        material: Material
    ) -> Treatment:
        """
        Get the highest priority treatment for the given damage.
        
        Args:
            damage_type: Type of damage detected
            severity: Severity level of damage
            material: Monument material
            
        Returns:
            Highest priority treatment
        """
        treatments = self.get_treatments(damage_type, severity, material)
        if not treatments:
            return None
        return max(treatments, key=lambda x: x.priority)
    
    def get_treatment_plan(
        self,
        damages: List[Tuple[DamageType, Severity]],
        material: Material
    ) -> List[Treatment]:
        """
        Get a comprehensive treatment plan for multiple damages.
        
        Args:
            damages: List of (damage_type, severity) tuples
            material: Monument material
            
        Returns:
            Prioritized list of treatments
        """
        all_treatments = []
        for damage_type, severity in damages:
            treatments = self.get_treatments(damage_type, severity, material)
            all_treatments.extend(treatments)
        
        # Sort by priority (highest first)
        return sorted(all_treatments, key=lambda x: x.priority, reverse=True)
    
    def _get_preventive_measures(
        self,
        material: Material,
        damages: List[Tuple[DamageType, Severity]],
        ai_analysis: Dict
    ) -> List[str]:
        """Get preventive measures based on material and AI analysis."""
        measures = []
        
        # Material-specific measures
        if material == Material.MARBLE:
            measures.extend([
                "Regular cleaning to prevent acid rain damage",
                "Apply protective coating every 2 years",
                "Monitor for discoloration"
            ])
        elif material == Material.SANDSTONE:
            measures.extend([
                "Regular inspection for salt crystallization",
                "Clean drainage systems",
                "Monitor for water seepage"
            ])
        
        # Add AI-suggested measures
        measures.extend(ai_analysis.get("preventive_measures", []))
        
        return measures
    
    def _get_monitoring_schedule(
        self,
        material: Material,
        damages: List[Tuple[DamageType, Severity]],
        ai_analysis: Dict
    ) -> Dict[str, List[str]]:
        """Get monitoring schedule based on material and damage severity."""
        schedule = {
            "daily": [],
            "weekly": [],
            "monthly": [],
            "quarterly": [],
            "yearly": []
        }
        
        # Add material-specific monitoring
        if material == Material.MARBLE:
            schedule["monthly"].append("Check for acid rain damage")
            schedule["quarterly"].append("Inspect protective coating")
        elif material == Material.SANDSTONE:
            schedule["weekly"].append("Check for salt crystallization")
            schedule["monthly"].append("Inspect drainage systems")
        
        # Add damage-specific monitoring
        for damage_type, severity in damages:
            if severity == Severity.HIGH:
                schedule["daily"].append(f"Monitor {damage_type.value} progression")
            elif severity == Severity.MEDIUM:
                schedule["weekly"].append(f"Check {damage_type.value} status")
            else:
                schedule["monthly"].append(f"Inspect {damage_type.value}")
        
        return schedule 