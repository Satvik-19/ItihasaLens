"""
Preservation suggestion module for recommending conservation treatments.
"""

from typing import Dict, List, Tuple, Optional
import json
import os
from dataclasses import dataclass
from enum import Enum
import requests
import cv2
import numpy as np

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

@dataclass
class MaterialAnalysis:
    material: Material
    characteristics: List[str]
    preservation_notes: str

class PreservationAdvisor:
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the preservation advisor.
        
        Args:
            knowledge_base_path: Path to JSON knowledge base file
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.hf_api_key = os.getenv("HF_API_KEY", "")  # Get API key from environment variable
    
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
        
        # Default knowledge base
        return {
            "sandstone": {
                "crack": {
                    "low": [
                        {
                            "name": "Visual Monitoring",
                            "description": "Regular visual inspection and documentation",
                            "priority": 1,
                            "estimated_cost": "Low",
                            "time_frame": "Monthly",
                            "requirements": ["Camera", "Documentation forms"]
                        }
                    ],
                    "medium": [
                        {
                            "name": "Crack Filling",
                            "description": "Fill cracks with compatible mortar",
                            "priority": 2,
                            "estimated_cost": "Medium",
                            "time_frame": "1-2 weeks",
                            "requirements": ["Mortar", "Tools", "Expert consultation"]
                        }
                    ],
                    "high": [
                        {
                            "name": "Structural Reinforcement",
                            "description": "Install structural supports and repair major cracks",
                            "priority": 3,
                            "estimated_cost": "High",
                            "time_frame": "1-2 months",
                            "requirements": ["Structural engineer", "Heavy equipment", "Expert team"]
                        }
                    ]
                },
                "erosion": {
                    "low": [
                        {
                            "name": "Surface Cleaning",
                            "description": "Gentle cleaning of eroded surfaces",
                            "priority": 1,
                            "estimated_cost": "Low",
                            "time_frame": "Weekly",
                            "requirements": ["Soft brushes", "Water"]
                        }
                    ],
                    "medium": [
                        {
                            "name": "Surface Consolidation",
                            "description": "Apply consolidating agents to strengthen surface",
                            "priority": 2,
                            "estimated_cost": "Medium",
                            "time_frame": "2-3 weeks",
                            "requirements": ["Consolidating agents", "Expert application"]
                        }
                    ],
                    "high": [
                        {
                            "name": "Major Restoration",
                            "description": "Complete surface restoration and protection",
                            "priority": 3,
                            "estimated_cost": "High",
                            "time_frame": "2-3 months",
                            "requirements": ["Restoration team", "Specialized materials"]
                        }
                    ]
                }
            }
        }
    
    def detect_material(self, image_path: str) -> Optional[MaterialAnalysis]:
        """
        Detect material using image analysis and rule-based approach.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MaterialAnalysis object with detected material and details
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate average color and texture metrics
            avg_color = np.mean(hsv, axis=(0, 1))
            std_color = np.std(hsv, axis=(0, 1))
            
            # Calculate texture metrics using Sobel operator
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            texture_magnitude = np.mean(np.sqrt(sobelx**2 + sobely**2))
            
            # Material detection rules based on color and texture
            if avg_color[1] < 30:  # Low saturation
                if texture_magnitude < 20:
                    material = Material.MARBLE
                else:
                    material = Material.GRANITE
            elif avg_color[0] < 30:  # Reddish hue
                material = Material.BRICK
            elif std_color[1] > 40:  # High saturation variation
                material = Material.SANDSTONE
            else:
                material = Material.LIMESTONE
            
            # Get material characteristics based on type
            characteristics = {
                Material.SANDSTONE: ["Porous", "Weather-resistant", "Traditional material", "Requires regular maintenance"],
                Material.MARBLE: ["Dense", "Polished surface", "Vulnerable to acid rain", "Requires careful cleaning"],
                Material.GRANITE: ["Very hard", "Weather-resistant", "Low maintenance", "Durable"],
                Material.LIMESTONE: ["Soft", "Vulnerable to water", "Traditional material", "Requires protection"],
                Material.BRICK: ["Porous", "Traditional material", "Requires pointing", "Vulnerable to salt damage"]
            }
            
            # Get preservation notes based on material
            preservation_notes = {
                Material.SANDSTONE: "Requires regular cleaning and protection from water damage. Monitor for salt crystallization.",
                Material.MARBLE: "Protect from acid rain and pollution. Use gentle cleaning methods. Regular polishing may be needed.",
                Material.GRANITE: "Most durable material. Regular cleaning and inspection sufficient. Monitor for any cracks.",
                Material.LIMESTONE: "Highly vulnerable to water damage. Requires water repellent treatment. Regular inspection needed.",
                Material.BRICK: "Monitor mortar joints. Protect from water and salt damage. Regular repointing may be needed."
            }
            
            return MaterialAnalysis(
                material=material,
                characteristics=characteristics[material],
                preservation_notes=preservation_notes[material]
            )
            
        except Exception as e:
            print(f"Error in material detection: {str(e)}")
            # Return default analysis on error
            return MaterialAnalysis(
                material=Material.SANDSTONE,
                characteristics=["Porous", "Weather-resistant", "Traditional material", "Requires regular maintenance"],
                preservation_notes="Requires regular cleaning and protection from water damage. Monitor for salt crystallization."
            )
    
    def get_detailed_analysis(self, image_path: str, damage_type: DamageType, severity: Severity) -> Dict:
        """
        Get detailed preservation analysis using Hugging Face API.
        
        Args:
            image_path: Path to the image file
            damage_type: Type of damage detected
            severity: Severity of damage
            
        Returns:
            Dictionary containing detailed analysis and recommendations
        """
        # Get material analysis first
        material_analysis = self.detect_material(image_path)
        material_info = f"Material: {material_analysis.material.value}" if material_analysis else "Material: Unknown"
        
        if not self.hf_api_key:
            # Return basic analysis if no API key
            return {
                "material_analysis": material_analysis,
                "damage_analysis": f"""
                Current Condition Assessment:
                - Material: {material_info}
                - Damage Type: {damage_type.value}
                - Severity: {severity.value}
                
                Immediate Preservation Needs:
                - Regular inspection and monitoring
                - Basic cleaning and maintenance
                - Protection from environmental factors
                
                Recommended Treatments:
                {self._get_basic_treatment_plan(damage_type, severity)}
                
                Preventive Measures:
                - Regular cleaning schedule
                - Environmental monitoring
                - Documentation of condition
                
                Long-term Maintenance Plan:
                - Quarterly inspections
                - Annual detailed assessment
                - Regular maintenance as needed
                """,
                "recommendations": self.get_treatments(damage_type, severity, 
                    material_analysis.material if material_analysis else Material.SANDSTONE)
            }
            
        try:
            # Call Hugging Face API for detailed analysis
            API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            prompt = f"""
            Heritage Monument Analysis:
            {material_info}
            Damage Type: {damage_type.value}
            Severity: {severity.value}
            
            Please provide a detailed preservation analysis including:
            1. Current condition assessment
            2. Immediate preservation needs
            3. Recommended treatments
            4. Preventive measures
            5. Long-term maintenance plan
            
            Focus on practical, actionable recommendations suitable for heritage conservation.
            """
            
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
            analysis = response.json()
            
            # If API fails, use basic analysis
            if not analysis or "error" in analysis:
                analysis = self._get_basic_analysis(damage_type, severity)["damage_analysis"]
            
            return {
                "material_analysis": material_analysis,
                "damage_analysis": analysis,
                "recommendations": self.get_treatments(damage_type, severity, 
                    material_analysis.material if material_analysis else Material.SANDSTONE)
            }
            
        except Exception as e:
            print(f"Error in detailed analysis: {str(e)}")
            return self._get_basic_analysis(damage_type, severity)
    
    def _get_basic_treatment_plan(self, damage_type: DamageType, severity: Severity) -> str:
        """Generate a basic treatment plan based on damage type and severity."""
        treatments = {
            DamageType.CRACK: {
                Severity.LOW: "- Monitor crack development\n- Document changes\n- Basic sealing if needed",
                Severity.MEDIUM: "- Fill cracks with compatible mortar\n- Structural assessment\n- Regular monitoring",
                Severity.HIGH: "- Structural reinforcement\n- Professional assessment\n- Emergency stabilization if needed"
            },
            DamageType.EROSION: {
                Severity.LOW: "- Gentle cleaning\n- Surface protection\n- Regular monitoring",
                Severity.MEDIUM: "- Surface consolidation\n- Protective coating\n- Regular maintenance",
                Severity.HIGH: "- Major restoration\n- Structural assessment\n- Long-term protection plan"
            }
        }
        
        return treatments.get(damage_type, {}).get(severity, "Consult with conservation experts for specific treatment plan.")
    
    def _get_basic_analysis(self, damage_type: DamageType, severity: Severity) -> Dict:
        """Get basic analysis when API is not available."""
        return {
            "material_analysis": None,
            "damage_analysis": f"Damage type: {damage_type.value}, Severity: {severity.value}",
            "recommendations": self.get_treatments(damage_type, severity, Material.SANDSTONE)
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
            return []
    
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