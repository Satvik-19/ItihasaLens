# Dataset Structure and Annotation Guidelines

This document outlines the organization of the ItihāsaLens dataset and provides guidelines for data annotation.

## Directory Structure

```
data/
├── images/                    # Monument images
│   ├── taj_mahal/            # Images of Taj Mahal
│   ├── khajuraho/            # Images of Khajuraho Temples
│   └── hampi/                # Images of Hampi Ruins
├── annotations/              # COCO format annotations
│   ├── damage/              # Damage segmentation masks
│   └── monuments/           # Monument bounding boxes
└── metadata/                # Additional information
    ├── monuments.json       # Monument metadata
    └── treatments.json      # Treatment knowledge base
```

## Image Collection Guidelines

1. **Image Quality**
   - Resolution: Minimum 1920x1080 pixels
   - Format: JPG or PNG
   - Lighting: Well-lit, avoid harsh shadows
   - Focus: Sharp and clear

2. **Image Types**
   - Multiple angles of each monument
   - Different times of day
   - Various weather conditions
   - Close-up shots of details
   - Wide shots of full structure

3. **Quantity**
   - 200-500 images per monument
   - 50-100 images with damage annotations
   - Multiple images of each damage type

## Annotation Guidelines

### Monument Recognition

1. **Bounding Boxes**
   - Format: COCO JSON
   - Include full monument in frame
   - Multiple angles and views
   - Label with monument name and style

2. **Classification**
   - Monument name
   - Architectural style
   - Historical period
   - Material type

### Damage Detection

1. **Segmentation Masks**
   - Format: COCO JSON with polygon annotations
   - Label damage types:
     - Cracks
     - Erosion
     - Discoloration
     - Structural damage
     - Biological growth

2. **Damage Severity**
   - Low: < 5% of surface area
   - Medium: 5-15% of surface area
   - High: > 15% of surface area

3. **Damage Location**
   - Structural elements
   - Decorative elements
   - Base/foundation
   - Roof/dome

## Metadata Format

### monuments.json
```json
{
    "taj_mahal": {
        "name": "Taj Mahal",
        "style": "Mughal Architecture",
        "period": "1632-1653",
        "material": "marble",
        "location": "Agra, Uttar Pradesh",
        "description": "Mausoleum complex in Agra...",
        "damage_types": ["discoloration", "erosion"]
    }
}
```

### treatments.json
```json
{
    "marble": {
        "discoloration": {
            "low": [
                {
                    "name": "Surface Cleaning",
                    "description": "Gentle cleaning with appropriate solutions",
                    "priority": 1,
                    "estimated_cost": "Low",
                    "time_frame": "1-2 days",
                    "requirements": ["Cleaning solutions", "Soft brushes"]
                }
            ]
        }
    }
}
```

## Annotation Tools

1. **Recommended Tools**
   - LabelMe for segmentation
   - CVAT for bounding boxes
   - VGG Image Annotator for classification

2. **Quality Control**
   - Double-check all annotations
   - Verify damage severity levels
   - Ensure consistent labeling
   - Cross-reference with experts

## Data Privacy and Ethics

1. **Image Rights**
   - Ensure proper permissions
   - Credit photographers
   - Respect cultural heritage

2. **Usage Guidelines**
   - Educational purposes
   - Heritage preservation
   - Research and development

## Contributing

1. **Adding New Data**
   - Follow directory structure
   - Use consistent naming
   - Include metadata
   - Document sources

2. **Updating Annotations**
   - Maintain format
   - Add new damage types
   - Update severity criteria
   - Document changes 