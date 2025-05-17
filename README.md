# ItihāsaLens: AI-Powered Heritage Monument Analysis Tool

ItihāsaLens is an advanced AI system for analyzing and preserving Indian heritage monuments. The tool provides monument recognition, damage detection, severity assessment, and preservation suggestions using computer vision and deep learning techniques.

## Features

- **Monument & Style Recognition**: Identifies monuments and their architectural styles using CNN-based models
- **Damage Detection**: Segments and highlights damaged regions in monument images
- **Severity Assessment**: Evaluates damage severity based on area and location
- **Preservation Suggestions**: Provides conservation recommendations based on damage analysis
- **Digital Visualization**: Interactive web interface for image analysis and comparison

## Project Structure

```
itihasalens/
├── data/                  # Dataset and annotations
│   ├── images/           # Monument images
│   ├── annotations/      # COCO/YOLO format annotations
│   └── metadata/         # Monument metadata and knowledge base
├── models/               # Model definitions and weights
├── src/                  # Source code
│   ├── recognition/      # Monument recognition module
│   ├── damage/          # Damage detection module
│   ├── preservation/    # Preservation suggestion module
│   └── visualization/   # Web interface
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/itihasalens.git
cd itihasalens
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web interface:
```bash
streamlit run src/visualization/app.py
```

2. Upload monument images through the interface
3. View recognition results, damage analysis, and preservation suggestions

## Dataset Preparation

1. Organize monument images in `data/images/{monument_name}/`
2. Create annotations in COCO format in `data/annotations/`
3. Add monument metadata in `data/metadata/monuments.json`

## Model Training

1. Train monument recognition model:
```bash
python src/recognition/train.py
```

2. Train damage detection model:
```bash
python src/damage/train.py
```

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Archaeological Survey of India (ASI) guidelines
- UNESCO World Heritage Centre
- Open-source computer vision community 