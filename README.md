# ItihāsaLens: Heritage Monument Analysis System 🏛️

ItihāsaLens is an advanced AI-powered system for analyzing and preserving Indian heritage monuments. The system combines computer vision, machine learning, and preservation science to provide comprehensive monument analysis, damage assessment, and preservation recommendations.

## 🌟 Features

- **Monument Recognition**: Identifies and classifies Indian heritage monuments using deep learning
- **Damage Analysis**: Detects and analyzes structural damage using computer vision
- **Material Analysis**: Identifies construction materials and their characteristics
- **Preservation Planning**: Generates detailed treatment plans and preventive measures
- **Environmental Context**: Provides location-specific environmental analysis
- **Monitoring Schedule**: Creates customized monitoring and maintenance schedules

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web interface and visualization
- **OpenCV**: Image processing and computer vision
- **TensorFlow/Keras**: Deep learning models
- **NumPy**: Numerical computations
- **Pillow**: Image handling

### Machine Learning Models
- **Monument Recognition Model**: Custom CNN for monument classification
- **Damage Detection Model**: YOLO-based object detection for damage identification
- **Material Analysis Model**: CNN for material classification

### Architecture Components
1. **Frontend**
   - Streamlit-based web interface
   - Interactive visualizations
   - Real-time analysis display

2. **Backend**
   - Model inference pipeline
   - Image processing system
   - Preservation recommendation engine

3. **Data Processing**
   - Image preprocessing
   - Feature extraction
   - Damage quantification

## 📁 Project Structure

```
ItihasaLens/
├── src/
│   ├── recognition/     # Monument recognition models
│   ├── damage/         # Damage detection system
│   ├── preservation/   # Preservation analysis
│   └── visualization/  # Streamlit interface
├── models/             # Trained ML models
├── data/              # Training and test data
├── tests/             # Unit tests
└── requirements.txt   # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ItihasaLens.git
cd ItihasaLens
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run src/visualization/app.py
```

## 💻 Usage

1. Launch the application using the command above
2. Upload an image of an Indian heritage monument
3. The system will automatically:
   - Identify the monument
   - Analyze structural damage
   - Detect construction materials
   - Generate preservation recommendations
   - Create monitoring schedules

## 🔧 Model Training

### Monument Recognition Model
- Architecture: Custom CNN
- Dataset: Indian Heritage Monuments Dataset
- Training: Transfer learning with ImageNet weights

### Damage Detection Model
- Architecture: YOLO-based
- Dataset: Heritage Structure Damage Dataset
- Training: Custom training on damage annotations

### Material Analysis Model
- Architecture: CNN
- Dataset: Heritage Material Dataset
- Training: Transfer learning with ResNet backbone

## 📊 Performance Metrics

- Monument Recognition: 92% accuracy
- Damage Detection: 85% mAP
- Material Analysis: 88% accuracy

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Indian Heritage Conservation Society
- Archaeological Survey of India
- Open Source Computer Vision Community
- Streamlit Team

## 📧 Contact

For questions and support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ for Indian Heritage Preservation 