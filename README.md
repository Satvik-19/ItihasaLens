# 🏛️ ItihāsaLens: Heritage Monument Analysis & Preservation System

**ItihāsaLens** is an AI-powered heritage conservation system designed to assist in the digital preservation of Indian monuments. It integrates computer vision, deep learning, and domain knowledge in archaeology to recognize monuments, detect damage, analyze materials, and recommend conservation actions through a user-friendly web interface.

---

## 🌟 Key Features

### 🏷️ Monument Recognition

* Identifies Indian heritage monuments using a custom-trained convolutional neural network (CNN).
* Supports diverse architectural styles with real-time prediction and confidence scores.

### 🧱 Damage Detection

* Detects and categorizes structural and surface damage using a YOLO-based object detection model.
* Damage types include:

  * Cracks
  * Erosion
  * Discoloration
  * Structural damage
  * Biological growth
* Provides damage metrics, heatmaps, and severity assessment.

### 🪨 Material Analysis

* Recognizes monument construction materials using a ResNet-based CNN model.
* Offers details on material properties, historical usage, and conservation techniques.

### 🤖 AI-Powered Analysis

* Uses language models (OpenAI GPT and BART CNN) for:

  * Historical significance insights
  * Architectural style interpretation
  * Preservation strategies
  * Environmental impact estimation
  * Step-by-step treatment planning

### 🛡️ Preservation Advisor

* Recommends practical and historical preservation methods.
* Includes:

  * Priority-based actions
  * Time and cost estimates
  * Similar case references
  * Monitoring schedules

---

## 🛠️ Tech Stack

### Core Technologies

* Python 3.8+
* Streamlit (Web Interface)
* TensorFlow / Keras
* OpenCV & scikit-image
* NumPy & Pillow

### AI/ML Components

* Custom CNN (Monument Recognition)
* YOLO-based Object Detection (Damage Detection)
* ResNet-based CNN (Material Analysis)
* OpenAI GPT & BART-Large CNN (AI Summaries)

### APIs & Services

* OpenAI API
* OpenRouter API (fallback for open-access LLMs)

---
🧠 Architecture Overview

![image](https://github.com/user-attachments/assets/74a09606-2e0b-4987-9bfc-2f551dff4e83)

Visual representation of the system workflow and technologies integrated into ItihāsaLens.

---

## 📁 Project Structure

```
ItihasaLens/
├── src/
│   ├── recognition/        # Monument classification logic
│   ├── damage/             # Damage detection models and utilities
│   ├── material/           # Material analysis components
│   ├── preservation/       # AI summary generation & preservation logic
│   └── visualization/      # Streamlit-based frontend
├── models/                 # Pretrained and fine-tuned model files
├── data/                   # Processed data and testing samples
├── tests/                  # Unit and integration tests
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### ✅ Prerequisites

* Python 3.8 or higher
* `pip` (Python package manager)
* Git
* (Recommended) Virtual environment tool: `venv` or `conda`

### 📦 Installation

```bash
git clone https://github.com/yourusername/itihasalens.git
cd itihasalens
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 🔐 Configure Environment Variables

```bash
cp .env.example .env
# Edit .env to add your API keys (OpenAI, OpenRouter, etc.)
```

---

## 💻 Running the Application

To launch the app:

```bash
streamlit run src/visualization/app.py
```

Once running:

1. Upload an image of a monument.
2. The system will automatically analyze and display:

   * Monument identification
   * Damage overlays and metrics
   * Material classification
   * AI-generated preservation insights

---

## 📊 Model Performance Overview

| Component            | Metric                  | Value |
| -------------------- | ----------------------- | ----- |
| Monument Recognition | Accuracy                | \~95% |
| Damage Detection     | mAP (YOLO)              | \~85% |
| Material Analysis    | Classification Accuracy | \~90% |

---

## 🔧 Model Details

### Monument Recognition

* **Architecture:** Custom CNN
* **Training:** Fine-tuned with Indian heritage datasets

### Damage Detection

* **Architecture:** YOLO-based object detector
* **Data:** Manually annotated heritage damage images

### Material Analysis

* **Architecture:** ResNet-based CNN
* **Data:** Labeled construction material samples from monuments

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m "Add YourFeature"`
4. Push and open a PR: `git push origin feature/YourFeature`

---

## 📧 Contact

For questions, issues, or access to datasets and models, reach out to the maintainer:

**Satvik Mishra**
📧 Email: `satvikmishrayt19@gmail.com`

---

## 📝 Important Notes

If cloning or setup fails, contact the maintainer for:

* Model files (e.g., `monument_recognition_model.h5`, `damage_detection_model.h5`)
* Complete project folder
* Dataset access
* Technical support

We aim to ensure a smooth setup and a high-impact use of ItihāsaLens for heritage conservation.
