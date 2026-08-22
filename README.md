# DermaVision: ISIC 2019 ML Pipeline & Dashboard

![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

DermaVision is an intelligent skin image classification system. This repository contains the complete machine learning pipeline and a premium Streamlit-based web application designed to predict skin lesions using a **MobileNetV2** model fine-tuned on the **ISIC 2019** dataset.

---

## 📖 Project Overview

DermaVision aims to provide a fast, secure, and visually stunning screening tool for skin lesions. The model categorizes dermatoscopic images into one of three critical classes, leveraging a modular and strictly decoupled Python pipeline for preprocessing, validation, prediction, and evaluation.

## 📊 Dataset & Classes

The model is trained on a curated subset of the **ISIC 2019 dataset**.

- **Total Images**: 12,445
- **Training Set**: 9,956 images
- **Validation Set**: 2,489 images
- **Image Resolution**: 224 × 224 × 3 (RGB)

### Supported Classes
1. **BCC (Basal Cell Carcinoma)**
2. **Melanoma**
3. **Nevus**

> **Note on Large Files**: The ISIC datasets (`*.csv`, image folders) and model weights (`*.keras`) are excluded from this repository via `.gitignore` to keep the repo lightweight.

## 🧠 Model Architecture & Results

- **Architecture**: MobileNetV2 (pretrained on ImageNet) with custom top classification layers.
- **Official Validation Accuracy**: `90.96%`
- **Official Weighted F1-Score**: `90.93%`

*(Evaluated on the official Kaggle validation split of 2,489 images).*

## ⚙️ ML Pipeline Architecture

The system is highly modularized for robust production deployment:

- `validator.py`: Performs physical integrity validation of uploaded images (corruptions, formats).
- `preprocessing.py`: Implements pixel normalization (`/ 255.0`), reshaping (`224x224`), and RGB casting to exactly match the training pipeline.
- `prediction.py`: Encapsulates TensorFlow model loading and inference generation.
- `evaluation.py`: Independent script to calculate batched metrics on labeled datasets (Precision, Recall, F1, AUC, Confusion Matrix).
- `utils.py`: Shared utilities for mapping JSON classes and handling dynamic UI rendering.

## 📂 Project Structure

```text
DermaVision/
│
├── models/                     # Ignored in git (place best_mobilenetv2.keras here)
│   └── class_names.json        # Class index mapping
├── metrics/                    # Generated performance metrics and ROC curves
├── preprocessing.py            # Data standardization logic
├── validator.py                # Input sanitization
├── prediction.py               # ML Inference logic
├── evaluation.py               # Metrics and testing script
├── utils.py                    # Helper functions
├── streamlit_app.py            # Main frontend application
├── requirements.txt            # Python dependencies
└── README.md
```

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dermavision.git
   cd dermavision
   ```

2. **Set up the environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Model Weights**
   - Download your trained `best_mobilenetv2.keras` model and place it inside the `models/` directory.

5. **Launch the Dashboard**
   ```bash
   streamlit run streamlit_app.py
   ```

## ⚕️ Medical Disclaimer

> **DISCLAIMER**: This diagnostic screening report is compiled by an artificial intelligence model for research and educational purposes only. DermaVision is **NOT** clinically validated for medical use and does not replace a clinical examination or biopsy by a certified dermatologist. Any suspected lesion must be evaluated by a healthcare professional immediately.
