# 🩺 Medical Diagnosis of Chest X-ray using Deep Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)

![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)

![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)

![License](https://img.shields.io/badge/License-MIT-yellow)

</p>

<p align="center">
  <img src="assets/banner.png" alt="Project Banner" width="100%">
</p>

## 📌 Overview

This project is an AI-powered medical diagnosis system that analyzes chest X-ray images using Deep Learning. It helps identify multiple thoracic diseases from X-ray images with high accuracy using the DenseNet121 architecture.

The application provides an interactive Streamlit interface where users can upload a chest X-ray image, receive disease predictions, confidence scores, Grad-CAM visualizations, and download a medical report.

---

## ✨ Features

- 🔍 Chest X-ray Disease Detection
- 🧠 DenseNet121 Transfer Learning
- 📊 Confidence Score for Predictions
- 🔥 Grad-CAM Visualization
- 📄 PDF Report Generation
- 🌐 Interactive Streamlit Interface
- 📈 Model Performance Dashboard

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| DenseNet121 | Feature Extraction |
| Streamlit | Web Application |
| OpenCV | Image Processing |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Matplotlib | Visualization |
| ReportLab | PDF Report Generation |

---

## 📂 Project Structure

```text
Medical-Diagnosis-of-Chest-X-ray-using-Deep-Learning/
│
├── app.py
├── requirements.txt
├── models/
├── screenshots/
├── docs/
├── assets/
└── README.md
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Tmane8329/Medical-Diagnosis-of-Chest-X-ray-using-Deep-Learning.git
```

Move into the project

```bash
cd Medical-Diagnosis-of-Chest-X-ray-using-Deep-Learning
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m streamlit run app.py
```

---

## 📊 Dataset

- NIH Chest X-ray Dataset
- Multi-label Classification
- 14 Disease Categories
- Image Size: 224 × 224 pixels

---

## 🧠 Model

- Architecture: DenseNet121
- Framework: TensorFlow/Keras
- Transfer Learning
- Binary Cross Entropy Loss
- Adam Optimizer

---

## 📸 Screenshots

(Add screenshots inside the **screenshots** folder)

| Home | Prediction |
|------|------------|
| ![](screenshots/home.png) | ![](screenshots/prediction.png) |

| Result | Grad-CAM |
|---------|----------|
| ![](screenshots/result.png) | ![](screenshots/gradcam.png) |

---

## 🎯 Future Improvements

- Mobile Application
- Explainable AI Enhancements
- Cloud Deployment
- Electronic Health Record Integration
- Multi-modal Medical Diagnosis

---

## 👨‍💻 Author

**Tejas Mane**

Bachelor of Engineering (Computer Engineering)

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.