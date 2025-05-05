# 🎤 Speech Emotion Recognition with CNN

![Banner](https://user-images.githubusercontent.com/your-image.png) <!-- Replace with your image link -->

[![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Medium](https://img.shields.io/badge/Read-Blog%20Post-000000?logo=medium&logoColor=white)](https://medium.com/@diego-rios/speech-emotion-recognition-with-convolutional-neural-network-ae5406a1c0f7)

---

## 🧠 Project Overview

**Speech Emotion Recognition (SER)** aims to classify the emotional state of a speaker using audio recordings. This project uses a **1D Convolutional Neural Network** (CNN) to classify audio into **8 different emotions** from the RAVDESS and TESS datasets.

> 🎯 Goal: Build an efficient deep learning model that can understand **how** something is said, not just **what** is said.

---

## 🎬 Demo

<p float="left">
  <img src="https://user-images.githubusercontent.com/demo-waveform.png" width="300" />
  <img src="https://user-images.githubusercontent.com/demo-mfcc.png" width="300" />
</p>

🔊 Example audio clips and MFCC visualizations provided in the `/demo` folder.

---

## 🛠 Tech Stack

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas
- Librosa (for audio processing)
- Scikit-learn
- Matplotlib & Seaborn

---

## ✨ Features

- 🎧 Real-time or pre-recorded audio classification
- 🎯 76% test accuracy with CNN
- 📊 Model evaluation with confusion matrix
- 📁 RAVDESS + TESS dataset integration
- 🔍 MFCC feature extraction and visualization

---

## 🧱 Model Architecture

Input (MFCCs)
↓
Conv1D (64 filters, kernel size=5)
↓
ReLU Activation
↓
MaxPooling1D (pool size=2)
↓
Dropout (0.3)
↓
Conv1D (128 filters, kernel size=5)
↓
ReLU Activation
↓
MaxPooling1D (pool size=2)
↓
Dropout (0.3)
↓
Flatten
↓
Dense (64 units) → ReLU
↓
Dense (8 units) → Softmax (Emotion Classes)



🧠 Trained on 8 emotions:
- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 📁 Dataset

### 🎵 RAVDESS: [Link](https://zenodo.org/record/1188976)
- 24 actors (12 male, 12 female)
- 8 emotions
- 7356 files
- Clean, studio-quality audio

### 🎵 TESS: [Link](https://tspace.library.utoronto.ca/handle/1807/24487)
- 2 female actors (older adults)
- 7 emotions
- 2800+ files
- Canadian English speakers

📌 All audio files were preprocessed by:
- Resampling to 22,050 Hz
- Normalization
- Extraction of MFCC features (13 coefficients)

---

## 🚀 Getting Started





### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/speech-emotion-cnn.git
cd speech-emotion-cnn

```


 ### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```


### 3️⃣ Run the Project

```bash
python train_model.py

```


```bash
python test_model.py

```


```bash
python predict_live.py

```


| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 76%    |
| Model Type    | 1D CNN |
| Epochs        | 50     |
| Optimizer     | Adam   |


📊 Confusion Matrix:
🎧 Audio Features:
Waveform:
MFCC Representation:

🧠 Insights & Observations

Emotions like happy, sad, and angry were more accurately predicted.
Confusions occurred between neutral and calm, and between fearful and disgust — possibly due to vocal similarities.
Increasing model depth improved accuracy but led to overfitting on a small dataset — regularization was key.
