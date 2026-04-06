# FireFound

### Forest Fire Detection using Support Vector Machine

**HOG · HSV Histogram · Texture Features · PCA · RBF-SVM**

---

## Overview

FireFound is a machine learning-based web application that detects whether a forest or aerial image contains an active wildfire.

Built entirely using **classical machine learning (no deep learning frameworks)**, this project extracts meaningful visual features from images and classifies them using an SVM model.

Users can upload an image through a web interface and instantly get:

* Prediction (Wildfire / No Fire)
* Confidence score

---

## Problem

Wildfires cause massive environmental and economic damage every year.

Monitoring forests manually using cameras, drones, or satellite imagery is not scalable. By the time a fire is detected, it may already be too late.

The challenge:

> Detect wildfire **early and automatically** from images.

---

## Solution

This project solves the problem using a complete ML pipeline:

* Image preprocessing
* Feature extraction (HOG + HSV + texture)
* Feature scaling and dimensionality reduction (PCA)
* SVM classification with probability output

---

## Features

* Image upload via web interface
* Real-time prediction
* Pre-trained SVM model
* Confidence score output
* End-to-end ML pipeline

---

## Methodology

### Image Pre-processing

* Resize images to **128×128**
* Convert to **Grayscale and HSV**
* Normalize using histogram techniques

---

### Feature Engineering

**1. HOG (Histogram of Oriented Gradients)**

* Captures edge structure
* ~3780 features per image

**2. HSV Histogram**

* Captures fire’s red-orange color pattern
* 96 features

**3. Grayscale Grid Statistics**

* 4×4 grid-based texture analysis
* 48 features

---

### Feature Processing

* Combined feature vector: ~3924 dimensions
* StandardScaler for normalization
* PCA reduces to ~300 dimensions (95% variance retained)

---

### Classification

* Model: **SVM (RBF Kernel)**
* Hyperparameters tuned using GridSearchCV
* Probability output using Platt Scaling

---

## Results

| Metric             | Value              |
| ------------------ | ------------------ |
| Test Accuracy      | **91.18%**         |
| 5-Fold CV Accuracy | **89.58% ± 2.48%** |

### Class Performance

| Class   | Precision | Recall   | F1-score |
| ------- | --------- | -------- | -------- |
| No Fire | 0.88      | **1.00** | 0.94     |
| Fire    | **1.00**  | 0.73     | 0.84     |

**Insight:**

* Perfect precision → no false fire alarms
* Slightly lower recall → some fires missed

---

## Project Structure

```
wild-fire-detection/
│── model/
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── svm_model.pkl
│
│── forest_fire/        # Dataset (~1900 images)
│── static/
│   └── uploads/        # Ignored
│
│── templates/
│   └── *.html
│
│── FireFound-Documentation.docx
│   
│
│── app.py
│── README.md


---

## Dataset

* ~1900 wildfire and non-wildfire images
* Included in repository for easy setup

---


---

## Author

**Saxi Patel**
AI Engineer

