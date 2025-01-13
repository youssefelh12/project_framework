# Face Recognition & Classification Pipeline

This repository provides a comprehensive pipeline for:
1. Cleaning and preprocessing label data
2. Detecting and cropping faces from images/videos using **YuNet (OpenCV)**
3. Training classification models using **FaceNet** embeddings
4. Predicting on new/test images
5. Evaluating the model’s performance

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Clean Labels](#1-clean-labels)
  - [2. Crop Faces (Trainset)](#2-crop-faces-trainset)
  - [3. Crop Faces (Testset)](#3-crop-faces-testset)
  - [4. Train Model](#4-train-model)
  - [5. Test a Single Image](#5-test-a-single-image)
  - [6. Predict on a Batch of Images](#6-predict-on-a-batch-of-images)
  - [7. Evaluate Accuracy](#7-evaluate-accuracy)
- [Models & Checkpoints](#models--checkpoints)
- [Credits](#credits)

---

## Prerequisites

- **Python 3.7+** (recommended 3.8+)
- A working C/C++ compiler toolchain (for some libraries like `xgboost`, `catboost`, etc.)
- (Optional) GPU support if using TensorFlow for embedding generation

---

## Directory Structure

Here is the updated directory structure:

```
your_project/
├── classifiers/           # Trained models, normalizers, label encoders, etc.
├── data/
│   ├── trainset/          # Original training images/videos
│   ├── testset/           # Original test images/videos
│   ├── sample_submission.csv
│   ├── cleaned_labels.csv  # Generated after cleaning
│   └── testset_labels.csv  # Ground-truth labels for test evaluation
│
├── models/                # Face detection models (e.g., YuNet)
│   └── face_detection_yunet_2023mar.onnx
│
├── processed_data/
│   ├── cropped_faces/         # Cropped faces from trainset
│   ├── cropped_faces_test/    # Cropped faces from testset
│
├── results/              # Results such as prediction outputs
│   └── predictions.csv  # Example output CSV of predictions
│
├── scripts/              # All scripts for the pipeline
│   ├── clean_labels.py
│   ├── crop_faces.py
│   ├── crop_faces_test.py
│   ├── training_models.py
│   ├── test_predictions.py
│   ├── predictions.py
│   └── evaluation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/youssefelh12/facial_recognition_project.git
   cd your-repo
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   # or
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Clean Labels

- **Script**: `scripts/cleanlabels.py`  
- **Purpose**: Reads `sample_submission.csv`, cleans and corrects label names, and saves the results to `cleaned_labels.csv`.  
- **Run**:
   ```bash
   python scripts/cleanlabels.py
   ```

---

### 2. Crop Faces (Trainset)

- **Script**: `scripts/crop_faces.py`  
- **Purpose**: Detects and crops faces in `data/trainset` images or videos. Outputs cropped faces to `processed_data/cropped_faces/`.  
- **Run**:
   ```bash
   python scripts/crop_faces.py
   ```

---

## Optional: Data Augmentation

- **Script**: `augment.py`  
- **Purpose**: Augments the cropped face images to increase dataset size and variety. This step is optional but may help improve model performance.  
- **Run**:
   ```bash
   python augment.py
   ```

---

### 3. Crop Faces (Testset)

- **Script**: `scripts/crop_faces_test.py`  
- **Purpose**: Detects and crops faces in `data/testset` images or videos. Outputs cropped faces to `processed_data/cropped_faces_test/`.  
- **Run**:
   ```bash
   python scripts/crop_faces_test.py
   ```

---

### 4. Train Model

- **Script**: `scripts/training_models.py`  
- **Purpose**:  
  1. Loads cropped face images from `processed_data/cropped_faces/`  
  2. Generates FaceNet embeddings  
  3. Trains a classifier (e.g., CatBoost, XGBoost, SVM)  
  4. Saves the trained model, normalizer, and label encoder to `classifiers/`  

- **Run**:
   ```bash
   python scripts/training_models.py
   ```

---

### 5. Test a Single Image

- **Script**: `scripts/test_predictions.py`  
- **Purpose**: Tests how the trained model classifies a single cropped face image.  
- **Run**:
   ```bash
   python scripts/test_predictions.py
   ```
- Update the `image_path` variable in the script to point to your test image.

---

### 6. Predict on a Batch of Images

- **Script**: `scripts/predictions.py`  
- **Purpose**: Generates predictions for all cropped testset images and saves them to a CSV file in the `results/` directory.  
- **Run**:
   ```bash
   python scripts/predictions.py
   ```

---

### 7. Evaluate Accuracy

- **Script**: `scripts/evaluation.py`  
- **Purpose**: Compares predictions against ground truth labels (`testset_labels.csv`) and calculates metrics:  
  - Hamming Loss
  - Precision / Recall / F1
  - Order-sensitive Accuracy  

- **Run**:
   ```bash
   python scripts/evaluation.py
   ```



## Models & Checkpoints

- **YuNet Model**: `models/face_detection_yunet_2023mar.onnx` (for face detection)  
- **Trained Classifiers**: Saved in `classifiers/` (e.g., `catboost_face_classifier.pkl`)  
- **Normalizer**: `classifiers/catboost/catboost_normalizer.pkl`  
- **Label Encoder**: `classifiers/catboost/catboost_label_encoder.pkl`  
- **normalizer**: `classifiers/catboost/catboost_normalizer.pkl` 

Ensure the file paths in scripts match your setup.

---

## Credits

- **OpenCV** for YuNet-based face detection  
- **FaceNet** (via [keras-facenet](https://pypi.org/project/keras-facenet/)) for embedding generation  
- Various classifiers: CatBoost, XGBoost, LightGBM, scikit-learn, etc.  

For questions or issues, feel free to open an Issue or submit a Pull Request.
```

