# EEG Classification Model

## 🧠 Overview

This repository contains a deep learning model for classifying EEG data into **7 different mental health disorder categories**. The model is built using **PyTorch** and processes EEG signals to predict diagnostic labels.

---

## 📊 Dataset

The dataset used is:  
**`EEG.machinelearing_data_BRMH.csv`** (from Kaggle)  
It includes EEG recordings labeled with the following categories:

- Addictive disorder  
- Trauma and stress related disorder  
- Mood disorder  
- Healthy control  
- Obsessive compulsive disorder  
- Schizophrenia  
- Anxiety disorder  

---

## 🏗️ Model Architectures

### 1. Full Attribute Model

- **Input:** 1140 features (all EEG channels)  
- **Architecture:**
  - `Linear(1140, 700)` → ReLU  
  - `Linear(700, 343)` → ReLU  
  - `Linear(343, 49)` → ReLU  
  - `Linear(49, 7)` → Sigmoid  
- **Loss Function:** `CrossEntropyLoss` with class weighting  
- **Optimizer:** `SGD` (learning rate: 0.001)  

### 2. PCA-Reduced Model

- **Input:** First 15 principal components (explaining most variance)  
- **Architecture:**
  - `Linear(15, 700)` → ReLU  
  - `Linear(700, 343)` → ReLU  
  - `Linear(343, 49)` → ReLU  
  - `Linear(49, 7)` → Sigmoid  
- **Loss & Optimizer:** Same as above  

---

## ⚙️ Implementation Details

### 🔄 Data Preprocessing

- Mounted Google Drive to access dataset  
- Split data into **80% train** / **20% test**  
- Removed `NaN` values from column 114  
- Converted text labels to numerical classes  
- One-hot encoded labels into 7-dimensional vectors  

### 📉 Class Distribution

- The dataset is imbalanced  
- Handled using:
  - Class weighting in the loss function  
  - Visualizations of class distribution  

---

## 🏋️ Training

- Trained both models for **1000 iterations**  
- Tracked loss on **training and test** sets  
- Accuracy calculated by comparing predicted vs. true labels  

---

## 📈 Evaluation

- Reported training and test accuracy  
- Generated confusion matrices for both sets  

---

## 🧪 Results

### Full Attribute Model
- **Training accuracy:** 39.68%  
- **Test accuracy:** 26.46%  

### PCA-Reduced Model (15 components)
- **Training accuracy:** 50.26%  
- **Test accuracy:** 18.52%  

---

## ▶️ Usage

1. Mount Google Drive and specify dataset path  
2. Run preprocessing cells  
3. Select model architecture (full or PCA-reduced)  
4. Train model (**1000 iterations** by default)  
5. Evaluate using accuracy + confusion matrix  

---

## 📦 Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📝 Notes

- Models show **relatively low accuracy**, indicating room for improvement  
- PCA did not improve test performance, despite better training accuracy  
- **Class imbalance** remains a challenge  
- Further hyperparameter tuning is recommended  

---

## 🔮 Future Work

- Try different architectures (e.g., **CNNs**, **RNNs**)  
- Use advanced techniques for handling class imbalance  
- Implement **early stopping**  
- Add **regularization** to reduce overfitting  
- Explore alternative **dimensionality reduction** methods  

