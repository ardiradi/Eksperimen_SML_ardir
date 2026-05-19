# 🔬 Heart Disease MLOps — Experimentation & Preprocessing

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Dataset-UCI%20Heart%20Disease-red?logo=data:image/svg+xml;base64,..." alt="Dataset">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Status">
</p>

## 📋 Overview

Tahap **Eksperimen & Preprocessing** dari end-to-end MLOps pipeline untuk prediksi penyakit jantung menggunakan dataset [Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease) dari UCI ML Repository.

Pipeline ini melakukan:
- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Missing Value Handling
- ⚙️ Feature Engineering (5 fitur baru)
- 📦 Export hasil preprocessing ke CSV & PKL

## 🗂️ Struktur Direktori

```
├── Eksperimen_ardir.ipynb          # Notebook eksperimen lengkap (Run All)
├── automate_ardir.py               # Script preprocessing otomatis
├── heart_disease_preprocessing/    # Hasil preprocessing
│   ├── X_train.csv                 # Training features (70%)
│   ├── X_val.csv                   # Validation features (10%)
│   ├── X_test.csv                  # Test features (20%)
│   ├── y_train.csv                 # Training labels
│   ├── y_val.csv                   # Validation labels
│   ├── y_test.csv                  # Test labels
│   ├── scaler.pkl                  # StandardScaler fitted
│   ├── encoders.pkl                # Label encoders
│   └── feature_names.csv           # Nama fitur
└── .github/workflows/
    └── preprocessing.yml           # GitHub Actions workflow
```

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | UCI ML Repository — Cleveland |
| **Samples** | 303 |
| **Features** | 13 original + 5 engineered = 18 |
| **Target** | Binary (0: No Disease, 1: Disease) |
| **Class Balance** | 164 no disease / 139 disease |

## ⚙️ Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `heart_rate_reserve` | `thalach / (220 - age)` | Kapasitas jantung relatif |
| `cholesterol_age_ratio` | `chol / age` | Risiko kolesterol per usia |
| `bp_chol_interaction` | `trestbps × chol / 10000` | Interaksi tekanan darah & kolesterol |
| `exercise_risk` | `oldpeak × (exang + 1)` | Risiko saat exercise |
| `age_sex_risk` | `age × (1 + sex × 0.2)` | Risiko berdasarkan usia & jenis kelamin |

## 🚀 Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/ardiradi/Eksperimen_SML_ardir.git
cd Eksperimen_SML_ardir

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Jalankan preprocessing otomatis
python automate_ardir.py

# Atau buka notebook untuk eksperimen interaktif
jupyter notebook Eksperimen_ardir.ipynb
```

## 📈 Hasil Eksperimen

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| **Random Forest** | ~0.98 | ~0.87 | ~0.85 |
| Gradient Boosting | ~0.99 | ~0.84 | ~0.82 |
| Logistic Regression | ~0.86 | ~0.84 | ~0.82 |

## 🔗 Related Repositories

| Component | Repository |
|-----------|------------|
| 📦 Model Building | [Membangun-Model-SML](https://github.com/ardiradi/Membangun-Model-SML) |
| 🔄 CI/CD Workflow | [Workflow-CI](https://github.com/ardiradi/Workflow-CI) |
| 📊 Monitoring | [Monitoring-Logging-SML](https://github.com/ardiradi/Monitoring-Logging-SML) |

---

<p align="center">
  <b>Part of the Heart Disease MLOps Pipeline</b><br>
  Built as part of Dicoding — Membangun Sistem Machine Learning
</p>
