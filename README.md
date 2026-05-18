# Eksperimen Dataset Pelatihan - Wine Quality

## Deskripsi
Repository ini berisi eksperimen preprocessing dataset Wine Quality untuk proyek Machine Learning.

## Struktur
`
+-- .github/workflows/preprocessing.yml  # CI Pipeline
+-- preprocessing/
¦   +-- Eksperimen_ardir.ipynb           # Notebook eksperimen
¦   +-- automate_ardir.py                # Script preprocessing otomatis
¦   +-- wine_quality_preprocessing/      # Hasil preprocessing
+-- winequality-red.csv                  # Raw data (red wine)
+-- winequality-white.csv                # Raw data (white wine)
`

## Dataset
- **Source**: UCI Machine Learning Repository
- **Red Wine**: 1599 samples, 11 features
- **White Wine**: 4898 samples, 11 features

## Preprocessing Pipeline
1. Load raw data (red & white wine)
2. Feature engineering & encoding
3. Train/Validation/Test split (70/15/15)
4. Feature scaling
5. Export preprocessed data

## CI/CD
GitHub Actions workflow otomatis menjalankan preprocessing pipeline setiap push.
