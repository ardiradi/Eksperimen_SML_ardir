"""
automate_ardir.py
Automated Data Preprocessing Pipeline for Heart Disease Dataset.

Script ini mengonversi langkah-langkah preprocessing dari notebook eksperimen
menjadi pipeline otomatis yang siap digunakan untuk pelatihan model.

Dataset: Heart Disease (Cleveland) - UCI ML Repository
Target: Binary classification (0 = no disease, 1 = disease)

Author: ardir
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. DATA LOADING
# ============================================================
def load_data(file_path=None):
    """
    Memuat dataset Heart Disease dari file CSV atau URL UCI ML Repository.
    
    Parameters
    ----------
    file_path : str, optional
        Path ke file CSV. Jika None, download dari UCI.
    
    Returns
    -------
    pd.DataFrame
        DataFrame Heart Disease dataset.
    """
    UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
        'ca', 'thal', 'target'
    ]
    
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path, names=column_names, na_values='?')
    else:
        print("[INFO] Mengunduh dataset Heart Disease dari UCI...")
        df = pd.read_csv(UCI_URL, names=column_names, na_values='?')
    
    # Konversi target menjadi binary (0 = no disease, 1 = disease)
    df['target'] = (df['target'] > 0).astype(int)
    
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[INFO] No Disease: {(df['target'] == 0).sum()}, Disease: {(df['target'] == 1).sum()}")
    
    return df


# ============================================================
# 2. DATA CLEANING
# ============================================================
def clean_data(df):
    """
    Membersihkan dataset dari missing values dan duplikasi.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah.
    
    Returns
    -------
    pd.DataFrame
        DataFrame yang sudah dibersihkan.
    """
    df_clean = df.copy()
    
    # Cek dan handle missing values
    missing = df_clean.isnull().sum()
    if missing.sum() > 0:
        print(f"[INFO] Missing values ditemukan:\n{missing[missing > 0]}")
        # Isi missing values numerik dengan median
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        # Isi missing values kategorikal dengan modus
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    else:
        print("[INFO] Tidak ada missing values.")
    
    # Hapus duplikasi
    n_before = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    n_after = len(df_clean)
    print(f"[INFO] Duplikasi dihapus: {n_before - n_after} baris")
    
    return df_clean


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
def engineer_features(df):
    """
    Membuat fitur baru berdasarkan domain knowledge kardiologi.
    
    Fitur baru yang dibuat:
    - age_group: Kategori umur (young, middle, senior, elderly)
    - heart_rate_reserve: Perbedaan max heart rate dengan resting
    - bp_category: Kategori tekanan darah
    - risk_score: Skor risiko gabungan
    - cholesterol_age_ratio: Rasio kolesterol terhadap umur
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    df_eng = df.copy()
    
    # Buat fitur baru
    # 1. Heart rate reserve (estimasi: 220 - age - resting BP proxy)
    df_eng['heart_rate_reserve'] = df_eng['thalach'] / (220 - df_eng['age']).replace(0, 1)
    
    # 2. Cholesterol-age ratio
    df_eng['cholesterol_age_ratio'] = df_eng['chol'] / df_eng['age'].replace(0, 1)
    
    # 3. BP-cholesterol interaction
    df_eng['bp_chol_interaction'] = df_eng['trestbps'] * df_eng['chol'] / 10000
    
    # 4. Exercise-induced risk (oldpeak * exang)
    df_eng['exercise_risk'] = df_eng['oldpeak'] * (df_eng['exang'] + 1)
    
    # 5. Age-sex risk factor
    df_eng['age_sex_risk'] = df_eng['age'] * (1 + df_eng['sex'] * 0.2)
    
    print(f"[INFO] Feature engineering selesai. Total fitur: {df_eng.shape[1]}")
    print(f"[INFO] Distribusi target:\n{df_eng['target'].value_counts()}")
    
    return df_eng


# ============================================================
# 4. ENCODING
# ============================================================
def encode_features(df):
    """
    Encode fitur kategorikal jika diperlukan.
    Sebagian besar fitur Heart Disease sudah dalam format numerik.
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame, dict
        DataFrame yang sudah di-encode dan dictionary label encoders.
    """
    df_enc = df.copy()
    encoders = {}
    
    # Target sudah binary (0/1), tidak perlu encoding
    # Simpan label mapping untuk referensi
    le_target = LabelEncoder()
    le_target.classes_ = np.array(['no_disease', 'disease'])
    encoders['target'] = le_target
    
    # Pastikan semua kolom numerik
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le
    
    print(f"[INFO] Encoding selesai.")
    print(f"[INFO] Label mapping: {{0: 'no_disease', 1: 'disease'}}")
    
    return df_enc, encoders


# ============================================================
# 5. SCALING & SPLITTING
# ============================================================
def scale_and_split(df, target_col='target', test_size=0.2, val_size=0.1, random_state=42):
    """
    Melakukan scaling pada fitur numerik dan membagi data menjadi
    train, validation, dan test set.
    
    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    test_size : float
    val_size : float
    random_state : int
    
    Returns
    -------
    dict
        Dictionary berisi X_train, X_val, X_test, y_train, y_val, y_test, scaler.
    """
    # Tentukan fitur dan target
    drop_cols = [target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Split: train + val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split: train vs val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )
    
    # Scaling
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    print(f"[INFO] Data split selesai:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler,
        'feature_names': num_cols
    }


# ============================================================
# 6. SAVE PREPROCESSED DATA
# ============================================================
def save_preprocessed(data_dict, encoders, output_dir='heart_disease_preprocessing'):
    """
    Menyimpan data yang sudah dipreproses ke folder output.
    
    Parameters
    ----------
    data_dict : dict
    encoders : dict
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Simpan datasets
    data_dict['X_train'].to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    data_dict['X_val'].to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    data_dict['X_test'].to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    data_dict['y_train'].to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    data_dict['y_val'].to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)
    data_dict['y_test'].to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Simpan scaler dan encoders
    joblib.dump(data_dict['scaler'], os.path.join(output_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(output_dir, 'encoders.pkl'))
    
    # Simpan feature names
    pd.Series(data_dict['feature_names']).to_csv(
        os.path.join(output_dir, 'feature_names.csv'), index=False
    )
    
    print(f"[INFO] Data preprocessed berhasil disimpan ke: {output_dir}/")


# ============================================================
# MAIN PIPELINE
# ============================================================
def run_preprocessing_pipeline(output_dir='heart_disease_preprocessing'):
    """
    Menjalankan keseluruhan pipeline preprocessing secara otomatis.
    
    Parameters
    ----------
    output_dir : str
        Direktori output untuk menyimpan data yang sudah dipreproses.
    
    Returns
    -------
    dict
        Dictionary berisi data yang siap dilatih.
    """
    print("=" * 60)
    print("  HEART DISEASE - AUTOMATED PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[STEP 1/5] Loading data...")
    df = load_data()
    
    # Step 2: Clean data
    print("\n[STEP 2/5] Cleaning data...")
    df = clean_data(df)
    
    # Step 3: Feature engineering
    print("\n[STEP 3/5] Feature engineering...")
    df = engineer_features(df)
    
    # Step 4: Encoding
    print("\n[STEP 4/5] Encoding features...")
    df, encoders = encode_features(df)
    
    # Step 5: Scale and split
    print("\n[STEP 5/5] Scaling and splitting...")
    data_dict = scale_and_split(df)
    
    # Save
    print("\n[SAVE] Menyimpan hasil preprocessing...")
    save_preprocessed(data_dict, encoders, output_dir)
    
    print("\n" + "=" * 60)
    print("  PREPROCESSING PIPELINE SELESAI!")
    print("=" * 60)
    
    return data_dict, encoders


if __name__ == '__main__':
    data, encoders = run_preprocessing_pipeline()
