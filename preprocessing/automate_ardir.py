"""
automate_ardir.py
Automated Data Preprocessing Pipeline for Wine Quality Dataset.

Script ini mengonversi langkah-langkah preprocessing dari notebook eksperimen
menjadi pipeline otomatis yang siap digunakan untuk pelatihan model.

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
def load_data(red_path=None, white_path=None):
    """
    Memuat dataset Wine Quality dari file CSV atau URL UCI ML Repository.
    
    Parameters
    ----------
    red_path : str, optional
        Path ke file red wine CSV. Jika None, download dari UCI.
    white_path : str, optional
        Path ke file white wine CSV. Jika None, download dari UCI.
    
    Returns
    -------
    pd.DataFrame
        DataFrame gabungan red dan white wine dengan kolom 'wine_type'.
    """
    RED_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    WHITE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    if red_path and os.path.exists(red_path):
        df_red = pd.read_csv(red_path, sep=';')
    else:
        print("[INFO] Mengunduh dataset red wine dari UCI...")
        df_red = pd.read_csv(RED_URL, sep=';')
    
    if white_path and os.path.exists(white_path):
        df_white = pd.read_csv(white_path, sep=';')
    else:
        print("[INFO] Mengunduh dataset white wine dari UCI...")
        df_white = pd.read_csv(WHITE_URL, sep=';')
    
    # Tambahkan kolom wine_type
    df_red['wine_type'] = 'red'
    df_white['wine_type'] = 'white'
    
    # Gabungkan dataset
    df = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    print(f"[INFO] Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    print(f"[INFO] Red wine: {len(df_red)}, White wine: {len(df_white)}")
    
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
    Membuat fitur baru dan mengkategorikan target variable.
    
    Quality di-bin menjadi 3 kategori:
    - 'low' (quality 3-4)
    - 'medium' (quality 5-6)
    - 'high' (quality 7-9)
    
    Parameters
    ----------
    df : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
    """
    df_eng = df.copy()
    
    # Buat fitur baru
    df_eng['total_acidity'] = df_eng['fixed acidity'] + df_eng['volatile acidity']
    df_eng['free_sulfur_ratio'] = (
        df_eng['free sulfur dioxide'] / df_eng['total sulfur dioxide'].replace(0, 1)
    )
    df_eng['alcohol_density_ratio'] = df_eng['alcohol'] / df_eng['density'].replace(0, 1)
    
    # Kategorikan quality menjadi 3 kelas
    def categorize_quality(q):
        if q <= 4:
            return 'low'
        elif q <= 6:
            return 'medium'
        else:
            return 'high'
    
    df_eng['quality_category'] = df_eng['quality'].apply(categorize_quality)
    
    print(f"[INFO] Feature engineering selesai. Total fitur: {df_eng.shape[1]}")
    print(f"[INFO] Distribusi kelas:\n{df_eng['quality_category'].value_counts()}")
    
    return df_eng


# ============================================================
# 4. ENCODING
# ============================================================
def encode_features(df):
    """
    Encode fitur kategorikal (wine_type dan quality_category).
    
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
    
    # Encode wine_type
    le_wine = LabelEncoder()
    df_enc['wine_type'] = le_wine.fit_transform(df_enc['wine_type'])
    encoders['wine_type'] = le_wine
    
    # Encode quality_category sebagai target
    le_quality = LabelEncoder()
    df_enc['quality_label'] = le_quality.fit_transform(df_enc['quality_category'])
    encoders['quality_category'] = le_quality
    
    print(f"[INFO] Encoding selesai.")
    print(f"[INFO] Label mapping: {dict(zip(le_quality.classes_, le_quality.transform(le_quality.classes_)))}")
    
    return df_enc, encoders


# ============================================================
# 5. SCALING & SPLITTING
# ============================================================
def scale_and_split(df, target_col='quality_label', test_size=0.2, val_size=0.1, random_state=42):
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
    drop_cols = [target_col, 'quality', 'quality_category']
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
def save_preprocessed(data_dict, encoders, output_dir='wine_quality_preprocessing'):
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
def run_preprocessing_pipeline(output_dir='wine_quality_preprocessing'):
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
    print("  WINE QUALITY - AUTOMATED PREPROCESSING PIPELINE")
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
