"""
automate_Dicoding-Student.py
Automated preprocessing pipeline untuk Heart Disease dataset.
Membaca dari file heart_disease_raw.csv yang sudah ada.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
# Path ke raw dataset - sesuaikan jika perlu
RAW_DATA_PATHS = [
    'heart_disease_raw.csv',           # jika dijalankan dari root repo
    '../heart_disease_raw.csv',        # jika dijalankan dari folder preprocessing
]

COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
IMPUTE_COLS    = ['ca', 'thal']
SCALE_COLS     = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
OUTPUT_DIR     = 'heart_disease_preprocessing'


def find_raw_data():
    """Cari file heart_disease_raw.csv di beberapa lokasi."""
    for path in RAW_DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "heart_disease_raw.csv tidak ditemukan! "
        "Pastikan file ada di root repo atau folder preprocessing."
    )


def load_data() -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    print('[1/7] Loading data...')
    raw_path = find_raw_data()

    # Coba baca dengan header dulu
    df = pd.read_csv(raw_path, na_values='?')

    # Jika kolom tidak sesuai, baca tanpa header
    if list(df.columns) != COLUMN_NAMES:
        df = pd.read_csv(raw_path, names=COLUMN_NAMES, na_values='?', header=None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'      ✓ Data dimuat dari: {raw_path}')
    print(f'      ✓ Shape: {df.shape}')
    return df


def convert_target(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi target ke binary (0/1)."""
    print('[2/7] Converting target to binary...')
    df = df.copy()
    df['target'] = (df['target'] > 0).astype(int)
    print(f'      ✓ No Disease: {(df["target"]==0).sum()}, Disease: {(df["target"]==1).sum()}')
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values dengan median imputation."""
    print('[3/7] Handling missing values...')
    before = df[IMPUTE_COLS].isnull().sum().sum()
    imputer = SimpleImputer(strategy='median')
    df = df.copy()
    df[IMPUTE_COLS] = imputer.fit_transform(df[IMPUTE_COLS])
    after = df[IMPUTE_COLS].isnull().sum().sum()
    print(f'      ✓ Missing values: {before} → {after}')
    return df


def handle_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Handle outlier dengan IQR capping (winsorization)."""
    print('[4/7] Handling outliers (IQR capping)...')
    df = df.copy()
    for col in NUMERICAL_COLS:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        clipped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if clipped > 0:
            print(f'      ✓ {col}: {clipped} nilai di-clip')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: konversi tipe data dan tambah age_group."""
    print('[5/7] Feature engineering...')
    df = df.copy()
    df['ca']   = df['ca'].astype(int)
    df['thal'] = df['thal'].astype(int)
    df['age_group'] = pd.cut(
        df['age'], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3]
    ).astype(int)
    print(f'      ✓ Total fitur: {df.shape[1]-1}')
    return df


def split_and_scale(df: pd.DataFrame):
    """Split data dan lakukan feature scaling."""
    print('[6/7] Splitting and scaling...')
    X, y = df.drop('target', axis=1), df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])
    print(f'      ✓ Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test


def save_preprocessed_data(X_train, X_test, y_train, y_test) -> dict:
    """Simpan hasil preprocessing ke folder output."""
    print('[7/7] Saving preprocessed data...')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = X_train.copy(); train_df['target'] = y_train.values
    test_df  = X_test.copy();  test_df['target']  = y_test.values

    train_path = os.path.join(OUTPUT_DIR, 'train.csv')
    test_path  = os.path.join(OUTPUT_DIR, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f'      ✓ Train: {train_path}')
    print(f'      ✓ Test : {test_path}')
    return {'train_path': train_path, 'test_path': test_path}


def run_preprocessing_pipeline() -> dict:
    """Jalankan seluruh pipeline preprocessing secara otomatis."""
    print('=' * 55)
    print('  AUTOMATED PREPROCESSING PIPELINE')
    print('  Heart Disease Classification Dataset')
    print('=' * 55)

    df = load_data()
    df = convert_target(df)
    df = handle_missing_values(df)
    df = handle_outliers_iqr(df)
    df = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_and_scale(df)
    result = save_preprocessed_data(X_train, X_test, y_train, y_test)

    # Verifikasi output
    train_check = pd.read_csv(result['train_path'])
    test_check  = pd.read_csv(result['test_path'])

    print('\n' + '=' * 55)
    print('  PREPROCESSING SELESAI ✓')
    print('=' * 55)
    print(f'  Train set : {train_check.shape[0]} baris, {train_check.shape[1]} kolom')
    print(f'  Test set  : {test_check.shape[0]} baris, {test_check.shape[1]} kolom')
    print(f'  Missing   : {train_check.isnull().sum().sum()} nilai')
    print(f'  Output    : ./{OUTPUT_DIR}/')
    print('=' * 55)

    return result


if __name__ == '__main__':
    run_preprocessing_pipeline()
