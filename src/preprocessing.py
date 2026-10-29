"""
preprocessing.py
Feature engineering and preprocessing pipeline for neonatal mortality prediction.
Paper: Ben-Koko & Warribo (2026), WorldS4 2026, Springer LNNS, London
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

FEATURE_COLUMNS = [
    'SBA', 'ANC', 'BIRTH_INTERVAL', 'BIRTH_ORDER',
    'CHILD_SIZE', 'MAT_AGE_FIRST', 'WEALTH_INDEX',
    'GEO_ZONE', 'IMMUNISATION'
]

TARGET_COLUMN = 'NEONATAL_DEATH'

ORDINAL_MAPPINGS = {
    'ANC': {'0': 0, '1-3': 1, '4+': 2},
    'BIRTH_INTERVAL': {'<24': 0, '24-47': 1, '48+': 2},
    'BIRTH_ORDER': {'1st': 0, '2nd-3rd': 1, '4th+': 2},
    'MAT_AGE_FIRST': {'<18': 0, '18-24': 1, '25+': 2},
    'WEALTH_INDEX': {'poorest': 0, 'poor': 1, 'middle': 2, 'rich': 3, 'richest': 4}
}

CATEGORICAL_COLUMNS = ['SBA', 'CHILD_SIZE', 'GEO_ZONE', 'IMMUNISATION']


def load_data(filepath):
    """Load NDHS data from CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def handle_missing_values(df):
    """Handle missing values using median/mode imputation."""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_features(df):
    """Encode ordinal and categorical features."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS, drop_first=True)
    return df


def prepare_features(filepath):
    """Full preprocessing pipeline. Returns X and y."""
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_features(df)
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    print(f"Features: {X.shape} | Class distribution: {y.value_counts().to_dict()}")
    return X, y


def split_and_resample(X, y, test_size=0.2, random_state=42):
    """Stratified train-test split then SMOTE on training set only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"Before SMOTE: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
    return X_train_res, X_test, y_train_res, y_test
