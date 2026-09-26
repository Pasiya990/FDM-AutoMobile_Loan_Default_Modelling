"""
Data Preprocessing Pipeline for Automobile Loan Default Prediction.

Why we fit ONLY on X_train (Preventing Data Leakage):
-----------------------------------------------------
Data leakage happens if the test set is used to calculate statistics like
medians or standard deviations. In real life, future loan applicants arrive
one at a time, so the bank cannot know future numbers in advance.
By calculating medians and scales ONLY on X_train, and simply applying them to
X_test, we ensure the test evaluation is 100% fair and realistic.
"""

import os
import sys

# Add project root to sys.path so we can import 'src' easily
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.data_cleaning import clean_raw_data
from src.preprocessing.feature_engineering import create_domain_features

# List of all text (categorical) columns in the dataset
CATEGORICAL_COLS = [
    'Accompany_Client', 'Client_Income_Type', 'Client_Education',
    'Client_Marital_Status', 'Client_Gender', 'Loan_Contract_Type',
    'Client_Housing_Type', 'Client_Occupation', 'Client_Permanent_Match_Tag',
    'Client_Contact_Work_Tag', 'Type_Organization'
]


def preprocess_data(
    data_path: str = 'data/raw/Train_Dataset.csv',
    output_dir: str = 'data/processed',
    model_dir: str = 'models'
):
    """
    Runs the entire preprocessing pipeline step-by-step:
    1. Loads, cleans, and engineers features.
    2. Splits into train (80%) and test (20%) sets.
    3. Fills missing values and scales numeric columns.
    4. Fills missing values and one-hot encodes text columns.
    5. Saves the final processed data and preprocessor tools to disk.
    """
    # 1. Load data and run initial cleaning and feature engineering
    print(f"Loading data from '{data_path}'...")
    df = pd.read_csv(data_path, low_memory=False)
    df = clean_raw_data(df)                    # Fixes rogue characters ($, x, &) and drops ID
    df = create_domain_features(df)            # Adds debt ratios, age in years, bureau scores

    # 2. Separate input features (X) from the target answer (y: Default)
    X = df.drop(columns=['Default'])
    y = df['Default']

    # 3. Split data into 80% train and 20% test BEFORE doing any transformations
    # stratify=y preserves the exact 91.9% vs 8.1% class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Identify which columns are categorical and which are numerical
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # 4. Numeric Imputation & Scaling
    # SimpleImputer replaces missing values with the median of each column
    # We call fit_transform on X_train, but ONLY transform on X_test (prevents data leakage)
    num_imputer = SimpleImputer(strategy='median')
    X_train_num_imp = num_imputer.fit_transform(X_train[num_cols].apply(pd.to_numeric, errors='coerce'))
    X_test_num_imp = num_imputer.transform(X_test[num_cols].apply(pd.to_numeric, errors='coerce'))

    # StandardScaler standardizes numbers to have mean=0 and standard deviation=1
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num_imp)
    X_test_num_scaled = scaler.transform(X_test_num_imp)

    # 5. Categorical Imputation & One-Hot Encoding
    # Fill any missing text with the most frequent value (mode)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train_cat_imp = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat_imp = cat_imputer.transform(X_test[cat_cols])

    # OneHotEncoder converts categories into separate 0/1 binary columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat_enc = encoder.fit_transform(X_train_cat_imp)
    X_test_cat_enc = encoder.transform(X_test_cat_imp)

    # 6. Combine scaled numeric columns and one-hot encoded columns back together
    all_feature_names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    X_train_proc = pd.DataFrame(np.hstack([X_train_num_scaled, X_train_cat_enc]), columns=all_feature_names)
    X_test_proc = pd.DataFrame(np.hstack([X_test_num_scaled, X_test_cat_enc]), columns=all_feature_names)

    # 7. Save the processed data and fitted preprocessors using joblib
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save training and testing tables for model training
    joblib.dump(X_train_proc, os.path.join(output_dir, 'X_train.joblib'))
    joblib.dump(X_test_proc, os.path.join(output_dir, 'X_test.joblib'))
    joblib.dump(y_train.reset_index(drop=True), os.path.join(output_dir, 'y_train.joblib'))
    joblib.dump(y_test.reset_index(drop=True), os.path.join(output_dir, 'y_test.joblib'))

    # Save fitted preprocessor objects so the web app / test script can use them later
    preprocessor = {
        'num_imputer': num_imputer,
        'scaler': scaler,
        'cat_imputer': cat_imputer,
        'encoder': encoder,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'feature_names': all_feature_names,
    }
    joblib.dump(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))

    print(f"Preprocessing complete! Total features: {len(all_feature_names)}")
    print(f"Saved artifacts to '{output_dir}/' and '{model_dir}/preprocessor.joblib'")
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor


if __name__ == '__main__':
    preprocess_data()
