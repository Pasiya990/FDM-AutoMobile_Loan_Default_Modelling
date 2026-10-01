"""LightGBM Baseline Model
Automobile Loan Default Prediction (SLIIT IT3051)

Finding:
- Test set: ROC-AUC = 0.748, PR-AUC = 0.215, Recall = 0.648, Precision = 0.168.
- Catches ~64.8% of defaulters with scale_pos_weight=11.37, outperforming Decision Tree (60.5%).
- Top predictive features: Credit_to_Annuity_Ratio, Mean_Bureau_Score, and Age_Days.

Decision:
- Saved to models/baseline/lightgbm.pkl as our gradient boosting baseline.
- Exported as build_lightgbm_pipeline(), train_lightgbm(), evaluate_lightgbm(), and save_lightgbm_model().
"""

import os
import time
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def _clean_columns(df):
    """Replaces colons and spaces in column names with underscores.
    LightGBM's C++ core requires feature names without special characters.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df.columns = df.columns.astype(str).str.replace(":", "_").str.replace(" ", "_")
    return df


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    """Pipeline transformer to sanitize feature names before passing data to LightGBM."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _clean_columns(X)


def train_lightgbm(
    X_train,
    y_train,
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    scale_pos_weight: float = 11.37,
):
    """Initializes and trains a LightGBM gradient boosted tree model.

    Parameters:
    - n_estimators=100: Number of boosting trees to build.
    - learning_rate=0.05: Step size shrinkage to prevent overfitting.
    - max_depth=5: Limits tree depth to keep individual trees simple.
    - scale_pos_weight=11.37: Balances 91.9% non-defaulters vs 8.1% defaulters.
    """
    # 1. Clean column names for LightGBM
    X_train = _clean_columns(X_train)

    # 2. Initialize the model
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,  # Suppress internal C++ logs
    )

    # 3. Fit on training data and measure training duration
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 2)

    return model, train_time


def build_lightgbm_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    scale_pos_weight: float = 11.37,
    random_state: int = 42,
):
    """Builds an end-to-end scikit-learn Pipeline with preprocessing and LightGBM."""
    from src.preprocessing.preprocessing import build_preprocessing_steps
    from src.preprocessing.feature_engineering import build_feature_engineering_steps

    steps = (
        build_preprocessing_steps()
        + build_feature_engineering_steps()
        + [
            ("feature_name_cleaner", FeatureNameCleaner()),
            (
                "model",
                LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    verbose=-1,
                ),
            ),
        ]
    )
    return Pipeline(steps)


def evaluate_lightgbm(model, X_test, y_test):
    """Evaluates the model on test data and returns metrics and confusion matrix."""
    # 1. Clean test column names
    X_test = _clean_columns(X_test)

    # 2. Predict default classes (0 or 1) and default probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 3. Extract confusion matrix values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # tn = Correct Non-Defaulters
    # fp = Non-Defaulters falsely flagged as Default
    # fn = Real Defaulters missed by model
    # tp = Real Defaulters correctly caught

    # 4. Compute classification metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics, cm


def save_lightgbm_model(model, filepath="models/baseline/lightgbm.pkl"):
    """Saves the trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Quick test run from terminal: python src/models/lightgbm_model.py

    # 1. Load preprocessed datasets
    print("Loading preprocessed data...")
    X_train = joblib.load("data/processed/X_train.joblib")
    y_train = joblib.load("data/processed/y_train.joblib")
    X_test = joblib.load("data/processed/X_test.joblib")
    y_test = joblib.load("data/processed/y_test.joblib")

    # 2. Train the model
    print("Training LightGBM model...")
    model, train_time = train_lightgbm(X_train, y_train)
    print(f"Trained in {train_time:.2f} seconds.")

    # 3. Evaluate the model
    print("Evaluating model...")
    metrics, cm = evaluate_lightgbm(model, X_test, y_test)

    # 4. Display results
    print("\n--- LightGBM Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 5. Save model to disk
    save_lightgbm_model(model)
