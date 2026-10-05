"""LightGBM Baseline Model
Automobile Loan Default Prediction (SLIIT IT3051)

Finding (5-fold stratified CV, shared pipeline, threshold 0.5):
- Baseline: ROC-AUC = 0.748, PR-AUC = 0.228, Recall = 0.644, Precision = 0.168.
- Tuned (notebooks/lightgbm.ipynb): ROC-AUC = 0.757, PR-AUC = 0.248, Recall = 0.560, Precision = 0.203.
- scale_pos_weight lets it catch about 64% of defaulters at 0.5, with many false alarms.

Decision:
- Gradient boosting baseline, built with build_lightgbm_pipeline().
"""

import os

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing.feature_engineering import build_feature_engineering_steps
from src.preprocessing.preprocessing import build_preprocessing_steps

# Weight for the default class: approximates the training split's ratio of
# non-defaulters to defaulters (87,648 / 7,724 = 11.35). Kept as a fixed
# hyperparameter so the logged CV results stay reproducible.
SCALE_POS_WEIGHT = 11.37


def _clean_columns(df):
    """Replaces colons and spaces in column names with underscores.
    LightGBM's C++ core requires feature names without special characters.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        df.columns = df.columns.astype(str).str.replace(":", "_").str.replace(" ", "_")
    return df


class FeatureNameCleaner(BaseEstimator, TransformerMixin):
    """Pipeline transformer to sanitize feature names before passing data to LightGBM.

    fit() records the cleaned column names so scikit-learn sees the step as
    fitted; without a fitted attribute, a sliced pipeline ending here (for
    example pipe[:-1]) is reported as not fitted.
    """

    def fit(self, X, y=None):
        self.output_columns_ = list(_clean_columns(X).columns)
        return self

    def transform(self, X):
        return _clean_columns(X)


def build_lightgbm_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    scale_pos_weight: float = SCALE_POS_WEIGHT,
    random_state: int = RANDOM_STATE,
):
    """Builds an end-to-end scikit-learn Pipeline with preprocessing and LightGBM.

    - n_estimators=100: number of boosting trees to build.
    - learning_rate=0.05: step size shrinkage to prevent overfitting.
    - max_depth=5: limits tree depth to keep individual trees simple.
    - scale_pos_weight: balances non-defaulters vs defaulters (see SCALE_POS_WEIGHT).
    """
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
