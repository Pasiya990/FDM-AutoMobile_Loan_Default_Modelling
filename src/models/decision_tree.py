"""Decision Tree Baseline Model
Automobile Loan Default Prediction (SLIIT IT3051)
"""

import os
import time
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def train_decision_tree(X_train, y_train, max_depth=6, min_samples_leaf=20):
    """Initializes and trains a Decision Tree model.

    Parameters:
    - max_depth=6: Stops the tree from growing too deep (prevents overfitting).
    - min_samples_leaf=20: Requires at least 20 applicants per leaf.
    - class_weight='balanced': Gives more weight to defaulters (handles imbalanced data).
    - random_state=42: Makes results reproducible.
    """
    # 1. Create the Decision Tree model
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )

    # 2. Record start time and fit the model on training data
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time  # Training time in seconds

    return model, train_time


def build_decision_tree_pipeline(
    max_depth: int = 6,
    min_samples_leaf: int = 20,
    random_state: int = 42,
):
    """Builds an end-to-end scikit-learn Pipeline with preprocessing and Decision Tree."""
    from sklearn.pipeline import Pipeline
    from src.preprocessing.feature_engineering import build_pipeline

    return build_pipeline(
        DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
    )


def evaluate_decision_tree(model, X_test, y_test):
    """Evaluates the model on test data and returns metrics and confusion matrix."""
    # 1. Predict class labels (0 or 1)
    y_pred = model.predict(X_test)

    # 2. Predict default probabilities for class 1 (needed for ROC-AUC and PR-AUC)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 3. Compute confusion matrix and extract values
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    # tn = Correct Non-Defaulters
    # fp = Non-Defaulters incorrectly flagged as Default
    # fn = Defaulters missed by model
    # tp = Defaulters correctly caught

    # 4. Calculate classification metrics
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


def save_decision_tree_model(model, filepath="models/baseline/decision_tree.pkl"):
    """Saves the trained model to a file using joblib."""
    # Create the folder if it does not already exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Quick test run from terminal: python src/models/decision_tree.py

    # 1. Load preprocessed data
    print("Loading data...")
    X_train = joblib.load("data/processed/X_train.joblib")
    y_train = joblib.load("data/processed/y_train.joblib")
    X_test = joblib.load("data/processed/X_test.joblib")
    y_test = joblib.load("data/processed/y_test.joblib")

    # 2. Train the model
    print("Training Decision Tree...")
    model, train_time = train_decision_tree(X_train, y_train)
    print(f"Done in {train_time:.2f} seconds.")

    # 3. Evaluate on test set
    print("Evaluating...")
    metrics, cm = evaluate_decision_tree(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 4. Save model to disk
    save_decision_tree_model(model)
