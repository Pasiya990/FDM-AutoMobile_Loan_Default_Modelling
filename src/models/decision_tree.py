"""Decision Tree Baseline Model
Automobile Loan Default Prediction (SLIIT IT3051)

Finding:
- 5-fold CV: ROC-AUC = 0.708, PR-AUC = 0.181, Recall = 0.607, Precision = 0.152.
- Held-out test set, reference only: ROC-AUC = 0.719, PR-AUC = 0.186, Recall = 0.587, Precision = 0.163.
- Catches about 61% of defaulters with balanced weights, but has many false alarms.

Decision:
- Interpretable non-linear baseline, built with build_decision_tree_pipeline().
- The fitted pipeline is saved to models/baseline/decision_tree.pkl by notebooks/decision_tree.ipynb.
"""

import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing.feature_engineering import build_pipeline


def build_decision_tree_pipeline(
    max_depth: int = 6,
    min_samples_leaf: int = 20,
    random_state: int = RANDOM_STATE,
):
    """Builds an end-to-end scikit-learn Pipeline with preprocessing and Decision Tree.

    - max_depth=6: stops the tree from growing too deep (prevents overfitting).
    - min_samples_leaf=20: requires at least 20 applicants per leaf.
    - class_weight='balanced': gives more weight to defaulters (handles imbalanced data).
    """
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
