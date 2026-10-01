"""Shared evaluation harness: stratified k-fold cross-validation with the
metrics appropriate for an ~8%-positive binary classification problem -
ROC-AUC (primary, threshold-free), PR-AUC (tie-break), recall/precision/F1
for the default class, and the aggregate confusion matrix. Every model in
src/models/ is scored with this same function, so notebook 06's comparison
is apples to apples.
"""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
    cv: int = 5,
    threshold: float = 0.5,
) -> dict:
    """Stratified `cv`-fold CV. `pipeline` is cloned fresh for every fold
    (via sklearn.base.clone, which copies constructor params but never
    fitted state), so the same unfitted pipeline object can be reused
    across models and folds without leaking state between them.

    Returns a dict with the mean/std of each metric across folds, the
    summed confusion matrix, and total fit time - ready to log as one row
    via results_to_row().
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    roc_aucs, pr_aucs, recalls, precisions, f1s = [], [], [], [], []
    total_cm = np.zeros((2, 2), dtype=int)
    fit_seconds = 0.0

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        fold_pipeline = clone(pipeline)

        start = perf_counter()
        fold_pipeline.fit(X_train_fold, y_train_fold)
        fit_seconds += perf_counter() - start

        proba = fold_pipeline.predict_proba(X_val_fold)[:, 1]
        preds = (proba >= threshold).astype(int)

        roc_aucs.append(roc_auc_score(y_val_fold, proba))
        pr_aucs.append(average_precision_score(y_val_fold, proba))
        recalls.append(recall_score(y_val_fold, preds))
        precisions.append(precision_score(y_val_fold, preds, zero_division=0))
        f1s.append(f1_score(y_val_fold, preds, zero_division=0))
        total_cm += confusion_matrix(y_val_fold, preds, labels=[0, 1])

    return {
        "model_name": model_name,
        "cv_folds": cv,
        "threshold": threshold,
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "recall_mean": float(np.mean(recalls)),
        "precision_mean": float(np.mean(precisions)),
        "f1_mean": float(np.mean(f1s)),
        "confusion_matrix": total_cm.tolist(),
        "fit_seconds": round(fit_seconds, 2),
    }


def results_to_row(results: dict) -> pd.DataFrame:
    """Flatten an evaluate_model() result into a one-row DataFrame ready to
    append to experiments/results/experiments.csv."""
    row = {k: v for k, v in results.items() if k != "confusion_matrix"}
    tn, fp, fn, tp = np.array(results["confusion_matrix"]).ravel()
    row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return pd.DataFrame([row])
