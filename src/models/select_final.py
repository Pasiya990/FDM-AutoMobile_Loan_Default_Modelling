"""Writes selection.json for the final model from the notebook 07 XGBoost search.

train_final.py reads its model, parameters and operating threshold from
experiments/results/fair_comparison/dedup/selection.json. That file is normally
written by src/models/fair_comparison.py, which runs its own long parameter
search. The comparison in notebook 06b ranks the XGBoost configuration found by
notebook 07 first, so this script builds the same file from that configuration
(experiments/results/xgboost_best_parameters.csv) without a new search:

  1. run stratified 5-fold CV on the training split with the project pipeline,
     the same folds as evaluate_model();
  2. pick the operating threshold from the pooled out-of-fold predictions with
     the project rule (highest threshold that still reaches recall 0.60);
  3. write selection.json, plus the out-of-fold predictions (used to choose the
     risk bands).

The held-out test set is not touched here.

Usage (from the project root):
    python -m src.models.select_final
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE, TARGET
from src.data.data_cleaning import clean_and_split
from src.models.fair_comparison import SELECTION_PATH, TARGET_RECALL, at_threshold, spw_for, threshold_for_recall
from src.models.xgboost_model import build_xgboost
from src.preprocessing.feature_engineering import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
BEST_PARAMETERS_PATH = RESULTS_DIR / "xgboost_best_parameters.csv"
OOF_PATH = RESULTS_DIR / "final_oof_predictions.csv"

CV_FOLDS = 5
INTEGER_PARAMETERS = {"n_estimators", "max_depth"}
SELECTED_BY = "highest mean 5-fold CV ROC-AUC in notebook 06b (XGBoost configuration from the notebook 07 search)"


def load_xgboost_parameters(path=BEST_PARAMETERS_PATH):
    """Best parameters saved by notebook 07, without the pipeline's 'model__' prefix."""
    table = pd.read_csv(path)
    params = {}
    for name, value in zip(table["parameter"], table["value"]):
        name = name.removeprefix("model__")
        params[name] = int(round(value)) if name in INTEGER_PARAMETERS else float(value)
    return params


def out_of_fold_probabilities(params, X, y):
    """Pooled out-of-fold default scores (stratified, shuffled, seed 42), the fold of each row, and per-fold metrics."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba = np.zeros(len(y))
    fold_of_row = np.zeros(len(y), dtype=int)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_train = y.iloc[train_idx]
        model = build_xgboost(scale_pos_weight=spw_for(y_train, 1.0), **params)
        pipeline = clone(build_pipeline(model))
        pipeline.fit(X.iloc[train_idx], y_train)
        p = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        proba[val_idx] = p
        fold_of_row[val_idx] = fold
        rows.append({"roc_auc": roc_auc_score(y.iloc[val_idx], p), "pr_auc": average_precision_score(y.iloc[val_idx], p)})
        print(f"fold {fold}: ROC-AUC {rows[-1]['roc_auc']:.4f}, PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)
    return proba, fold_of_row, pd.DataFrame(rows)


def build_selection(params, y, proba, fold_metrics):
    """selection.json content, in the layout train_final.py reads."""
    threshold = threshold_for_recall(y, proba)
    at = at_threshold(y, proba, threshold)
    return {
        "model": "xgboost_tuned",
        "family": "xgboost",
        "params": {**params, "spw_factor": 1.0},
        "selected_by": SELECTED_BY,
        "operating_threshold": float(threshold),
        "target_recall": TARGET_RECALL,
        "oof_at_threshold": {k: float(at[k]) for k in ("recall", "precision", "share_flagged")},
        "cv_reference": {
            "roc_auc_mean": float(fold_metrics["roc_auc"].mean()),
            "roc_auc_std": float(fold_metrics["roc_auc"].std(ddof=0)),
            "pr_auc_mean": float(fold_metrics["pr_auc"].mean()),
            "pr_auc_std": float(fold_metrics["pr_auc"].std(ddof=0)),
        },
    }


def select_final(selection_path=SELECTION_PATH, oof_path=OOF_PATH, parameters_path=BEST_PARAMETERS_PATH):
    params = load_xgboost_parameters(parameters_path)
    train_df, _ = clean_and_split()  # held-out test rows are discarded here
    X, y = train_df.drop(columns=[TARGET]), train_df[TARGET]

    proba, fold_of_row, fold_metrics = out_of_fold_probabilities(params, X, y)
    selection = build_selection(params, y, proba, fold_metrics)

    selection_path = Path(selection_path)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"y_true": y.to_numpy(), "fold": fold_of_row, "xgboost_tuned": proba}).to_csv(oof_path, index=False)
    return selection


if __name__ == "__main__":
    print(json.dumps(select_final(), indent=2))
