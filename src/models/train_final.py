"""Fits the final model on the full training split, evaluates it ONCE on the
held-out test set at the chosen operating threshold, and saves the fitted
pipeline and its metadata to models/final/.

The model, its parameters and its threshold come from
experiments/results/fair_comparison/dedup/selection.json, written by
src/models/fair_comparison.py (selection on cross-validation only).

The test set is used here and nowhere else for model selection. To stop it
being reused by accident, the script refuses to run again once
models/final/metadata.json exists, unless --force is passed.

Usage (from the project root):
    python -m src.models.train_final
"""

import argparse
import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score

from src.config import RANDOM_STATE, TARGET
from src.data.data_cleaning import clean_and_split
from src.models.fair_comparison import SELECTION_PATH, build_rf, spw_for
from src.models.xgboost_model import build_xgboost
from src.preprocessing.feature_engineering import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DIR = PROJECT_ROOT / "models" / "final"

MODEL_FILE = "final_pipeline.joblib"
METADATA_FILE = "metadata.json"


def evaluate_at_threshold(y_true, proba, threshold):
    """Threshold-free scores plus the confusion-matrix metrics at `threshold`."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged.astype(int), labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)),
        "f1": float(f1_score(y_true, flagged.astype(int))),
        "share_flagged": float(flagged.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "rows": int(len(y_true)),
    }


def build_selected_model(selection, y_train):
    """The estimator named in selection.json, with its selected parameters."""
    params = dict(selection["params"])
    if selection["family"] == "xgboost":
        spw = spw_for(y_train, params.pop("spw_factor"))
        return build_xgboost(scale_pos_weight=spw, **params)
    return build_rf(**params)


def train_final(output_dir=FINAL_DIR, selection_path=SELECTION_PATH, force=False):
    output_dir = Path(output_dir)
    metadata_path = output_dir / METADATA_FILE
    if metadata_path.exists() and not force:
        raise FileExistsError(
            f"{metadata_path} already exists, so the held-out test set has already been used for the final model. "
            "Re-running would reuse it. Pass force=True (--force) only if that is intended."
        )

    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    threshold = float(selection["operating_threshold"])

    train_df, test_df = clean_and_split()
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    pipeline = build_pipeline(build_selected_model(selection, y_train))
    pipeline.fit(X_train, y_train)

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_at_threshold(y_test, test_proba, threshold)

    metadata = {
        "model": selection["model"],
        "params": selection["params"],
        "selected_by": selection["selected_by"],
        "trained_on": date.today().isoformat(),
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "input_columns": list(X_train.columns),
        "operating_threshold": threshold,
        "target_recall": float(selection["target_recall"]),
        "cv_reference": {
            **selection["cv_reference"],
            "note": "5-fold stratified CV on the leak-free training split (near-duplicate applicants removed)",
        },
        "oof_at_threshold": selection["oof_at_threshold"],
        "test_set": test_metrics,
        "libraries": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / MODEL_FILE)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return pipeline, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="re-run even if the test set was already used")
    args = parser.parse_args()
    try:
        _, meta = train_final(force=args.force)
    except FileExistsError as error:
        sys.exit(str(error))
    print(json.dumps({k: meta[k] for k in ("model", "operating_threshold", "test_set")}, indent=2))
