"""Ablation study: do feature engineering, feature selection and the uncertain
columns actually help the final model? (Assignment Stage 7; plan M4-O2.)

Every variant uses the final XGBoost configuration (selection.json) and the same
stratified 5-fold CV on the training split as evaluate_model(), so the folds are
identical and the per-fold differences from the full pipeline are paired. The
held-out test set is not used.

Variants
    full                    the final pipeline, unchanged
    no_engineered_features  without the columns we created: score summaries
                            (score_mean, score_min, scores_available), income
                            ratios, derived features (Age_Years, Credit_Loan_Ratio,
                            has_children, contact_count; Age_Days and
                            Loan_Annuity are then kept as given) and the
                            "_was_missing" flags
    no_feature_selection    without the importance-based selector
    no_bureau_scores        without Score_Source_1/2/3 and their summaries: how
                            much the model depends on bureau scores, which are
                            often missing in practice
    no_uncertain_columns    without Active_Loan and Social_Circle_Default, whose
                            meaning in the dataset is uncertain

Usage (from the project root): python -m src.models.ablation
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE, SCORE_COLS, TARGET
from src.models.fair_comparison import SELECTION_PATH, spw_for
from src.models.xgboost_model import build_xgboost
from src.preprocessing.feature_engineering import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "ablation.csv"
CV_FOLDS = 5
UNCERTAIN_COLUMNS = ["Active_Loan", "Social_Circle_Default"]

# variant -> (input columns to remove, pipeline steps switched off, extra step parameters)
VARIANTS = {
    "full": ([], [], {}),
    "no_engineered_features": (
        [],
        ["score_summary", "income_ratios", "derived_features"],
        {"imputer__flag_threshold": 1.0},  # a missing share is never above 1, so no flags
    ),
    "no_feature_selection": ([], ["feature_selector"], {}),
    "no_bureau_scores": (SCORE_COLS, ["score_summary"], {}),
    "no_uncertain_columns": (UNCERTAIN_COLUMNS, [], {}),
}


def final_model_params(selection_path=SELECTION_PATH):
    """The final XGBoost parameters and its class-weight factor."""
    params = dict(json.loads(Path(selection_path).read_text(encoding="utf-8"))["params"])
    return params, params.pop("spw_factor")


def build_variant(variant, model):
    """(columns to drop from the input, unfitted pipeline) for one variant."""
    drop_columns, steps_off, step_params = VARIANTS[variant]
    pipeline = build_pipeline(model)
    pipeline.set_params(**{step: "passthrough" for step in steps_off}, **step_params)
    return list(drop_columns), pipeline


def cross_validate_variant(variant, X, y, params, spw_factor):
    """Per-fold ROC-AUC and PR-AUC on the shared folds."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_train = y.iloc[train_idx]
        drop_columns, pipeline = build_variant(variant, build_xgboost(scale_pos_weight=spw_for(y_train, spw_factor), **params))
        X_used = X.drop(columns=drop_columns)
        pipeline.fit(X_used.iloc[train_idx], y_train)
        proba = pipeline.predict_proba(X_used.iloc[val_idx])[:, 1]
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "roc_auc": roc_auc_score(y.iloc[val_idx], proba),
                "pr_auc": average_precision_score(y.iloc[val_idx], proba),
                "features_used": pipeline[:-1].transform(X_used.iloc[val_idx[:5]]).shape[1],
            }
        )
        print(f"{variant} fold {fold}: ROC-AUC {rows[-1]['roc_auc']:.4f}, PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean and std per variant, and the paired per-fold change against the full pipeline."""
    full = folds[folds["variant"] == "full"].set_index("fold")
    rows = []
    for variant, part in folds.groupby("variant", sort=False):
        part = part.set_index("fold")
        row = {"variant": variant, "features_used": int(part["features_used"].iloc[0])}
        for metric in ("roc_auc", "pr_auc"):
            change = part[metric] - full[metric]
            row.update(
                {
                    f"{metric}_mean": part[metric].mean(),
                    f"{metric}_std": part[metric].std(ddof=0),
                    f"{metric}_change": change.mean(),
                    f"{metric}_change_std": change.std(ddof=0),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_ablation(results_path=RESULTS_PATH):
    from src.data.data_cleaning import clean_and_split

    train_df, _ = clean_and_split()  # held-out test rows are discarded here
    X, y = train_df.drop(columns=[TARGET]), train_df[TARGET]
    params, spw_factor = final_model_params()

    folds = pd.concat([cross_validate_variant(v, X, y, params, spw_factor) for v in VARIANTS], ignore_index=True)
    summary = summarise(folds)
    summary.to_csv(results_path, index=False)
    return summary


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run_ablation().round(4).to_string(index=False))
