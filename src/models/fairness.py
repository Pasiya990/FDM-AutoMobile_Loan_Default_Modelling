"""Fairness and subgroup check for the final model (plan M3-O4; proposal
commitment to check prediction differences across groups).

For each group of applicants (gender, age band, marital status, education,
income type) it reports, at the operating threshold:
    recall               share of the group's defaulters that are flagged
    false_positive_rate  share of the group's good applicants that are flagged
                         (sent to extra review without need)
    precision            share of flagged applicants who default
    share_flagged, default_rate and mean_score (the score is a ranking, so
                         mean_score is compared between groups, not with the rate)
Per attribute, the gap is the largest minus the smallest value across groups
with enough applicants; gaps above GAP_LIMIT (10 percentage points) are flagged.

Two models are compared on the same 5-fold CV of the training split (the held-out
test set is not used):
    final               the deployed model's out-of-fold predictions
                        (experiments/results/final_oof_predictions.csv)
    without_sensitive   the same configuration without gender, marital status
                        and age (feature set S2), with its own 60%-recall
                        threshold. Gender and marital status are dropped; age is
                        set to one constant value for everyone, because the
                        pipeline derives Age_Years from it.

Usage (from the project root): python -m src.models.fairness
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE, TARGET
from src.models.ablation import final_model_params
from src.models.fair_comparison import spw_for, threshold_for_recall
from src.models.select_final import OOF_PATH
from src.models.xgboost_model import build_xgboost
from src.preprocessing.feature_engineering import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
BY_GROUP_PATH = RESULTS_DIR / "fairness_by_group.csv"
SUMMARY_PATH = RESULTS_DIR / "fairness_summary.csv"

CV_FOLDS = 5
GAP_LIMIT = 0.10
MIN_APPLICANTS, MIN_DEFAULTERS = 500, 50  # smaller groups are reported but left out of the gaps
AGE_BANDS = [21, 31, 41, 51, 61, 70]
AGE_LABELS = ["21-30", "31-40", "41-50", "51-60", "61-69"]
MARITAL = {"D": "Divorced", "M": "Married", "S": "Single", "W": "Widowed"}
SENSITIVE_DROPPED = ["Client_Gender", "Client_Marital_Status"]


def group_columns(X: pd.DataFrame) -> pd.DataFrame:
    """The attributes to compare, with readable group names ('Not given' for blanks)."""
    age = pd.cut(X["Age_Days"] / 365.25, bins=AGE_BANDS, right=False, labels=AGE_LABELS)
    groups = pd.DataFrame(
        {
            "gender": X["Client_Gender"],
            "age_band": age.astype("object"),
            "marital_status": X["Client_Marital_Status"].map(MARITAL),
            "education": X["Client_Education"],
            "income_type": X["Client_Income_Type"],
        },
        index=X.index,
    )
    return groups.astype("object").where(groups.notna(), "Not given")


def group_metrics(y, scores, threshold, groups: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per (attribute, group) with the error rates at `threshold`."""
    data = pd.DataFrame({"y": np.asarray(y), "score": np.asarray(scores)}, index=groups.index)
    data["flagged"] = data["score"] >= threshold
    rows = []
    for attribute in groups.columns:
        for group, part in data.groupby(groups[attribute]):
            defaulters, good = part[part["y"] == 1], part[part["y"] == 0]
            rows.append(
                {
                    "model": model,
                    "attribute": attribute,
                    "group": group,
                    "applicants": len(part),
                    "defaulters": len(defaulters),
                    "default_rate": part["y"].mean(),
                    "mean_score": part["score"].mean(),
                    "share_flagged": part["flagged"].mean(),
                    "recall": defaulters["flagged"].mean() if len(defaulters) else np.nan,
                    "false_positive_rate": good["flagged"].mean() if len(good) else np.nan,
                    "precision": part.loc[part["flagged"], "y"].mean() if part["flagged"].any() else np.nan,
                    "enough_data": len(part) >= MIN_APPLICANTS and len(defaulters) >= MIN_DEFAULTERS,
                }
            )
    return pd.DataFrame(rows)


def gaps(by_group: pd.DataFrame) -> pd.DataFrame:
    """Largest minus smallest rate per model and attribute, over groups with enough data."""
    rows = []
    reliable = by_group[by_group["enough_data"]]
    for (model, attribute), part in reliable.groupby(["model", "attribute"], sort=False):
        row = {"model": model, "attribute": attribute, "groups_compared": len(part)}
        for metric in ("recall", "false_positive_rate", "precision", "share_flagged"):
            low, high = part.loc[part[metric].idxmin()], part.loc[part[metric].idxmax()]
            row[f"{metric}_gap"] = high[metric] - low[metric]
            row[f"{metric}_range"] = f"{low['group']} {low[metric]:.2f} - {high['group']} {high[metric]:.2f}"
        row["gap_above_limit"] = max(row["recall_gap"], row["false_positive_rate_gap"]) > GAP_LIMIT
        rows.append(row)
    return pd.DataFrame(rows)


def without_sensitive(X: pd.DataFrame) -> pd.DataFrame:
    """Feature set S2: no gender or marital status, and the same age for everyone."""
    X = X.drop(columns=SENSITIVE_DROPPED)
    X["Age_Days"] = X["Age_Days"].median()
    return X


def out_of_fold_scores(X, y, params, spw_factor):
    """Out-of-fold scores of the final configuration on `X` (the shared folds)."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        y_train = y.iloc[train_idx]
        pipeline = build_pipeline(build_xgboost(scale_pos_weight=spw_for(y_train, spw_factor), **params))
        pipeline.fit(X.iloc[train_idx], y_train)
        scores[val_idx] = pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        print(f"without_sensitive fold {fold} done", flush=True)
    return scores


def run_fairness():
    from src.data.data_cleaning import clean_and_split

    train_df, _ = clean_and_split()  # held-out test rows are discarded here
    X, y = train_df.drop(columns=[TARGET]), train_df[TARGET]
    groups = group_columns(X)

    oof = pd.read_csv(OOF_PATH)
    if not (oof["y_true"].to_numpy() == y.to_numpy()).all():
        raise ValueError("final_oof_predictions.csv does not match the training split; rerun select_final")
    final_scores = oof["xgboost_tuned"].to_numpy()

    params, spw_factor = final_model_params()
    s2_scores = out_of_fold_scores(without_sensitive(X), y, params, spw_factor)

    by_group, overall = [], []
    for model, scores in (("final", final_scores), ("without_sensitive", s2_scores)):
        threshold = threshold_for_recall(y, scores)
        by_group.append(group_metrics(y, scores, threshold, groups, model))
        overall.append(
            {"model": model, "threshold": threshold, "roc_auc": roc_auc_score(y, scores),
             "pr_auc": average_precision_score(y, scores)}
        )
    by_group = pd.concat(by_group, ignore_index=True)
    summary = gaps(by_group).merge(pd.DataFrame(overall), on="model")
    by_group.to_csv(BY_GROUP_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    return by_group, summary


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    by_group, summary = run_fairness()
    columns = ["model", "attribute", "group", "applicants", "default_rate", "mean_score", "share_flagged",
               "recall", "false_positive_rate", "precision", "enough_data"]
    print(by_group[columns].round(3).to_string(index=False))
    print(summary.drop(columns=[c for c in summary if c.endswith("_range")]).round(4).to_string(index=False))
    print(summary[["model", "attribute"] + [c for c in summary if c.endswith("_range")]].to_string(index=False))
