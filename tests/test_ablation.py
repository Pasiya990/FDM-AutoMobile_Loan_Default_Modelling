"""Tests for the ablation variants: each one really removes what it claims to,
checked by fitting on a small sample of the training split."""

import os
import sys

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RAW_DATA_PATH, SCORE_COLS, TARGET
from src.models.ablation import UNCERTAIN_COLUMNS, VARIANTS, build_variant, summarise

ENGINEERED = ["score_mean", "score_min", "scores_available", "credit_to_income", "annuity_to_income",
              "income_per_member", "Age_Years", "Credit_Loan_Ratio", "has_children", "contact_count"]


@pytest.fixture(scope="module")
def sample():
    if not RAW_DATA_PATH.exists():
        pytest.skip("raw data not on this machine")
    from src.data.data_cleaning import clean_and_split

    train_df, _ = clean_and_split()
    part = train_df.sample(3000, random_state=0)
    return part.drop(columns=[TARGET]), part[TARGET]


def _features(variant, sample):
    X, y = sample
    drop_columns, pipeline = build_variant(variant, LogisticRegression(max_iter=1000))
    pipeline.set_params(feature_selector="passthrough")  # keep every column visible; the scaler drops none
    X_used = X.drop(columns=drop_columns)
    return list(pipeline.fit(X_used, y)[:-1].transform(X_used).columns)


def test_full_pipeline_has_the_engineered_features(sample):
    features = _features("full", sample)
    assert all(column in features for column in ENGINEERED)
    assert any(column.endswith("_was_missing") for column in features)


def test_no_engineered_features_removes_all_of_them(sample):
    features = _features("no_engineered_features", sample)
    assert not any(column in features for column in ENGINEERED)
    assert not any(column.endswith("_was_missing") for column in features)
    assert "Age_Days" in features and "Loan_Annuity" in features  # raw columns kept


def test_no_bureau_scores_removes_scores_and_summaries(sample):
    features = _features("no_bureau_scores", sample)
    assert not any(column.startswith("Score_Source") or column.startswith("score") for column in features)


def test_no_uncertain_columns_removes_them(sample):
    features = _features("no_uncertain_columns", sample)
    assert not any(column.startswith(name) for name in UNCERTAIN_COLUMNS for column in features)


def test_no_feature_selection_switches_off_only_the_selector():
    _, pipeline = build_variant("no_feature_selection", LogisticRegression())
    assert pipeline.named_steps["feature_selector"] == "passthrough"
    assert pipeline.named_steps["scaler"] != "passthrough"


def test_summary_reports_paired_change_against_full():
    folds = pd.DataFrame(
        {
            "variant": ["full"] * 2 + ["other"] * 2,
            "fold": [0, 1, 0, 1],
            "roc_auc": [0.75, 0.73, 0.70, 0.70],
            "pr_auc": [0.25, 0.23, 0.20, 0.20],
            "features_used": [50, 50, 40, 40],
        }
    )
    summary = summarise(folds).set_index("variant")

    assert summary.loc["full", "roc_auc_change"] == 0
    assert summary.loc["other", "roc_auc_change"] == pytest.approx(-0.04)
    assert summary.loc["other", "roc_auc_change_std"] == pytest.approx(0.01)
    assert list(summary.index) == ["full", "other"]
    assert set(VARIANTS) >= {"full", "no_engineered_features", "no_feature_selection"}
