"""Gate 1 tests: the pipeline must build, fit on train, and correctly
transform/predict for a single raw row containing a junk token, with no
leakage (ID/Default never inputs, sentinel never reaches the scaler raw).

Uses a 5,000-row sample of the training split for test speed - this
checks mechanics/correctness, not production-scale statistics.
"""

import os
import sys

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config
from src.pipeline import build_pipeline


@pytest.fixture(scope="module")
def fitted_pipeline():
    train_path = os.path.join("data", "processed", "train.csv")
    train_df = pd.read_csv(train_path, low_memory=False)
    sample = train_df.sample(n=5000, random_state=config.RANDOM_STATE)

    X = sample.drop(columns=[config.TARGET])
    y = sample[config.TARGET]

    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(X, y)
    return pipe, X


@pytest.fixture
def junk_row(fitted_pipeline):
    _, X = fitted_pipeline
    row = X.iloc[[0]].to_dict("records")[0]
    row["Client_Income"] = "$"  # junk token
    row["Employed_Days"] = config.SENTINEL_VALUE  # sentinel
    row["Client_Gender"] = "XNA"  # disguised missing
    row["Type_Organization"] = "A Brand New Employer Never Seen Before"  # unseen category
    return pd.DataFrame([row])


def test_pipeline_fits_on_train(fitted_pipeline):
    pipe, _ = fitted_pipeline
    assert pipe is not None
    assert "model" in pipe.named_steps


def test_transforms_single_raw_row_with_junk_token(fitted_pipeline, junk_row):
    pipe, _ = fitted_pipeline
    preprocessed = pipe[:-1].transform(junk_row)
    assert preprocessed.shape[0] == 1
    assert preprocessed.isnull().sum().sum() == 0


def test_id_and_target_never_in_model_inputs(fitted_pipeline, junk_row):
    pipe, _ = fitted_pipeline
    preprocessed = pipe[:-1].transform(junk_row)
    assert config.ID_COL not in preprocessed.columns
    assert config.TARGET not in preprocessed.columns


def test_mobile_tag_dropped_by_selector(fitted_pipeline):
    pipe, _ = fitted_pipeline
    assert "Mobile_Tag" not in pipe.named_steps["selector"].keep_cols_


def test_sentinel_never_reaches_scaler_raw(fitted_pipeline, junk_row):
    pipe, _ = fitted_pipeline
    preprocessed = pipe[:-1].transform(junk_row)
    assert preprocessed["Is_Retired_Or_Unemployed"].iloc[0] == 1
    # After imputation + scaling, Employed_Days should be a small z-score,
    # never the raw sentinel (365243) or a z-score blown up by it
    assert abs(preprocessed["Employed_Days"].iloc[0]) < 5


def test_unseen_category_mapped_to_other(fitted_pipeline, junk_row):
    pipe, _ = fitted_pipeline
    grouper = pipe.named_steps["rare_category_grouper"]
    cleaned = pipe.named_steps["cleaner"].transform(junk_row)
    engineered = pipe.named_steps["feature_engineer"].transform(cleaned)
    grouped = grouper.transform(engineered)
    assert grouped["Type_Organization"].iloc[0] == "Other"


def test_predicts_valid_probability_for_single_row(fitted_pipeline, junk_row):
    pipe, _ = fitted_pipeline
    proba = pipe.predict_proba(junk_row)[:, 1][0]
    assert 0.0 <= proba <= 1.0


def test_no_row_count_change_through_preprocessing(fitted_pipeline):
    pipe, X = fitted_pipeline
    preprocessed = pipe[:-1].transform(X)
    assert preprocessed.shape[0] == X.shape[0]
