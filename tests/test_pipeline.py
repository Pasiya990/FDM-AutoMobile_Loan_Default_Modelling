"""Gate 1 tests: the fitted sklearn.Pipeline (src/preprocessing/feature_engineering.py's
build_pipeline()) must fit on train, transform train/test consistently, and -
critically - correctly transform a single new raw application row using only
what it learned at fit time, with no train_df required. This is the property
the fit/transform refactor exists for: it's what lets the backend (Stage 9)
call pipe.predict_proba() on one live application.

Runs on the full raw dataset (~10s: dedup + split + impute + encode + an RF
importance fit + a LogisticRegression fit), so the numbers below are the
same ones reported in notebooks 03/04 and verified during the refactor.
"""

import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CAR_AGE_COL, COLS_RESERVED_FOR_FEATURE_ENGINEERING, HAS_CAR_AGE_COL, ID_COL, SENTINEL_FLAG_COL, TARGET
from src.data.data_cleaning import clean_and_split
from src.preprocessing.feature_engineering import build_pipeline


@pytest.fixture(scope="module")
def cleaned_split():
    return clean_and_split()


@pytest.fixture(scope="module")
def fitted_pipeline(cleaned_split):
    train_df, test_df = cleaned_split
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    pipe = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test


def test_clean_and_split_reproduces_known_shape(cleaned_split):
    train_df, test_df = cleaned_split
    # 121,856 raw rows - 2,640 exact duplicates (ignoring ID)
    # - 12,020 near-duplicate applicant copies = 107,196
    assert train_df.shape[0] + test_df.shape[0] == 107196
    assert train_df.shape[0] == 85756
    assert test_df.shape[0] == 21440


def test_split_is_stratified(cleaned_split):
    train_df, test_df = cleaned_split
    assert round(train_df[TARGET].mean(), 2) == 0.08
    assert round(test_df[TARGET].mean(), 2) == 0.08


def test_id_column_dropped(cleaned_split):
    train_df, test_df = cleaned_split
    assert ID_COL not in train_df.columns
    assert ID_COL not in test_df.columns


def test_car_age_and_sentinel_columns_present(cleaned_split):
    train_df, _ = cleaned_split
    assert CAR_AGE_COL in train_df.columns
    assert HAS_CAR_AGE_COL in train_df.columns
    assert SENTINEL_FLAG_COL in train_df.columns
    assert "Own_House_Age" not in train_df.columns


def test_pipeline_fits_on_train(fitted_pipeline):
    pipe, *_ = fitted_pipeline
    assert pipe is not None
    assert "model" in pipe.named_steps


def test_no_missing_values_after_full_pipeline(fitted_pipeline):
    pipe, X_train, X_test, *_ = fitted_pipeline
    assert pipe[:-1].transform(X_train).isnull().sum().sum() == 0
    assert pipe[:-1].transform(X_test).isnull().sum().sum() == 0


def test_row_counts_preserved_through_pipeline(fitted_pipeline):
    pipe, X_train, X_test, *_ = fitted_pipeline
    assert pipe[:-1].transform(X_train).shape[0] == X_train.shape[0]
    assert pipe[:-1].transform(X_test).shape[0] == X_test.shape[0]


def test_train_test_columns_aligned(fitted_pipeline):
    pipe, X_train, X_test, *_ = fitted_pipeline
    train_cols = pipe[:-1].transform(X_train).columns
    test_cols = pipe[:-1].transform(X_test).columns
    assert list(train_cols) == list(test_cols)


def test_target_and_id_never_in_transformed_features(fitted_pipeline):
    pipe, X_train, *_ = fitted_pipeline
    transformed = pipe[:-1].transform(X_train)
    assert TARGET not in transformed.columns
    assert ID_COL not in transformed.columns


def test_reserved_categorical_columns_encoded_away(fitted_pipeline):
    pipe, X_train, *_ = fitted_pipeline
    transformed = pipe[:-1].transform(X_train)
    for col in COLS_RESERVED_FOR_FEATURE_ENGINEERING:
        assert col not in transformed.columns


def test_strong_predictors_survive_feature_selection(fitted_pipeline):
    # score_mean/score_min are the strongest single predictors (F8 in the
    # implementation plan) - selection must not drop them
    pipe, X_train, *_ = fitted_pipeline
    transformed = pipe[:-1].transform(X_train)
    importances = pipe.named_steps["feature_selector"].importances_
    assert "score_mean" in transformed.columns
    assert "score_min" in transformed.columns
    assert importances.index[0] == "score_mean"


def test_scaled_continuous_features_have_zero_mean(fitted_pipeline):
    pipe, X_train, *_ = fitted_pipeline
    transformed = pipe[:-1].transform(X_train)
    assert abs(transformed["score_mean"].mean()) < 1e-6


def test_single_new_row_matches_batch_columns(fitted_pipeline):
    pipe, _, X_test, *_ = fitted_pipeline
    batch = pipe[:-1].transform(X_test)
    single_row = X_test.iloc[[0]]
    single = pipe[:-1].transform(single_row)
    assert single.shape[0] == 1
    assert list(single.columns) == list(batch.columns)
    assert single.isnull().sum().sum() == 0


def test_single_rows_are_transformed_exactly_like_the_batch(fitted_pipeline):
    # The backend scores one applicant at a time, so a row must get the same
    # features alone as inside a batch. One-hot encoding once dropped a row's
    # own category when it was scored alone, setting every dummy to 0.
    pipe, _, X_test, *_ = fitted_pipeline
    rows = X_test.iloc[:20]
    batch = pipe[:-1].transform(rows)
    for i in range(len(rows)):
        single = pipe[:-1].transform(rows.iloc[[i]])
        np.testing.assert_allclose(
            single.to_numpy(dtype=float), batch.iloc[[i]].to_numpy(dtype=float), err_msg=f"row {i}"
        )


def test_single_new_row_uses_train_statistics_not_its_own(fitted_pipeline):
    # The whole point of fit/transform: a lone row has no data to compute a
    # median or a scaler mean from - it must reuse what fit() learned from
    # train, not silently recompute (or crash) on a single-row input.
    # Checked right after the imputer step, before the scaler (further down
    # the pipeline) transforms the value again.
    pipe, X_train, X_test, *_ = fitted_pipeline
    single_row = X_test.iloc[[0]].copy()
    single_row["Score_Source_1"] = None  # force a missing value

    through_imputer = [name for name, _ in pipe.steps].index("imputer") + 1
    transformed = pipe[:through_imputer].transform(single_row)
    imputer = pipe.named_steps["imputer"]
    assert transformed["Score_Source_1"].iloc[0] == imputer.fill_values_["Score_Source_1"]


def test_unseen_category_handled_without_crashing(fitted_pipeline):
    pipe, _, X_test, *_ = fitted_pipeline
    single_row = X_test.iloc[[0]].copy()
    single_row["Type_Organization"] = "A Brand New Employer Never Seen Before"

    transformed = pipe[:-1].transform(single_row)
    assert transformed.shape[0] == 1
    assert transformed.isnull().sum().sum() == 0
    # falls back to the reference category: none of the Type_Organization
    # dummy columns fire for it
    type_org_cols = [c for c in transformed.columns if c.startswith("Type_Organization_")]
    assert transformed[type_org_cols].sum(axis=1).iloc[0] == 0


def test_predicts_valid_probability_for_single_row(fitted_pipeline):
    pipe, _, X_test, *_ = fitted_pipeline
    single_row = X_test.iloc[[0]]
    proba = pipe.predict_proba(single_row)[:, 1][0]
    assert 0.0 <= proba <= 1.0


@pytest.mark.parametrize("column", ["Client_Income", "Loan_Annuity"])
def test_zero_denominator_is_imputed_not_infinite(fitted_pipeline, column):
    # These columns are divided by in the ratio features; a zero must be
    # treated as missing and filled, not turned into infinity.
    pipe, _, X_test, *_ = fitted_pipeline
    single_row = X_test.iloc[[0]].copy()
    single_row[column] = 0

    transformed = pipe[:-1].transform(single_row)
    assert np.isfinite(transformed.to_numpy(dtype=float)).all()
    assert 0.0 <= pipe.predict_proba(single_row)[:, 1][0] <= 1.0


def test_missing_required_columns_raise_clear_error(fitted_pipeline):
    pipe, _, X_test, *_ = fitted_pipeline
    single_row = X_test.iloc[[0]].drop(columns=["Child_Count", "Client_Income"])

    with pytest.raises(ValueError, match="Missing required input columns") as error:
        pipe.predict_proba(single_row)
    assert "Child_Count" in str(error.value)
    assert "Client_Income" in str(error.value)


def test_missing_value_in_column_complete_in_training_is_filled(fitted_pipeline):
    # Homephone_Tag has no missing values in training; a new application
    # with it missing must still be filled before reaching the model.
    pipe, X_train, X_test, *_ = fitted_pipeline
    assert X_train["Homephone_Tag"].notna().all()
    single_row = X_test.iloc[[0]].copy()
    single_row["Homephone_Tag"] = np.nan

    assert pipe[:-1].transform(single_row).isnull().sum().sum() == 0
    assert 0.0 <= pipe.predict_proba(single_row)[:, 1][0] <= 1.0
