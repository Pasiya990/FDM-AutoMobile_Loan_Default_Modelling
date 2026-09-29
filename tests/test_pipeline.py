"""Gate 1 tests: the row-level cleaning (src/data/data_cleaning.py), the
fit-on-train preprocessing (src/preprocessing/preprocessing.py) and the
feature engineering (src/preprocessing/feature_engineering.py) must chain
together correctly on the real raw data, drop the identifier/leakage-prone
columns, leave no missing values, and preserve row counts - mirroring the
checks notebooks 03 and 04 already ran by hand.

Runs on the full raw dataset (~6s total: dedup + split + impute + encode +
an RF importance fit) rather than a sample, so the numbers below are the
same ones reported in notebooks 03/04 and the team implementation plan.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.data_cleaning import (
    CAR_AGE_COL,
    HAS_CAR_AGE_COL,
    ID_COL,
    SENTINEL_FLAG_COL,
    TARGET,
    clean_and_split,
)
from src.preprocessing.feature_engineering import (
    COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    engineer_features,
)
from src.preprocessing.preprocessing import preprocess


@pytest.fixture(scope="module")
def cleaned_split():
    return clean_and_split()


@pytest.fixture(scope="module")
def preprocessed(cleaned_split):
    train_df, test_df = cleaned_split
    return preprocess(train_df, test_df)


@pytest.fixture(scope="module")
def engineered(preprocessed):
    train_df, test_df = preprocessed
    train_df, test_df, importances = engineer_features(train_df, test_df)
    return train_df, test_df, importances


def test_clean_and_split_reproduces_known_shape(cleaned_split):
    train_df, test_df = cleaned_split
    # 121,856 raw rows - 2,640 exact duplicates (ignoring ID) = 119,216
    assert train_df.shape[0] + test_df.shape[0] == 119216
    assert train_df.shape[0] == 95372
    assert test_df.shape[0] == 23844


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


def test_no_missing_values_after_preprocessing(preprocessed):
    train_df, test_df = preprocessed
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0


def test_row_counts_preserved_through_preprocessing(cleaned_split, preprocessed):
    raw_train_df, raw_test_df = cleaned_split
    train_df, test_df = preprocessed
    assert train_df.shape[0] == raw_train_df.shape[0]
    assert test_df.shape[0] == raw_test_df.shape[0]


def test_train_test_columns_aligned_after_preprocessing(preprocessed):
    train_df, test_df = preprocessed
    assert list(train_df.columns) == list(test_df.columns)


def test_row_counts_preserved_through_feature_engineering(preprocessed, engineered):
    pre_train_df, pre_test_df = preprocessed
    train_df, test_df, _ = engineered
    assert train_df.shape[0] == pre_train_df.shape[0]
    assert test_df.shape[0] == pre_test_df.shape[0]


def test_no_missing_values_after_feature_engineering(engineered):
    train_df, test_df, _ = engineered
    assert train_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0


def test_reserved_categorical_columns_encoded_away(engineered):
    train_df, _, _ = engineered
    for col in COLS_RESERVED_FOR_FEATURE_ENGINEERING:
        assert col not in train_df.columns


def test_redundant_raw_columns_dropped(engineered):
    train_df, _, _ = engineered
    assert "Age_Days" not in train_df.columns
    assert "Loan_Annuity" not in train_df.columns


def test_train_test_columns_aligned_after_feature_engineering(engineered):
    train_df, test_df, _ = engineered
    assert list(train_df.columns) == list(test_df.columns)


def test_target_separable_without_leaking_id(engineered):
    train_df, _, _ = engineered
    X = train_df.drop(columns=[TARGET])
    assert TARGET not in X.columns
    assert ID_COL not in X.columns


def test_strong_predictors_survive_feature_selection(engineered):
    # score_mean/score_min are the strongest single predictors (F8 in the
    # implementation plan) - selection must not drop them
    train_df, _, importances = engineered
    assert "score_mean" in train_df.columns
    assert "score_min" in train_df.columns
    assert importances.index[0] == "score_mean"


def test_scaled_continuous_features_have_zero_mean(engineered):
    train_df, _, _ = engineered
    assert abs(train_df["score_mean"].mean()) < 1e-6
