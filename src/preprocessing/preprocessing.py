"""Fit-on-train preprocessing: score summaries, missing-value imputation,
income-ratio features, outlier treatment, categorical encoding. These steps
run after the train/test split (unlike src/data/data_cleaning.py) because
they learn statistics (medians, encoded columns) from the training data -
see notebooks/03_data_preprocessing.ipynb sections 6-10, which this module
mirrors.
"""

import numpy as np
import pandas as pd

from src.config import (
    CAR_AGE_COL,
    COLS_RESERVED_FOR_FEATURE_ENGINEERING,
    MISSING_FLAG_THRESHOLD,
    SCORE_COLS,
    TARGET,
)


def add_score_summary_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """score_mean/score_min/scores_available, computed before imputation so
    they reflect the genuine missingness pattern in Score_Source_1/2/3."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for d in (train_df, test_df):
        d["score_mean"] = d[SCORE_COLS].mean(axis=1, skipna=True)
        d["score_min"] = d[SCORE_COLS].min(axis=1, skipna=True)
        d["scores_available"] = d[SCORE_COLS].notna().sum(axis=1)
    return train_df, test_df


def impute_missing_values(
    train_df: pd.DataFrame, test_df: pd.DataFrame, flag_threshold: float = MISSING_FLAG_THRESHOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numeric -> median (fit on train), with a '_was_missing' flag added
    when the missing rate exceeds flag_threshold (car_age excluded, since
    has_car_age already serves that purpose, and it's filled with 0, not
    the median). Categorical -> 'Missing' category.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols_missing = [
        c
        for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in [TARGET, "has_car_age"] and train_df[c].isna().sum() > 0
    ]
    categorical_cols_missing = [
        c for c in train_df.select_dtypes(include=["object", "string"]).columns if train_df[c].isna().sum() > 0
    ]

    for col in numeric_cols_missing:
        if col != CAR_AGE_COL and train_df[col].isna().mean() > flag_threshold:
            flag_col = col + "_was_missing"
            train_df[flag_col] = train_df[col].isna().astype(int)
            test_df[flag_col] = test_df[col].isna().astype(int)

        fill_value = 0 if col == CAR_AGE_COL else train_df[col].median()
        train_df[col] = train_df[col].fillna(fill_value)
        test_df[col] = test_df[col].fillna(fill_value)

    for col in categorical_cols_missing:
        train_df[col] = train_df[col].fillna("Missing")
        test_df[col] = test_df[col].fillna("Missing")

    return train_df, test_df


def add_income_ratio_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """credit_to_income/annuity_to_income/income_per_member, computed before
    the Client_Income log-transform so they use genuine dollar-scale income."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for d in (train_df, test_df):
        d["credit_to_income"] = d["Credit_Amount"] / d["Client_Income"]
        d["annuity_to_income"] = d["Loan_Annuity"] / d["Client_Income"]
        d["income_per_member"] = d["Client_Income"] / d["Client_Family_Members"].clip(lower=1)
    return train_df, test_df


def log_transform_income(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """log1p(Client_Income) - corrects extreme right skew."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["Client_Income"] = np.log1p(train_df["Client_Income"])
    test_df["Client_Income"] = np.log1p(test_df["Client_Income"])
    return train_df, test_df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exclude_cols: list[str] = COLS_RESERVED_FOR_FEATURE_ENGINEERING,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode object/string columns not reserved for feature
    engineering, align train/test columns, cast dummy bools to int."""
    train_df = train_df.copy()
    test_df = test_df.copy()

    cols_to_encode = [c for c in train_df.select_dtypes(include=["object", "string"]).columns if c not in exclude_cols]

    train_df = pd.get_dummies(train_df, columns=cols_to_encode, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=cols_to_encode, drop_first=True)

    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    dummy_cols = [c for c in train_df.columns if train_df[c].dtype == bool]
    train_df[dummy_cols] = train_df[dummy_cols].astype(int)
    test_df[dummy_cols] = test_df[dummy_cols].astype(int)

    return train_df, test_df


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper chaining every fit-on-train step, in order."""
    train_df, test_df = add_score_summary_features(train_df, test_df)
    train_df, test_df = impute_missing_values(train_df, test_df)
    train_df, test_df = add_income_ratio_features(train_df, test_df)
    train_df, test_df = log_transform_income(train_df, test_df)
    train_df, test_df = encode_categoricals(train_df, test_df)
    return train_df, test_df
